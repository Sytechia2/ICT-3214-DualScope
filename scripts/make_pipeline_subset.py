"""Cut the fixed LANL subset that the pipeline runner's sample profile processes.

The subset keeps every authentication event of a small, seeded set of users from
dataset days 1-9 only. Days 17-30 (the held-out test days) are never read.

Selection rules (seed 0 by default; the same rules always give the same files):

1. Red-team users: every user with at least one red-team row on days 1-9, unless
   their total days 1-9 event count is above --redteam-cap (default 100,000).
   Users above the cap are listed in the manifest as excluded. With the standard
   data this removes U66@DOM1 (about 1.6M events on its own).
2. Human users: --humans (default 200) users drawn at random from accounts with
   no "$" in the name and not ANONYMOUS LOGON, excluding every red-team user (kept
   or excluded) and keeping only users with --human-min to --human-max events (default 20 to 1,500).
3. Machine accounts: --machines (default 10) accounts with a "$" in the name drawn
   at random with --machine-min to --machine-max events.
4. Every event whose acting (source) user is selected is kept, in the original
   file order. Candidate lists are sorted before sampling so the draw does not
   depend on read order.

Outputs go to --output (default data/samples/lanl_pipeline_subset):

    auth_subset.txt.gz   original 9-field lines, original order, gzip with mtime 0
    redteam_subset.txt   original red-team lines for the selected users on days 1-9
    line_map.txt.gz      original auth.txt line number of each subset line
    manifest.json        rules, seed, selected users, per-day counts, sha256 sums
    README.md            short description

By default the script reads the Task 2.2 Parquet output (fast). With --auth it
streams a raw auth.txt or auth.txt.gz instead and produces the same files; that
path reads the file twice (count users, then select) and stops after day 9.
Use --redteam with the raw path to point at redteam.txt.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

SECONDS_PER_DAY = 86_400
LAST_DAY = 9
AUTH_FIELDS = 9
DEFAULT_INGESTED = Path("data/processed/lanl_auth_days_01_30")
DEFAULT_OUTPUT = Path("data/samples/lanl_pipeline_subset")


def day_of(timestamp: int) -> int:
    return ((timestamp - 1) // SECONDS_PER_DAY) + 1


def is_human(user: str) -> bool:
    return "$" not in user and not user.startswith("ANONYMOUS LOGON")


def valid_auth_fields(fields: list[str]) -> bool:
    """Mirror the ingestion acceptance rules so raw and Parquet sources agree."""

    if len(fields) != AUTH_FIELDS or any(value == "" for value in fields[1:]):
        return False
    try:
        return int(fields[0]) >= 1
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Sources. Each yields what the selection and cutting steps need.
# ---------------------------------------------------------------------------


def _open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def raw_auth_lines(path: Path) -> Iterator[tuple[int, str]]:
    """Yield (original line number, line) for valid rows on days 1-LAST_DAY."""

    with _open_text(path) as handle:
        for number, line in enumerate(handle, start=1):
            line = line.rstrip("\r\n")
            fields = line.split(",")
            if not valid_auth_fields(fields):
                continue
            if day_of(int(fields[0])) > LAST_DAY:
                return
            yield number, line


def raw_redteam_lines(path: Path) -> list[str]:
    rows: list[str] = []
    with _open_text(path) as handle:
        for line in handle:
            line = line.rstrip("\r\n")
            fields = line.split(",")
            if len(fields) != 4 or not fields[0].isdigit():
                continue
            if day_of(int(fields[0])) > LAST_DAY:
                break
            rows.append(line)
    return rows


def _parquet_parts(root: Path, day: int) -> list[Path]:
    return sorted((root / "authentication" / "events" / f"dataset_day={day:02d}").glob("part-*.parquet"))


def parquet_user_counts(root: Path) -> Counter:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    counts: Counter = Counter()
    for day in range(1, LAST_DAY + 1):
        for part in _parquet_parts(root, day):
            column = pq.read_table(part, columns=["acting_user"]).column(0)
            for item in pc.value_counts(column).to_pylist():
                counts[item["values"]] += item["counts"]
    return counts


def parquet_selected_lines(root: Path, users: set[str]) -> Iterator[tuple[int, str]]:
    import pyarrow.parquet as pq

    for day in range(1, LAST_DAY + 1):
        for part in _parquet_parts(root, day):
            table = pq.read_table(
                part,
                columns=["acting_user", "source_line", "raw_record"],
                filters=[("acting_user", "in", sorted(users))],
            )
            yield from zip(
                table.column("source_line").to_pylist(), table.column("raw_record").to_pylist()
            )


def parquet_redteam_lines(root: Path) -> list[str]:
    import pyarrow.dataset as ds

    labels = ds.dataset(root / "redteam_labels" / "labels", format="parquet", partitioning="hive")
    table = labels.to_table(columns=["timestamp", "source_line", "raw_record"])
    rows = sorted(zip(table.column("source_line").to_pylist(), table.column("raw_record").to_pylist()))
    selected: list[str] = []
    for _, raw in rows:
        if day_of(int(raw.split(",")[0])) <= LAST_DAY:
            selected.append(raw)
    return selected


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def select_users(
    counts: Counter,
    redteam_users: Iterable[str],
    *,
    seed: int,
    redteam_cap: int,
    humans: int,
    machines: int,
    human_min: int,
    human_max: int,
    machine_min: int,
    machine_max: int,
) -> dict[str, object]:
    redteam = sorted(set(redteam_users))
    kept_redteam = [user for user in redteam if counts.get(user, 0) <= redteam_cap]
    excluded = [
        {
            "user": user,
            "events_days_1_9": counts[user],
            "reason": f"more than {redteam_cap} events on days 1-9 (cap)",
        }
        for user in redteam
        if counts.get(user, 0) > redteam_cap
    ]
    redteam_set = set(redteam)
    human_pool = sorted(
        user
        for user, n in counts.items()
        if is_human(user) and user not in redteam_set and human_min <= n <= human_max
    )
    machine_pool = sorted(
        user
        for user, n in counts.items()
        if "$" in user and user not in redteam_set and machine_min <= n <= machine_max
    )
    rng = random.Random(seed)
    chosen_humans = sorted(rng.sample(human_pool, min(humans, len(human_pool))))
    chosen_machines = sorted(rng.sample(machine_pool, min(machines, len(machine_pool))))
    return {
        "redteam_users": kept_redteam,
        "excluded_redteam_users": excluded,
        "human_users": chosen_humans,
        "machine_users": chosen_machines,
        "human_pool_size": len(human_pool),
        "machine_pool_size": len(machine_pool),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _gzip_writer(path: Path):
    """Gzip writer with a zero timestamp and no stored name so bytes are stable."""

    raw = path.open("wb")
    return raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cut_subset(
    lines: Iterable[tuple[int, str]], output: Path
) -> tuple[Counter, int]:
    """Write auth_subset.txt.gz and line_map.txt.gz; return per-day counts and total."""

    per_day: Counter = Counter()
    total = 0
    auth_raw, auth_gz = _gzip_writer(output / "auth_subset.txt.gz")
    map_raw, map_gz = _gzip_writer(output / "line_map.txt.gz")
    try:
        for number, line in lines:
            auth_gz.write((line + "\n").encode("utf-8"))
            map_gz.write(f"{number}\n".encode("ascii"))
            per_day[day_of(int(line.split(",", 1)[0]))] += 1
            total += 1
    finally:
        auth_gz.close()
        auth_raw.close()
        map_gz.close()
        map_raw.close()
    return per_day, total


README = """# LANL pipeline subset

A small fixed slice of the LANL authentication data that the pipeline runner's
`sample` profile processes from raw lines to dashboard incidents.

- `auth_subset.txt.gz`: {events:,} original auth.txt lines in original order.
- `line_map.txt.gz`: the original auth.txt line number of each subset line, so
  ingestion keeps true `auth.txt:<line>` evidence references.
- `redteam_subset.txt`: the original red-team lines for the selected users on
  days 1-9 ({redteam_rows} rows).
- `manifest.json`: selection rules, seed, selected users, per-day counts and
  sha256 sums of every file here.

## What is in it

Days 1-9 only. Events of {red} red-team users, {hum} randomly sampled human users
and {mac} randomly sampled machine accounts. {excluded}

Days 17-30 (the test days) are never read. Day 9 was part of the fusion model's
training data, so catches on that day are not a fair accuracy result.

## Scores differ from the full run

Features that use other users' history on the same hosts see only the selected
users here, so scores from the subset differ from the full days 1-30 run. Use it
to check that the pipeline works end to end, not to report accuracy.

## Regenerate

    python scripts/make_pipeline_subset.py

This reads the Task 2.2 Parquet output under {ingested}. Without that output,
stream the raw files instead:

    python scripts/make_pipeline_subset.py --auth path/to/auth.txt.gz --redteam path/to/redteam.txt

Both routes give byte-identical files. Seed: {seed}.
"""


def build(args: argparse.Namespace) -> dict[str, object]:
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    raw_mode = args.auth is not None
    if raw_mode:
        if args.redteam is None:
            raise SystemExit("--redteam is required together with --auth")
        counts: Counter = Counter()
        for _, line in raw_auth_lines(args.auth):
            counts[line.split(",", 2)[1]] += 1
        redteam_lines = raw_redteam_lines(args.redteam)
        source_description = {
            "kind": "raw files",
            "auth": args.auth.name,
            "redteam": args.redteam.name,
        }
    else:
        counts = parquet_user_counts(args.ingested)
        redteam_lines = parquet_redteam_lines(args.ingested)
        source_description = {
            "kind": "Task 2.2 Parquet ingestion of days 1-30 (days 1-9 read)",
            "root": args.ingested.as_posix(),
        }

    redteam_users = [line.split(",")[1] for line in redteam_lines]
    parameters = {
        "seed": args.seed,
        "redteam_cap": args.redteam_cap,
        "humans": args.humans,
        "machines": args.machines,
        "human_min": args.human_min,
        "human_max": args.human_max,
        "machine_min": args.machine_min,
        "machine_max": args.machine_max,
    }
    selection = select_users(counts, redteam_users, **parameters)
    selected = set(selection["redteam_users"]) | set(selection["human_users"]) | set(
        selection["machine_users"]
    )

    if raw_mode:
        lines = ((n, l) for n, l in raw_auth_lines(args.auth) if l.split(",", 2)[1] in selected)
    else:
        lines = parquet_selected_lines(args.ingested, selected)
    per_day, total = cut_subset(lines, output)

    kept_redteam = set(selection["redteam_users"])
    redteam_kept_lines = [line for line in redteam_lines if line.split(",")[1] in kept_redteam]
    (output / "redteam_subset.txt").write_text(
        "".join(line + "\n" for line in redteam_kept_lines), encoding="utf-8", newline="\n"
    )
    redteam_per_day = Counter(day_of(int(line.split(",")[0])) for line in redteam_kept_lines)

    excluded = selection["excluded_redteam_users"]
    excluded_text = (
        "Red-team users above the event cap were left out: "
        + ", ".join(f"{item['user']} ({item['events_days_1_9']:,} events)" for item in excluded)
        + "."
        if excluded
        else "No red-team user was above the event cap."
    )
    (output / "README.md").write_text(
        README.format(
            events=total,
            redteam_rows=len(redteam_kept_lines),
            red=len(selection["redteam_users"]),
            hum=len(selection["human_users"]),
            mac=len(selection["machine_users"]),
            excluded=excluded_text,
            ingested=DEFAULT_INGESTED.as_posix(),
            seed=args.seed,
        ),
        encoding="utf-8",
        newline="\n",
    )

    command = "python scripts/make_pipeline_subset.py" + (
        f" --auth <{args.auth.name}> --redteam <{args.redteam.name}>" if raw_mode else ""
    )
    files = ["auth_subset.txt.gz", "line_map.txt.gz", "redteam_subset.txt", "README.md"]
    manifest = {
        "description": "Fixed LANL subset for the pipeline runner sample profile (days 1-9 only)",
        "command": command,
        "source": source_description,
        "days_included": list(range(1, LAST_DAY + 1)),
        "parameters": parameters,
        "selection_rules": [
            "red-team users with a label row on days 1-9 and at most redteam_cap events on days 1-9",
            "humans: no '$', not ANONYMOUS LOGON, not a red-team user, human_min to human_max events",
            "machines: '$' accounts, not a red-team user, machine_min to machine_max events",
            "candidates sorted, then random.Random(seed).sample; humans drawn before machines",
            "keep every event whose source (acting) user is selected, in original file order",
        ],
        "selected_users": {
            "redteam": selection["redteam_users"],
            "human": selection["human_users"],
            "machine": selection["machine_users"],
        },
        "excluded_redteam_users": excluded,
        "candidate_pool_sizes": {
            "human": selection["human_pool_size"],
            "machine": selection["machine_pool_size"],
        },
        "events_total": total,
        "events_per_day": {str(d): per_day[d] for d in sorted(per_day)},
        "redteam_rows_total": len(redteam_kept_lines),
        "redteam_rows_per_day": {str(d): redteam_per_day[d] for d in sorted(redteam_per_day)},
        "sha256": {name: _sha256(output / name) for name in files},
        "bytes": {name: (output / name).stat().st_size for name in files},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ingested", type=Path, default=DEFAULT_INGESTED,
                        help="Task 2.2 output root (default source)")
    parser.add_argument("--auth", type=Path, help="Raw auth.txt or auth.txt.gz (streaming source)")
    parser.add_argument("--redteam", type=Path, help="Raw redteam.txt (required with --auth)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--redteam-cap", type=int, default=100_000)
    parser.add_argument("--humans", type=int, default=200)
    parser.add_argument("--machines", type=int, default=10)
    parser.add_argument("--human-min", type=int, default=20)
    parser.add_argument("--human-max", type=int, default=1_500)
    parser.add_argument("--machine-min", type=int, default=1_000)
    parser.add_argument("--machine-max", type=int, default=8_000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    manifest = build(parse_args(argv))
    print(
        json.dumps(
            {
                "events_total": manifest["events_total"],
                "events_per_day": manifest["events_per_day"],
                "redteam_rows_per_day": manifest["redteam_rows_per_day"],
                "users": {k: len(v) for k, v in manifest["selected_users"].items()},
                "excluded_redteam_users": manifest["excluded_redteam_users"],
                "bytes": manifest["bytes"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
