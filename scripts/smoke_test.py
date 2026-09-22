"""Load the tracked fixture and verify the shared setup assumptions."""

from __future__ import annotations

import csv
from pathlib import Path


REQUIRED_COLUMNS = {
    "event_id",
    "timestamp",
    "source_user",
    "destination_user",
    "source_host",
    "destination_host",
    "result",
    "protocol",
}


def load_events(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"Fixture is missing columns: {sorted(missing)}")
        events = list(reader)

    if not events:
        raise ValueError("Fixture contains no events")
    return events


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    fixture_path = repository_root / "data" / "fixtures" / "authentication_fixture.csv"
    events = load_events(fixture_path)

    timestamps = [int(event["timestamp"]) for event in events]
    results = {event["result"] for event in events}
    destinations = {event["destination_host"] for event in events}
    timestamp_counts = {timestamp: timestamps.count(timestamp) for timestamp in set(timestamps)}

    checks = {
        "fixture has successful and failed authentications": {"success", "failure"} <= results,
        "fixture includes a new destination": len(destinations) >= 3,
        "fixture includes repeated timestamps": max(timestamp_counts.values()) >= 2,
        "events are chronologically ordered": timestamps == sorted(timestamps),
    }

    failed = [name for name, passed in checks.items() if not passed]
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    print(f"Loaded {len(events)} fixture events from {fixture_path.relative_to(repository_root)}")
    if failed:
        raise SystemExit(f"Smoke check failed: {', '.join(failed)}")
    print("Smoke check passed.")
    return 0


if __name__ == "__main__":
    main()

