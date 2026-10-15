"""End-to-end pipeline runner (Task 8.1).

Runs the stages listed in ``config/pipeline_<profile>.json`` in order, each
into its own folder of the run directory, and keeps ``pipeline_state.json``
with what every stage did. The ``sample`` profile starts from the tracked
subset in ``data/samples/lanl_pipeline_subset`` and the ``full`` profile
reproduces the run of record, reusing verified outputs that already exist.
``docs/pipeline_runner.md`` describes the stages, reuse rules and failure
behaviour.

Stage records. ``status`` is what happened in the latest invocation: ``done``,
``reused`` (an existing verified output was used), ``skipped`` (up to date, not
selected, blocked or not possible) or ``failed``. ``completed`` keeps the last
successful run, with the fingerprint of its inputs, so the next invocation can
skip the stage when nothing changed.

Memory. ``peak_memory_mb`` is the highest resident set size (Windows: working
set) seen by a 0.2 second sampler. A stage that starts command-line scripts
reports the child process trees; an in-process stage reports the runner
process itself, so that figure includes the Python interpreter and the loaded
libraries (it is a peak, not a difference).

A failed investigation stage never blocks the dashboard check; any other
failed stage blocks the stages that need it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import psutil

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "config"
STATE_FILE = "pipeline_state.json"
SMALL_FILE_BYTES = 64 * 1024 * 1024
SAMPLE_INTERVAL_SECONDS = 0.2
DONE, REUSED, SKIPPED, FAILED = "done", "reused", "skipped", "failed"


class StageFailed(RuntimeError):
    """A stage could not finish; the message is shown to the user and saved in the state file."""


class StageSkipped(RuntimeError):
    """A stage cannot run in this invocation (for example the LLM is not enabled); not an error."""


class ConfigError(ValueError):
    """The profile file or the command line is not usable."""


# ─── Small helpers ─────────────────────────────────────────────────────


def rel(path: str | Path) -> str:
    """Repository-relative path (forward slashes) when inside the repository, else absolute."""
    path = Path(path)
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def signature(path: str | Path) -> dict[str, Any]:
    """sha256 for a file up to 64 MB; size and time for a larger one; file count, bytes and latest time for a folder."""
    path = Path(path)
    if path.is_file():
        stat = path.stat()
        if stat.st_size <= SMALL_FILE_BYTES:
            return {"sha256": sha256_file(path), "bytes": stat.st_size}
        return {"bytes": stat.st_size, "mtime": int(stat.st_mtime)}
    if path.is_dir():
        files = total = latest = 0
        for root, _, names in os.walk(path):
            for name in names:
                stat = (Path(root) / name).stat()
                files += 1
                total += stat.st_size
                latest = max(latest, int(stat.st_mtime))
        return {"files": files, "bytes": total, "latest_mtime": latest}
    return {"missing": True}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


# ─── Memory ────────────────────────────────────────────────────────────


class PeakMemory:
    """Samples the resident memory of registered process trees in a thread and keeps the maximum."""

    def __init__(self) -> None:
        self.roots: list[int] = []
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def add_root(self, pid: int) -> None:
        if pid not in self.roots:
            self.roots.append(pid)

    def _sample(self) -> None:
        total = 0
        for pid in list(self.roots):
            try:
                process = psutil.Process(pid)
                total += process.memory_info().rss
                for child in process.children(recursive=True):
                    try:
                        total += child.memory_info().rss
                    except psutil.Error:
                        pass
            except psutil.Error:
                continue
        self.peak_bytes = max(self.peak_bytes, total)

    def _loop(self) -> None:
        while not self._stop.wait(SAMPLE_INTERVAL_SECONDS):
            self._sample()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> float | None:
        self._sample()
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()
        return round(self.peak_bytes / 1_048_576, 1) if self.roots else None


# ─── State ─────────────────────────────────────────────────────────────


class PipelineState:
    """``pipeline_state.json`` of a run directory."""

    def __init__(self, path: Path, profile: str, run_dir: Path) -> None:
        self.path = path
        self.data: dict[str, Any] = {"profile": profile, "run_dir": str(run_dir), "stages": {}}
        if path.is_file():
            try:
                saved = read_json(path)
                if isinstance(saved, dict) and saved.get("profile") == profile and isinstance(saved.get("stages"), dict):
                    self.data = saved
                    self.data["run_dir"] = str(run_dir)
            except (OSError, ValueError):
                pass  # an unreadable or damaged state file means every stage runs again

    @property
    def stages(self) -> dict[str, Any]:
        return self.data["stages"]

    def completed(self, name: str) -> dict[str, Any] | None:
        return self.stages.get(name, {}).get("completed")

    def set(self, name: str, record: dict[str, Any], completed: Any = "keep") -> None:
        """Replace a stage's record. ``completed``: "keep" the earlier successful run, None to drop it, or a new one."""
        if completed == "keep":
            completed = self.completed(name)
        self.stages[name] = {**record, "completed": completed}
        self.data["updated_utc"] = utc_now()
        write_json(self.path, self.data)


# ─── Context and stages ────────────────────────────────────────────────


@dataclass
class Options:
    llm: bool = False
    workers: int | None = None
    auth: Path | None = None
    redteam: Path | None = None


@dataclass
class Outcome:
    """What a stage returns: its status, output paths by name and optional notes."""

    status: str = DONE
    outputs: dict[str, Path] = field(default_factory=dict)
    inputs: dict[str, Path] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)


class Context:
    def __init__(self, config: dict[str, Any], run_dir: Path, options: Options, state: PipelineState, stage: str,
                 memory: PeakMemory, log_file: Any, echo: Callable[[str], None]) -> None:
        self.config, self.run_dir, self.options, self.state, self.stage = config, run_dir, options, state, stage
        self.memory, self._log_file, self._echo = memory, log_file, echo
        self.commands: list[str] = []
        self.functions: list[str] = []

    def log(self, message: str) -> None:
        line = str(message).rstrip("\n")
        if self._log_file is not None:
            self._log_file.write(line + "\n")
            self._log_file.flush()
        self._echo(f"[{self.stage}] {line}")

    def cfg(self, name: str | None = None) -> dict[str, Any]:
        return self.config.get(name or self.stage, {})

    def path(self, value: str | Path) -> Path:
        """A config path with ``{run}`` replaced by the run directory; relative paths start at the repository root."""
        text = str(value).replace("{run}", self.run_dir.as_posix())
        path = Path(text)
        return path if path.is_absolute() else REPO_ROOT / path

    def stage_dir(self, key: str = "output") -> Path:
        return self.path(self.cfg()[key])

    def output(self, reference: str) -> Path | None:
        """Output ``key`` of an earlier stage, written ``stage.key``; None if that stage left no such output."""
        stage, key = reference.split(".", 1)
        done = self.state.completed(stage)
        entry = (done or {}).get("outputs", {}).get(key)
        return Path(entry["abs"]) if entry else None

    def need(self, reference: str) -> Path:
        path = self.output(reference)
        if path is None or not path.exists():
            raise StageFailed(f"output {reference} is missing; run stage {reference.split('.')[0]} first (--from {reference.split('.')[0]})")
        return path

    def fresh_dir(self, path: Path, create: bool = True) -> Path:
        """Empty and recreate a stage folder; refuses anything outside the run directory."""
        path = Path(path).resolve()
        if self.run_dir.resolve() not in path.parents:
            raise StageFailed(f"refusing to clear {path}, which is outside the run directory")
        if path.exists():
            shutil.rmtree(path)
        if create:
            path.mkdir(parents=True)
        return path

    def call(self, name: str) -> None:
        self.functions.append(name)


@dataclass
class Stage:
    name: str
    run: Callable[[Context], Outcome]
    requires: list[str] = field(default_factory=list)
    optional: list[str] = field(default_factory=list)
    inputs: Callable[[Context], dict[str, Path]] = lambda ctx: {}
    params: Callable[[Context], dict[str, Any]] | None = None
    description: str = ""
    in_process: bool = False  # True: peak memory is the runner process; False: the stage's command-line scripts


def run_command(ctx: Context, command: list[str], timeout: float | None = None) -> None:
    """Run a command line script, copy its output to the log and the console and watch its memory."""
    display = ["python" if part == sys.executable else part for part in command]
    ctx.commands.append(subprocess.list2cmdline(display))
    ctx.log("$ " + ctx.commands[-1])
    environment = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               encoding="utf-8", errors="replace", env=environment)
    ctx.memory.add_root(process.pid)
    timed_out = threading.Event()

    def kill() -> None:
        timed_out.set()
        try:
            parent = psutil.Process(process.pid)
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()
        except psutil.Error:
            pass

    timer = threading.Timer(timeout, kill) if timeout else None
    if timer:
        timer.start()
    last = ""
    try:
        assert process.stdout is not None
        for line in process.stdout:
            if line.strip():
                last = line.strip()
            ctx.log(line)
        code = process.wait()
    finally:
        if timer:
            timer.cancel()
    script = Path(command[1]).name if len(command) > 1 else command[0]
    if timed_out.is_set():
        raise StageFailed(f"{script} timed out after {timeout:.0f} s")
    if code != 0:
        raise StageFailed(f"{script} exited with code {code}: {last[:300]}")


# ─── Stage implementations ─────────────────────────────────────────────

SCRIPTS = REPO_ROOT / "scripts"


def _release_preprocessing_sha(ctx: Context) -> str:
    manifest = read_json(ctx.path(ctx.config.get("release_dir", "models/release")) / "manifest.json")
    return manifest["files"]["features/preprocessing.json"]["sha256"]


def check_subset_files(ctx: Context, cfg: dict[str, Any]) -> None:
    """Compare the sample files with the hashes in the subset manifest."""
    expected = read_json(ctx.path(cfg["subset_manifest"]))["sha256"]
    for key in ("auth", "redteam", "line_map"):
        path = ctx.path(cfg[key])
        if sha256_file(path) != expected.get(path.name):
            raise StageFailed(f"{rel(path)} differs from the hash in {cfg['subset_manifest']}")
    ctx.log("subset files match the manifest hashes")


def ingest_reuse_problem(cfg: dict[str, Any], root: Path) -> str | None:
    """Why an existing ingestion output cannot be reused, or None when it matches the reuse rule."""
    reuse = cfg["reuse"]
    summary_path = root / "authentication" / "summary.json"
    if not summary_path.is_file():
        return f"{rel(summary_path)} is missing"
    summary = read_json(summary_path)
    counts = summary.get("counts", {})
    selection = summary.get("selection", {})
    if (selection.get("day_start_inclusive"), selection.get("day_end_inclusive")) != (reuse["day_start"], reuse["day_end"]):
        return f"{rel(summary_path)} does not cover days {reuse['day_start']}-{reuse['day_end']}"
    if not counts.get("reconciled") or counts.get("accepted_rows") != reuse["rows"]:
        return f"{rel(summary_path)} is not reconciled with {reuse['rows']:,} rows"
    return None


def stage_ingest(ctx: Context) -> Outcome:
    cfg = ctx.cfg()
    if "reuse" in cfg:
        root = ctx.path(cfg["reuse"]["path"])
        problem = ingest_reuse_problem(cfg, root)
        if problem is None:
            ctx.log(f"reusing {rel(root)} (days {cfg['reuse']['day_start']}-{cfg['reuse']['day_end']}, reconciled, {cfg['reuse']['rows']:,} rows)")
            summary = root / "authentication" / "summary.json"
            return Outcome(REUSED, {"root": root, "summary": summary, "events": root / "authentication" / "events",
                                    "labels": root / "redteam_labels" / "labels"}, {"summary": summary})
        ctx.log(f"cannot reuse {rel(root)}: {problem}")
    auth = ctx.options.auth or ctx.path(cfg["auth"])
    redteam = ctx.options.redteam or ctx.path(cfg["redteam"])
    if not auth.is_file() or not redteam.is_file():
        raise StageFailed(f"raw LANL files not found ({rel(auth)}, {rel(redteam)}); pass --auth and --redteam")
    if "subset_manifest" in cfg:
        check_subset_files(ctx, cfg)
    out = ctx.fresh_dir(ctx.stage_dir())
    workers = ctx.options.workers or cfg.get("workers", 1)
    command = [sys.executable, rel(SCRIPTS / "ingest_lanl.py"), "--auth", rel(auth), "--redteam", rel(redteam),
               "--output", rel(out), "--day-start", str(cfg["day_start"]), "--day-end", str(cfg["day_end"]),
               "--workers", str(workers)]
    if cfg.get("line_map"):
        command += ["--line-map", rel(ctx.path(cfg["line_map"]))]
    if cfg.get("inspection") and workers > 1:
        command += ["--inspection", rel(ctx.path(cfg["inspection"]))]
    run_command(ctx, command)
    summary = out / "authentication" / "summary.json"
    counts = read_json(summary)["counts"]
    if not counts.get("reconciled"):
        raise StageFailed(f"ingestion counts do not reconcile: {counts}")
    return Outcome(DONE, {"root": out, "summary": summary, "events": out / "authentication" / "events",
                          "labels": out / "redteam_labels" / "labels"},
                   {"auth": auth, "redteam": redteam, **({"line_map": ctx.path(cfg["line_map"])} if cfg.get("line_map") else {})},
                   {"accepted_rows": counts["accepted_rows"]})


def stage_features(ctx: Context) -> Outcome:
    cfg = ctx.cfg()
    release_sha = _release_preprocessing_sha(ctx)
    if "reuse" in cfg:
        root = ctx.path(cfg["reuse"]["path"])
        problem = None
        try:
            summary = read_json(root / "summary.json")
            counts = summary["counts"]
            if sha256_file(root / "preprocessing.json") != release_sha:
                problem = "preprocessing.json differs from the release file"
            elif counts["raw_events_written"] != cfg["reuse"]["events"] or counts["transformed_events_written"] != cfg["reuse"]["events"]:
                problem = f"event counts differ from {cfg['reuse']['events']:,}"
        except (OSError, KeyError, ValueError) as error:
            problem = f"cannot read the build: {error}"
        if problem is None:
            ctx.log(f"reusing {rel(root)} (release preprocessing hash, {cfg['reuse']['events']:,} events)")
            return Outcome(REUSED, {"root": root, "preprocessing": root / "preprocessing.json", "summary": root / "summary.json"},
                           {"preprocessing": root / "preprocessing.json"})
        ctx.log(f"cannot reuse {rel(root)}: {problem}")
    events = ctx.need("ingest.events")
    out = ctx.fresh_dir(ctx.stage_dir(), create=False)  # the build script creates its output folder
    release_file = ctx.path(ctx.config.get("release_dir", "models/release")) / "features" / "preprocessing.json"
    base = [sys.executable, rel(SCRIPTS / "build_lanl_features.py"), "--events", rel(events), "--output", rel(out),
            "--feature-config", rel(ctx.path(cfg["feature_config"])), "--splits-config", rel(ctx.path(cfg["splits_config"])),
            "--splits-manifest", rel(ctx.path(cfg["splits_manifest"]))]
    if cfg["mode"] == "release_preprocessing":
        # Pass 1 fits a preprocessing file on the subset; the frozen GRU needs the release file, so it replaces it
        # before pass 2 transforms the saved raw features.
        run_command(ctx, base + ["--raw-only", "--workers", "1"])
        shutil.copy2(out / "summary.json", out / "summary_raw_pass.json")
        shutil.copy2(out / "preprocessing.json", out / "preprocessing_fitted_on_subset.json")
        shutil.copy2(release_file, out / "preprocessing.json")
        ctx.log("replaced the subset-fitted preprocessing.json with the release file")
        run_command(ctx, base + ["--transform-only", "--workers", "1"])
    else:
        workers = ctx.options.workers or cfg.get("workers", 2)
        run_command(ctx, base + ["--workers", str(workers), "--assembly-workers", str(cfg.get("assembly_workers", 1)),
                                 "--low-memory-read"])
    if sha256_file(out / "preprocessing.json") != release_sha:
        raise StageFailed("the build's preprocessing.json differs from the release file, so the frozen GRU cannot score it")
    return Outcome(DONE, {"root": out, "preprocessing": out / "preprocessing.json", "summary": out / "summary.json"},
                   {"events": events, "preprocessing": release_file})


def _top_alerts(frame, day: int, budget: int) -> set[tuple[str, int]]:
    from dualscope.handoff.alerts import select_daily_alerts

    picked = select_daily_alerts(frame[frame["day"] == day], budget)
    return set(zip(picked["user"], picked["hour"].astype(int)))


def reproduce_day(scores, release, cfg: dict[str, Any], ctx: Context) -> dict[str, Any]:
    """Compare the fresh scoring of one validation day with the stored validation record."""
    import numpy as np
    import pandas as pd

    from dualscope.fusion.hourly import model_inputs
    from dualscope.pipeline.scoring import reproduction_check

    day, tolerance, top_k = cfg["day"], cfg["gru_tolerance"], cfg["top_k"]
    units_path, counts_path = ctx.path(cfg["units"]), ctx.path(cfg["counts"])
    if not units_path.is_file() or not counts_path.is_file():
        return {"skipped": f"reference files not found ({rel(units_path)}, {rel(counts_path)})"}
    units = pd.read_parquet(units_path, filters=[("day", "==", day)])
    counts = pd.read_parquet(counts_path, filters=[("day", "==", day)])
    check = reproduction_check(scores, units, counts, tolerance=tolerance)
    result: dict[str, Any] = {**check, "gru_tolerance": tolerance, "reference": [rel(units_path), rel(counts_path)]}
    if check["units_match"]:
        mine = scores[scores["day"] == day]
        merged = mine.merge(units[["user", "hour", "A:max_event"]], on=["user", "hour"], validate="1:1")
        difference = np.abs(merged["gru_max_event"] - merged["A:max_event"])
        result["median_gru_difference"] = float(difference.median())
        # Same rows and order as the fresh scoring, with the reference GRU values, so ties break the same way.
        reference = scores.merge(units[["user", "hour", "A:max_event"]], on=["user", "hour"], how="left")
        replaced = reference["A:max_event"].where(reference["day"] == day, reference["gru_max_event"])
        reference["gru_max_event"] = replaced
        reference["fusion"] = release.fusion_model.predict_proba(model_inputs(reference))[:, 1]
        mine_top, reference_top = _top_alerts(scores, day, top_k), _top_alerts(reference, day, top_k)
        result["top_alerts"] = len(mine_top)
        result["top_alert_sets_equal"] = mine_top == reference_top
        result["top_alert_differences"] = len(mine_top ^ reference_top)
        result["ok"] = bool(check["ok"] and result["top_alert_sets_equal"])
    return result


def stage_score(ctx: Context) -> Outcome:
    from dualscope.pipeline.release import load_release
    from dualscope.pipeline.scoring import check_days, score_days

    cfg = ctx.cfg()
    days = check_days(cfg["days"])
    features = ctx.need("features.root")
    out = ctx.fresh_dir(ctx.stage_dir())
    ctx.call("dualscope.pipeline.release.load_release")
    started = time.time()
    release = load_release(ctx.path(ctx.config.get("release_dir", "models/release")), device=cfg.get("device", "cpu"), log=ctx.log)
    load_seconds = time.time() - started
    ctx.call("dualscope.pipeline.scoring.score_days")
    timings: dict[str, float] = {}
    scores = score_days(features, days, release, device=cfg.get("device", "cpu"), log=ctx.log, timings=timings)
    scores.to_parquet(out / "scores.parquet", index=False)
    summary = {
        "days": days, "user_hours": len(scores), "user_hours_per_day": {str(d): int(n) for d, n in scores["day"].value_counts().sort_index().items()},
        "user_hours_with_events_but_no_gru_score": scores.attrs.get("unscored_user_hours", {}),
        "release_load_seconds": round(load_seconds, 2), **{k: round(v, 2) for k, v in timings.items()},
        "release_warnings": release.warnings, "device": cfg.get("device", "cpu"),
    }
    write_json(out / "scores_summary.json", summary)
    outputs = {"scores": out / "scores.parquet", "summary": out / "scores_summary.json"}
    notes: dict[str, Any] = {"user_hours": len(scores), **{k: v for k, v in timings.items()}}
    if "reproduction" in cfg:
        ctx.call("dualscope.pipeline.runner.reproduce_day")
        result = reproduce_day(scores, release, cfg["reproduction"], ctx)
        write_json(out / "reproduction_check.json", result)
        outputs["reproduction"] = out / "reproduction_check.json"
        ctx.log("reproduction check: " + json.dumps(result))
        notes["reproduction"] = {k: result.get(k) for k in ("ok", "max_gru_difference", "median_gru_difference", "counts_equal", "top_alert_sets_equal")}
        if not result.get("ok", result.get("skipped") is not None):
            raise StageFailed("day-16 reproduction check failed: " + json.dumps(notes["reproduction"]))
    return Outcome(DONE, outputs, {"features": features}, notes)


def stage_alerts(ctx: Context) -> Outcome:
    import pandas as pd

    from dualscope.pipeline.alerts import build_alert_package

    cfg = ctx.cfg()
    if cfg["split_name"] == "test":
        raise StageFailed("test alerts only come from the stored one-time test; this stage builds validation queues")
    scores = pd.read_parquet(ctx.need("score.scores"))
    features = ctx.need("features.root")
    labels = ctx.output("ingest.labels")
    out = ctx.fresh_dir(ctx.stage_dir(), create=False)  # build_alert_package creates the folder itself
    ctx.call("dualscope.pipeline.alerts.build_alert_package")
    manifest = build_alert_package(
        scores, features, out, budget=cfg["budget"], split_name=cfg["split_name"], id_prefix=cfg["id_prefix"],
        labels_dir=labels if labels is not None and labels.is_dir() else None,
        splits_config=ctx.path(cfg["splits_config"]), overwrite=True, log=ctx.log,
        extra_manifest={"pipeline": {"profile": ctx.config["profile"], "in_sample": True, "note": ctx.config["in_sample_note"]}},
    )
    return Outcome(DONE, {"package": out, "manifest": out / "manifest.json", "incidents": out / "incidents.jsonl"},
                   {"scores": ctx.need("score.scores")}, {"counts": manifest["counts"]})


def stage_test_package(ctx: Context) -> Outcome:
    cfg = ctx.cfg()
    package = ctx.path(cfg["path"])
    if not (package / "manifest.json").is_file():
        ctx.log(f"{rel(package)} is missing; rebuilding it from the stored test scores")
        run_command(ctx, [sys.executable, rel(SCRIPTS / "export_alert_handoff.py")])
    manifest = read_json(package / "manifest.json")
    counts = manifest.get("counts", {})
    wrong = {k: (counts.get(k), v) for k, v in cfg["expect"].items() if counts.get(k) != v}
    if wrong:
        raise StageFailed(f"{rel(package)} does not match the record of the test (found, expected): {wrong}; it was not changed")
    recorded = manifest.get("files", {}).get("incidents.jsonl")
    if recorded and sha256_file(package / "incidents.jsonl") != recorded:
        raise StageFailed(f"{rel(package / 'incidents.jsonl')} differs from the hash in its manifest")
    ctx.log(f"reusing {rel(package)}: {counts['alerts']} alerts, {counts['incidents']} incidents, {counts['redteam_alerts']} red-team alert(s)")
    return Outcome(REUSED, {"package": package, "manifest": package / "manifest.json", "incidents": package / "incidents.jsonl"},
                   {"manifest": package / "manifest.json"}, {"counts": counts})


def _investigation_problem(folder: Path, incidents: int | None) -> str | None:
    """Why a finished investigation run is incomplete (None when every reply of both modes is ok)."""
    try:
        summary = read_json(folder / "summary.json")
    except (OSError, ValueError):
        return f"{rel(folder / 'summary.json')} is missing"
    if incidents is not None and summary.get("incidents") != incidents:
        return f"{rel(folder)} covers {summary.get('incidents')} incidents, not {incidents}"
    for mode in ("rag", "direct"):
        status = summary.get("modes", {}).get(mode, {}).get("status", {})
        bad = sum(n for key, n in status.items() if key != "ok")
        if bad or sum(status.values()) != summary.get("incidents"):
            first = ""
            for record in _read_jsonl(folder / f"generated_{mode}.jsonl"):
                if record.get("status") != "ok" and record.get("errors"):
                    first = f" (first error: {str(record['errors'][0])[:200]})"
                    break
            return f"{bad or 'some'} of {summary.get('incidents')} {mode} replies are not ok{first}"
    return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError):
        return []


def _investigation_command(handoff: Path, folder: Path, workers: int, dry_run: bool) -> list[str]:
    command = [sys.executable, rel(SCRIPTS / "run_investigations.py"), "--handoff-dir", rel(handoff), "--output-dir", rel(folder),
               "--workers", str(workers)]
    return command + (["--dry-run"] if dry_run else ["--retry-failed"])


def stage_investigate(ctx: Context) -> Outcome:
    cfg = ctx.cfg()
    handoff = ctx.need(cfg["handoff"])
    workers = ctx.options.workers or cfg.get("workers", 6)
    timeout = cfg.get("timeout_seconds")
    outputs: dict[str, Path] = {}
    if "llm_output" in cfg:  # full profile: the stored GenAI run for the test queue
        folder = ctx.path(cfg["llm_output"])
        outputs["investigations"] = folder
        if ctx.options.llm:
            run_command(ctx, _investigation_command(handoff, folder, workers, dry_run=False), timeout)
            problem = _investigation_problem(folder, None)
            if problem:
                raise StageFailed(problem + "; alerts and evidence are unchanged, rerun with --only investigate --llm to resume")
            status = DONE
        else:
            problem = _investigation_problem(folder, cfg["reuse"]["incidents"])
            if problem:
                raise StageSkipped(f"no complete GenAI run to reuse ({problem}); pass --llm to generate it")
            ctx.log(f"reusing {rel(folder)}: {cfg['reuse']['incidents']} incidents, all replies ok")
            status = REUSED
        validation = ctx.path(cfg["validation_output"])
        queue = ctx.need("alerts.package")
        dry = validation / "dry_run" / "packages.jsonl"
        if dry.is_file() and dry.stat().st_mtime > (queue / "incidents.jsonl").stat().st_mtime:
            ctx.log(f"{rel(validation)} is newer than the validation queue; keeping it")
        else:
            validation = ctx.fresh_dir(validation)
            run_command(ctx, _investigation_command(queue, validation, workers, dry_run=True), timeout)
        outputs["investigations_validation"] = validation
        return Outcome(status, outputs, {"handoff": handoff / "incidents.jsonl"}, {"llm": ctx.options.llm})
    folder = ctx.path(cfg["output"])
    folder.mkdir(parents=True, exist_ok=True)
    outputs["investigations"] = folder
    run_command(ctx, _investigation_command(handoff, folder, workers, dry_run=not ctx.options.llm), timeout)
    if ctx.options.llm:
        problem = _investigation_problem(folder, None)
        if problem:
            raise StageFailed(problem + "; alerts and evidence are unchanged, rerun with --only investigate --llm to resume")
    return Outcome(DONE, outputs, {"handoff": handoff / "incidents.jsonl"},
                   {"mode": "llm" if ctx.options.llm else "dry run (evidence packages, ATT&CK retrieval and prompts; no model calls)"})


def check_package(incidents_path: Path, investigations: Path | None) -> dict[str, Any]:
    """Load a package the way the dashboard does and check every incident's evidence references resolve."""
    from dualscope.attack.retrieval import TechniqueRetriever
    from dualscope.dashboard import investigation
    from dualscope.dashboard.data import load_incidents
    from dualscope.dashboard.evidence import events_path_for, incident_events, load_events
    from dualscope.dashboard.queue import summarise_incidents

    incidents = load_incidents(incidents_path)
    if not incidents:
        raise StageFailed(f"{rel(incidents_path)} has no incidents")
    events = load_events(events_path_for(incidents_path))
    summaries = summarise_incidents(incidents, events, TechniqueRetriever.from_file())
    unresolved: dict[str, int] = {}
    references = 0
    for incident in incidents:
        _, missing = incident_events(events, incident.raw)
        references += len(set(incident.raw.get("source_references") or []))
        if missing:
            unresolved[incident.incident_id] = len(missing)
    if unresolved:
        raise StageFailed(f"{len(unresolved)} incidents cite events missing from events.parquet, for example {next(iter(unresolved))}")
    result: dict[str, Any] = {
        "incidents_path": rel(incidents_path), "incidents": len(incidents),
        "alert_hours": sum(len(i.raw.get("alert_hours") or []) for i in incidents),
        "events": len(events), "source_references": references, "unresolved_references": 0,
        "incidents_with_attack_candidates": sum(bool(s.candidates) for s in summaries),
        "redteam_incidents": sum(bool(i.raw.get("ground_truth_redteam")) for i in incidents),
        "investigations_folder": rel(investigations) if investigations else None,
    }
    views = investigation.load_run(investigations) if investigations and investigations.is_dir() else {}
    result["investigation_states"] = {
        mode: dict(Counter(investigation.lookup(views, mode, i.incident_id).state for i in incidents))
        for mode in investigation.MODES}
    return result


def launch_command(incidents: Path, investigations: Path | None) -> str:
    command = f"streamlit run scripts/incident_dashboard.py -- --incidents {rel(incidents)}"
    return command + (f" --investigations {rel(investigations)}" if investigations else "")


def stage_dashboard(ctx: Context) -> Outcome:
    cfg = ctx.cfg()
    out = ctx.fresh_dir(ctx.stage_dir())
    results: dict[str, Any] = {}
    commands: dict[str, str] = {}
    inputs: dict[str, Path] = {}
    for package in cfg["packages"]:
        incidents = ctx.need(package["incidents"]) / "incidents.jsonl"
        folder = ctx.output(package["investigations"])
        ctx.call("dualscope.dashboard (load_incidents, load_events, summarise_incidents, investigation.load_run)")
        results[package["name"]] = check_package(incidents, folder if folder and folder.exists() else None)
        commands[package["name"]] = launch_command(incidents, folder if folder and folder.exists() else None)
        inputs[package["name"]] = incidents
        ctx.log(f"{package['name']}: " + json.dumps(results[package["name"]]))
    write_json(out / "check.json", {"checked_utc": utc_now(), "packages": results, "launch_commands": commands})
    return Outcome(DONE, {"check": out / "check.json"}, inputs, {"launch_commands": commands})


IN_PROCESS_STAGES = {"score", "alerts", "dashboard"}
STAGE_IMPLEMENTATIONS: dict[str, tuple[Callable[[Context], Outcome], str]] = {
    "ingest": (stage_ingest, "Raw LANL lines to day-partitioned Parquet events and red-team labels (scripts/ingest_lanl.py)."),
    "features": (stage_features, "Causal features, with the frozen release preprocessing (scripts/build_lanl_features.py)."),
    "score": (stage_score, "Score user-hours with the frozen GRU and fusion model (dualscope.pipeline.scoring)."),
    "alerts": (stage_alerts, "Daily alert queue, events and incidents (dualscope.pipeline.alerts)."),
    "test_package": (stage_test_package, "Days 17-30 alert package from the stored one-time test; never rescored."),
    "investigate": (stage_investigate, "Evidence packages, ATT&CK retrieval and GenAI summaries (scripts/run_investigations.py)."),
    "dashboard": (stage_dashboard, "Headless dashboard check: every incident loads and its evidence resolves."),
}


def stage_inputs(stage: str, ctx: Context) -> dict[str, Path]:
    """Input files whose hashes decide whether a finished stage is still up to date."""
    cfg = ctx.cfg(stage)
    paths: dict[str, Path] = {}
    for key in ("auth", "redteam", "line_map", "subset_manifest", "feature_config", "splits_config", "splits_manifest"):
        if cfg.get(key):
            paths[key] = ctx.path(cfg[key])
    if stage == "ingest" and "reuse" in cfg:
        paths["reused_summary"] = ctx.path(cfg["reuse"]["path"]) / "authentication" / "summary.json"
    if stage == "features":
        paths["release_preprocessing"] = ctx.path(ctx.config.get("release_dir", "models/release")) / "features" / "preprocessing.json"
        if "reuse" in cfg:
            paths["reused_preprocessing"] = ctx.path(cfg["reuse"]["path"]) / "preprocessing.json"
    if stage == "score":
        paths["release_manifest"] = ctx.path(ctx.config.get("release_dir", "models/release")) / "manifest.json"
    if stage == "test_package":
        paths["manifest"] = ctx.path(cfg["path"]) / "manifest.json"
    if stage == "investigate" and "llm_output" in cfg:
        paths["reused_summary"] = ctx.path(cfg["llm_output"]) / "summary.json"
    return paths


def build_stages(config: dict[str, Any]) -> list[Stage]:
    stages = []
    for entry in config["stages"]:
        name = entry["name"]
        if name not in STAGE_IMPLEMENTATIONS:
            raise ConfigError(f"unknown stage {name!r}")
        run, description = STAGE_IMPLEMENTATIONS[name]
        stages.append(Stage(name, run, list(entry.get("requires", [])), list(entry.get("optional", [])),
                            inputs=lambda ctx, name=name: stage_inputs(name, ctx), description=description,
                            in_process=name in IN_PROCESS_STAGES))
    return stages


# ─── Orchestration ─────────────────────────────────────────────────────


def validate_config(config: dict[str, Any]) -> None:
    """Refuse a profile that asks for test days or test labels before any stage runs."""
    from dualscope.pipeline.scoring import ScoringError, check_days

    try:
        if "score" in config:
            check_days(config["score"]["days"])
    except ScoringError as error:
        raise ConfigError(str(error)) from error
    if config.get("alerts", {}).get("split_name") == "test":
        raise ConfigError("alerts.split_name is 'test'; test alerts only come from the stored one-time test")


def load_profile(profile: str) -> dict[str, Any]:
    path = CONFIG_DIR / f"pipeline_{profile}.json"
    if not path.is_file():
        raise ConfigError(f"no profile file {rel(path)}")
    config = read_json(path)
    validate_config(config)
    return config


@dataclass
class StageResult:
    name: str
    status: str
    reason: str = ""
    seconds: float = 0.0
    peak_mb: float | None = None


@dataclass
class RunResult:
    results: list[StageResult]
    state: PipelineState
    run_dir: Path

    @property
    def failed(self) -> list[StageResult]:
        return [r for r in self.results if r.status == FAILED]

    @property
    def exit_code(self) -> int:
        return 1 if self.failed else 0


def _token(stage: str, state: PipelineState) -> Any:
    done = state.completed(stage)
    if done is None:
        failed = state.stages.get(stage, {})
        return f"none:{failed.get('started_utc', '')}"
    return done["token"]


def _fingerprint(stage: Stage, ctx: Context, config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    inputs = {name: {"path": rel(path), **signature(path)} for name, path in stage.inputs(ctx).items()}
    params = stage.params(ctx) if stage.params else config.get(stage.name, {})
    parts = {
        "params": params, "inputs": inputs,
        "upstream": {name: _token(name, ctx.state) for name in [*stage.requires, *stage.optional]},
        "llm": ctx.options.llm if stage.name == "investigate" else None,
        "workers": ctx.options.workers, "auth": str(ctx.options.auth), "redteam": str(ctx.options.redteam),
    }
    return digest(parts), inputs


def _up_to_date(stage: Stage, ctx: Context, fingerprint: str) -> bool:
    done = ctx.state.completed(stage.name)
    if done is None or done["status"] != DONE or done["fingerprint"] != fingerprint:
        return False
    return all(Path(entry["abs"]).exists() for entry in done["outputs"].values())


def _describe(entries: dict[str, Path]) -> dict[str, Any]:
    return {name: {"path": rel(path), "abs": str(Path(path).resolve()), **signature(path)} for name, path in entries.items()}


def console(line: str) -> None:
    print(line, flush=True)


def run_pipeline(
    config: dict[str, Any],
    run_dir: Path,
    options: Options | None = None,
    stages: list[Stage] | None = None,
    only: list[str] | None = None,
    from_stage: str | None = None,
    force: bool = False,
    echo: Callable[[str], None] = console,
) -> RunResult:
    """Run (or skip) the stages of a profile and return what happened; the state file is saved after every stage."""
    options = options or Options()
    stages = stages if stages is not None else build_stages(config)
    names = [s.name for s in stages]
    for name in [*(only or []), *([from_stage] if from_stage else [])]:
        if name not in names:
            raise ConfigError(f"unknown stage {name!r}; stages are: {', '.join(names)}")
    run_dir = Path(run_dir)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    state = PipelineState(run_dir / STATE_FILE, config["profile"], run_dir)
    broken: set[str] = set()
    results: list[StageResult] = []

    for index, stage in enumerate(stages):
        named = bool(only and stage.name in only) or bool(from_stage and index >= names.index(from_stage))
        selected = (not only or stage.name in only) and (not from_stage or index >= names.index(from_stage))
        if not selected:
            results.append(StageResult(stage.name, SKIPPED, "not selected"))
            continue
        blockers = [r for r in stage.requires if r in broken]
        if blockers:
            broken.add(stage.name)
            reason = f"blocked: {', '.join(blockers)} did not finish"
            state.set(stage.name, {"status": SKIPPED, "reason": reason})
            results.append(StageResult(stage.name, SKIPPED, reason))
            echo(f"[{stage.name}] skipped, {reason}")
            continue

        memory = PeakMemory()
        log_path = run_dir / "logs" / f"{stage.name}.log"
        with log_path.open("w", encoding="utf-8") as log_file:
            ctx = Context(config, run_dir, options, state, stage.name, memory, log_file, echo)
            fingerprint, fp_inputs = _fingerprint(stage, ctx, config)
            if not (named or force) and _up_to_date(stage, ctx, fingerprint):
                reason = "up to date: inputs unchanged since the last successful run"
                state.set(stage.name, {"status": SKIPPED, "reason": reason, "fingerprint": fingerprint})
                results.append(StageResult(stage.name, SKIPPED, "up to date"))
                echo(f"[{stage.name}] skipped, up to date")
                continue
            outcome, reason, error, seconds, peak = None, "", None, 0.0, None
            started = utc_now()
            missing = [r for r in stage.requires if state.completed(r) is None]
            if missing:
                status, error = FAILED, f"needs the output of {missing[0]}, which has not finished; run with --from {missing[0]}"
                ctx.log("FAILED: " + error)
            else:
                if stage.in_process:
                    memory.add_root(os.getpid())
                memory.start()
                t0 = time.perf_counter()
                ctx.log(f"started {started}")
                try:
                    outcome = stage.run(ctx)
                    status = outcome.status
                except StageSkipped as skipped:
                    status, reason = SKIPPED, str(skipped)
                    broken.add(stage.name)  # stages that need its output are skipped too
                except Exception as exc:  # a stage failure is recorded, never raised past the runner
                    status = FAILED
                    error = str(exc) if isinstance(exc, StageFailed) else f"{type(exc).__name__}: {exc}"
                    if not isinstance(exc, StageFailed):
                        ctx.log(traceback.format_exc())
                    ctx.log("FAILED: " + error)
                seconds = round(time.perf_counter() - t0, 2)
                peak = memory.stop()
                ctx.log(f"{status} in {seconds:.1f} s" + (f", peak memory {peak} MB" if peak is not None else ""))

        record: dict[str, Any] = {
            "status": status, "reason": reason, "started_utc": started, "wall_seconds": seconds, "peak_memory_mb": peak,
            "memory_method": None if peak is None else ("runner process RSS (includes the interpreter and loaded libraries)"
                                                        if stage.in_process else "RSS of the child process trees of the stage's scripts"),
            "commands": ctx.commands, "functions": ctx.functions, "error": error, "fingerprint": fingerprint, "log": rel(log_path),
        }
        completed: Any = "keep"
        if outcome is not None:
            record["inputs"] = {**fp_inputs, **{n: {"path": rel(p), **signature(p)} for n, p in outcome.inputs.items()}}
            record["outputs"] = _describe(outcome.outputs)
            record["notes"] = outcome.notes
        if status in (DONE, REUSED) and outcome is not None:
            token = f"{started}/{time.time_ns()}" if status == DONE else digest({n: {k: v for k, v in e.items() if k not in ("path", "abs")}
                                                          for n, e in record["outputs"].items()})
            completed = {k: record[k] for k in ("status", "started_utc", "wall_seconds", "peak_memory_mb", "fingerprint", "outputs", "notes")}
            completed["token"] = token
        elif status == FAILED:
            completed = None  # outputs of a failed run may be partial
            broken.add(stage.name)
        state.set(stage.name, record, completed)
        results.append(StageResult(stage.name, status, error or reason, seconds, peak))
    return RunResult(results, state, run_dir)


def summary_table(results: list[StageResult]) -> str:
    rows = [("stage", "status", "seconds", "peak MB", "note")]
    for r in results:
        rows.append((r.name, r.status, f"{r.seconds:.1f}" if r.seconds else "-", f"{r.peak_mb:,.0f}" if r.peak_mb is not None else "-", r.reason[:90]))
    widths = [max(len(row[i]) for row in rows) for i in range(5)]
    return "\n".join("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip() for row in rows)


def final_report(result: RunResult, profile: str, llm: bool) -> str:
    lines = ["", "Pipeline summary", summary_table(result.results), ""]
    dashboard = (result.state.completed("dashboard") or {}).get("notes", {}).get("launch_commands", {})
    failed = result.failed
    if failed:
        for r in failed:
            if r.name == "investigate":
                hint = f"python scripts/run_pipeline.py --profile {profile} --only investigate --llm"
                lines.append(f"FAILED: investigate. {r.reason}")
                lines.append("The alerts and evidence from the earlier stages are untouched and stay viewable in the dashboard. "
                             f"After fixing the cause, resume with: {hint}")
            else:
                lines.append(f"FAILED: {r.name}. {r.reason}")
                lines.append(f"Fix the cause, then rerun from that stage with: python scripts/run_pipeline.py --profile {profile} --from {r.name}")
        blocked = [r.name for r in result.results if r.reason.startswith("blocked")]
        if blocked:
            lines.append(f"Not run because of the failure: {', '.join(blocked)}.")
    else:
        lines.append("All selected stages finished.")
    if dashboard:
        lines.append("")
        lines.append("Open the dashboard with:")
        lines.extend(f"  {command}" + (f"    ({name})" if len(dashboard) > 1 else "") for name, command in dashboard.items())
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DualScope pipeline end to end (Task 8.1). See docs/pipeline_runner.md.")
    parser.add_argument("--profile", choices=["sample", "full"], default="sample", help="sample: fixed days 1-9 subset; full: the run of record, reusing verified outputs")
    parser.add_argument("--run-dir", type=Path, help="default: outputs/pipeline/<profile>")
    parser.add_argument("--from", dest="from_stage", metavar="STAGE", help="run this stage and every stage after it")
    parser.add_argument("--only", nargs="+", metavar="STAGE", help="run only these stages")
    parser.add_argument("--force", action="store_true", help="run the selected stages even when they are up to date")
    parser.add_argument("--llm", action="store_true", help="call the model in the investigate stage (needs gcloud; sends requests). Default is evidence and prompts only")
    parser.add_argument("--workers", type=int, help="worker count for ingest, a full feature build and investigate")
    parser.add_argument("--auth", type=Path, help="raw auth.txt(.gz), used when the full profile has to ingest")
    parser.add_argument("--redteam", type=Path, help="raw redteam.txt(.gz), used when the full profile has to ingest")
    parser.add_argument("--list-stages", action="store_true", help="print the stages of the profile and exit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    args = parse_args(argv)
    try:
        config = load_profile(args.profile)
        stages = build_stages(config)
        if args.list_stages:
            print(f"Profile {args.profile}: {config['description']}")
            for number, stage in enumerate(stages, 1):
                needs = f" (needs {', '.join(stage.requires)})" if stage.requires else ""
                print(f"  {number}. {stage.name}{needs}: {stage.description}")
            return 0
        run_dir = args.run_dir or REPO_ROOT / config["run_dir"]
        run_dir = run_dir if run_dir.is_absolute() else Path.cwd() / run_dir
        options = Options(args.llm, args.workers, args.auth, args.redteam)
        print(f"Profile {args.profile}, run directory {rel(run_dir)}" + (", LLM calls enabled" if args.llm else ", no model calls"))
        result = run_pipeline(config, run_dir, options, stages, args.only, args.from_stage, args.force)
    except ConfigError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(final_report(result, args.profile, args.llm))
    return result.exit_code
