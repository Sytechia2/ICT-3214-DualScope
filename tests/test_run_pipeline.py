"""Pipeline runner (Task 8.1): skipping, selection, state file, failure isolation and guards.

The stages here are small synthetic functions, except the investigation and
dashboard tests that use the real stage code on a tiny alert package with the
model call mocked. ``test_sample_profile_end_to_end`` runs the real sample
profile and is marked ``slow``.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dualscope.pipeline import runner
from dualscope.pipeline.runner import (
    ConfigError, Context, Options, Outcome, PeakMemory, Stage, StageFailed, StageSkipped,
    build_stages, final_report, run_command, run_pipeline, validate_config,
)
from test_dashboard_app import EVENTS, INCIDENTS, _restore_main_module  # noqa: F401  (fixture)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Recorder:
    """Synthetic stages that write one file each and count how often they ran."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp = tmp_path
        self.calls: list[str] = []
        self.source = tmp_path / "source.txt"
        self.source.write_text("one", encoding="utf-8")
        self.fail: set[str] = set()
        self.skip: set[str] = set()

    def stage(self, name: str, requires=(), optional=()) -> Stage:
        def run(ctx: Context) -> Outcome:
            self.calls.append(name)
            ctx.log(f"running {name}")
            if name in self.skip:
                raise StageSkipped("not possible here")
            if name in self.fail:
                raise StageFailed(f"{name} broke")
            out = ctx.run_dir / name
            out.mkdir(parents=True, exist_ok=True)
            (out / "result.txt").write_text(name + self.source.read_text(encoding="utf-8"), encoding="utf-8")
            return Outcome(DONE, {"result": out / "result.txt"}, {"source": self.source}, {"name": name})

        return Stage(name, run, list(requires), list(optional), inputs=lambda ctx, n=name: {"source": self.source} if n == "a" else {})

    def stages(self) -> list[Stage]:
        return [self.stage("a"), self.stage("b", ["a"]), self.stage("c", ["b"]), self.stage("d", ["a"], ["b"])]


DONE = runner.DONE
CONFIG = {"profile": "test", "stages": []}


def _run(rec: Recorder, run_dir: Path, **kwargs):
    return run_pipeline(CONFIG, run_dir, Options(), rec.stages(), echo=lambda _: None, **kwargs)


def _status(result) -> dict[str, str]:
    return {r.name: r.status for r in result.results}


def test_up_to_date_stages_are_skipped_until_an_input_changes(tmp_path):
    rec, run_dir = Recorder(tmp_path), tmp_path / "run"
    first = _run(rec, run_dir)
    assert _status(first) == {"a": "done", "b": "done", "c": "done", "d": "done"}
    assert rec.calls == ["a", "b", "c", "d"] and first.exit_code == 0

    second = _run(rec, run_dir)
    assert _status(second) == {"a": "skipped", "b": "skipped", "c": "skipped", "d": "skipped"}
    assert rec.calls == ["a", "b", "c", "d"]  # nothing ran again
    assert all(r.reason == "up to date" for r in second.results)

    rec.source.write_text("two", encoding="utf-8")  # a's input changed, so everything downstream reruns
    third = _run(rec, run_dir)
    assert _status(third) == {"a": "done", "b": "done", "c": "done", "d": "done"}
    assert (run_dir / "a" / "result.txt").read_text(encoding="utf-8") == "atwo"


def test_a_deleted_output_makes_the_stage_run_again(tmp_path):
    rec, run_dir = Recorder(tmp_path), tmp_path / "run"
    _run(rec, run_dir)
    (run_dir / "b" / "result.txt").unlink()
    result = _run(rec, run_dir)
    assert _status(result)["b"] == "done" and _status(result)["a"] == "skipped"


def test_force_only_and_from(tmp_path):
    rec, run_dir = Recorder(tmp_path), tmp_path / "run"
    _run(rec, run_dir)
    rec.calls.clear()

    assert _status(_run(rec, run_dir, force=True)) == {"a": "done", "b": "done", "c": "done", "d": "done"}
    rec.calls.clear()

    result = _run(rec, run_dir, only=["b"])  # named stages run even when up to date; others are left alone
    assert rec.calls == ["b"]
    assert _status(result) == {"a": "skipped", "b": "done", "c": "skipped", "d": "skipped"}
    assert [r.reason for r in result.results] == ["not selected", "", "not selected", "not selected"]
    rec.calls.clear()

    _run(rec, run_dir, from_stage="c")
    assert rec.calls == ["c", "d"]
    rec.calls.clear()

    _run(rec, run_dir, only=["a", "d"])
    assert rec.calls == ["a", "d"]
    with pytest.raises(ConfigError, match="unknown stage"):
        _run(rec, run_dir, only=["nope"])
    with pytest.raises(ConfigError, match="unknown stage"):
        _run(rec, run_dir, from_stage="nope")


def test_only_a_stage_whose_input_stage_never_ran_fails_clearly(tmp_path):
    rec = Recorder(tmp_path)
    result = _run(rec, tmp_path / "run", only=["c"])
    assert _status(result)["c"] == "failed" and "run with --from b" in result.failed[0].reason
    assert rec.calls == []


def test_state_file_records_every_stage(tmp_path):
    rec, run_dir = Recorder(tmp_path), tmp_path / "run"
    _run(rec, run_dir)
    state = json.loads((run_dir / "pipeline_state.json").read_text(encoding="utf-8"))
    assert state["profile"] == "test" and set(state["stages"]) == {"a", "b", "c", "d"}
    a = state["stages"]["a"]
    assert a["status"] == "done" and a["error"] is None and a["wall_seconds"] >= 0
    assert a["started_utc"].endswith("Z") and "peak_memory_mb" in a
    assert a["inputs"]["source"]["sha256"] == _sha(rec.source)
    assert a["outputs"]["result"]["sha256"] == _sha(run_dir / "a" / "result.txt")
    assert a["completed"]["status"] == "done" and a["completed"]["token"] and a["fingerprint"] == a["completed"]["fingerprint"]
    assert (run_dir / "logs" / "a.log").read_text(encoding="utf-8").count("running a") == 1
    _run(rec, run_dir)
    again = json.loads((run_dir / "pipeline_state.json").read_text(encoding="utf-8"))["stages"]["a"]
    assert again["status"] == "skipped" and again["completed"]["status"] == "done"  # the earlier success is kept


def test_failed_stage_blocks_its_dependents_and_names_the_rerun(tmp_path):
    rec, run_dir = Recorder(tmp_path), tmp_path / "run"
    rec.fail = {"b"}
    result = _run(rec, run_dir)
    assert _status(result) == {"a": "done", "b": "failed", "c": "skipped", "d": "done"}  # d needs only a (b is optional)
    assert result.exit_code == 1 and rec.calls == ["a", "b", "d"]
    assert result.results[2].reason.startswith("blocked: b")
    report = final_report(result, "test", llm=False)
    assert "FAILED: b. b broke" in report and "--profile test --from b" in report and "Not run because of the failure: c" in report
    state = json.loads((run_dir / "pipeline_state.json").read_text(encoding="utf-8"))["stages"]
    assert state["b"]["status"] == "failed" and state["b"]["error"] == "b broke" and state["b"]["completed"] is None

    rec.fail.clear()  # after the fix a normal run finishes the rest and reuses what is up to date
    again = _run(rec, run_dir)
    assert _status(again) == {"a": "skipped", "b": "done", "c": "done", "d": "done"}


def test_unexpected_exception_is_recorded_not_raised(tmp_path):
    rec = Recorder(tmp_path)

    def boom(ctx):
        raise ZeroDivisionError("oops")

    stages = [Stage("a", boom)]
    result = run_pipeline(CONFIG, tmp_path / "run", Options(), stages, echo=lambda _: None)
    assert result.failed[0].reason == "ZeroDivisionError: oops"
    assert "Traceback" in (tmp_path / "run" / "logs" / "a.log").read_text(encoding="utf-8")
    assert rec.calls == []


def test_a_stage_that_cannot_run_is_skipped_without_failing_the_run(tmp_path):
    rec = Recorder(tmp_path)
    rec.skip = {"b"}
    result = _run(rec, tmp_path / "run")
    assert result.exit_code == 0 and _status(result)["b"] == "skipped" and "not possible" in result.results[1].reason
    assert _status(result)["c"] == "skipped" and result.results[2].reason.startswith("blocked: b")


def test_test_days_are_refused_before_any_stage_runs():
    for days in ([17], [8, 16, 17], list(range(1, 31))):
        with pytest.raises(ConfigError, match="test days"):
            validate_config({"score": {"days": days}})
    with pytest.raises(ConfigError, match="stored one-time test"):
        validate_config({"alerts": {"split_name": "test"}})
    validate_config({"score": {"days": [8, 9]}, "alerts": {"split_name": "validation"}})
    for profile in ("sample", "full"):
        config = runner.load_profile(profile)
        assert max(config["score"]["days"]) <= 16
        assert [s.name for s in build_stages(config)] == [s["name"] for s in config["stages"]]


def test_peak_memory_and_run_command(tmp_path):
    memory = PeakMemory()
    memory.add_root(os.getpid())
    memory.start()
    block = np.ones(2_000_000)
    time.sleep(0.05)
    assert memory.stop() > 10 and block.sum() > 0
    assert PeakMemory().stop() is None  # nothing was watched

    lines: list[str] = []
    state = runner.PipelineState(tmp_path / "s.json", "test", tmp_path)
    ctx = Context({}, tmp_path, Options(), state, "x", PeakMemory(), None, lines.append)
    run_command(ctx, [sys.executable, "-c", "print('hello from the child')"])
    assert any("hello from the child" in line for line in lines) and ctx.memory.roots
    with pytest.raises(StageFailed, match="exited with code 3: boom"):
        run_command(ctx, [sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"])
    with pytest.raises(StageFailed, match="timed out"):
        run_command(ctx, [sys.executable, "-c", "import time; time.sleep(30)"], timeout=1)


def test_ingest_reuse_rule(tmp_path):
    cfg = {"reuse": {"path": str(tmp_path), "day_start": 1, "day_end": 30, "rows": 10}}
    assert "missing" in runner.ingest_reuse_problem(cfg, tmp_path)
    summary = tmp_path / "authentication" / "summary.json"
    summary.parent.mkdir()
    body = {"selection": {"day_start_inclusive": 1, "day_end_inclusive": 30}, "counts": {"reconciled": True, "accepted_rows": 10}}
    summary.write_text(json.dumps(body), encoding="utf-8")
    assert runner.ingest_reuse_problem(cfg, tmp_path) is None
    body["counts"]["accepted_rows"] = 9
    summary.write_text(json.dumps(body), encoding="utf-8")
    assert "not reconciled" in runner.ingest_reuse_problem(cfg, tmp_path)
    body.update(counts={"reconciled": True, "accepted_rows": 10}, selection={"day_start_inclusive": 1, "day_end_inclusive": 9})
    summary.write_text(json.dumps(body), encoding="utf-8")
    assert "does not cover" in runner.ingest_reuse_problem(cfg, tmp_path)


def test_investigation_problem_reads_the_summary(tmp_path):
    assert "missing" in runner._investigation_problem(tmp_path, None)
    summary = {"incidents": 2, "modes": {"rag": {"status": {"ok": 2}}, "direct": {"status": {"ok": 1, "failed": 1}}}}
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (tmp_path / "generated_direct.jsonl").write_text(json.dumps({"incident_id": "X", "status": "failed", "errors": ["gcloud missing"]}) + "\n", encoding="utf-8")
    problem = runner._investigation_problem(tmp_path, 2)
    assert "1 of 2 direct replies" in problem and "gcloud missing" in problem
    summary["modes"]["direct"]["status"] = {"ok": 2}
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    assert runner._investigation_problem(tmp_path, 2) is None
    assert "not 3" in runner._investigation_problem(tmp_path, 3)


# ─── Investigation failure keeps the alerts and the dashboard check ────


def _alerts_stage(tmp_path: Path) -> Stage:
    def run(ctx: Context) -> Outcome:
        out = ctx.run_dir / "alerts"
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(EVENTS).to_parquet(out / "events.parquet", index=False)
        (out / "incidents.jsonl").write_text("".join(json.dumps(i) + "\n" for i in INCIDENTS), encoding="utf-8")
        return Outcome(DONE, {"package": out})

    return Stage("alerts", run)


def _investigation_config() -> dict:
    return {"profile": "test", "investigate": {"handoff": "alerts.package", "output": "{run}/investigations", "workers": 2},
            "dashboard": {"packages": [{"name": "t", "incidents": "alerts.package", "investigations": "investigate.investigations"}],
                          "output": "{run}/dashboard"}}


def _stages(tmp_path: Path) -> list[Stage]:
    real = {s.name: s for s in build_stages({"stages": [
        {"name": "investigate", "requires": ["alerts"]}, {"name": "dashboard", "requires": ["alerts"], "optional": ["investigate"]}]})}
    return [_alerts_stage(tmp_path), real["investigate"], real["dashboard"]]


def test_failed_investigation_keeps_alerts_and_the_dashboard_check_runs(tmp_path, monkeypatch, _restore_main_module):
    seen = []

    def failing(ctx, command, timeout=None):
        seen.append(command)
        ctx.commands.append("run_investigations.py")
        raise StageFailed("run_investigations.py timed out after 7200 s")

    monkeypatch.setattr(runner, "run_command", failing)
    run_dir = tmp_path / "run"
    result = run_pipeline(_investigation_config(), run_dir, Options(llm=True), _stages(tmp_path), echo=lambda _: None)
    assert _status(result) == {"alerts": "done", "investigate": "failed", "dashboard": "done"}
    assert result.exit_code == 1 and "--dry-run" not in seen[0]
    before = {p.name: _sha(p) for p in (run_dir / "alerts").iterdir()}
    check = json.loads((run_dir / "dashboard" / "check.json").read_text(encoding="utf-8"))
    package = check["packages"]["t"]
    assert package["incidents"] == 2 and package["unresolved_references"] == 0
    assert package["investigation_states"]["AI + ATT&CK"] == {"not_generated": 2}
    report = final_report(result, "sample", llm=True)
    assert "FAILED: investigate" in report and "untouched" in report and "--only investigate --llm" in report
    assert "streamlit run scripts/incident_dashboard.py -- --incidents" in report

    # Nothing from earlier stages changes when the investigation is retried and fails again.
    run_pipeline(_investigation_config(), run_dir, Options(llm=True), _stages(tmp_path), only=["investigate"], echo=lambda _: None)
    assert {p.name: _sha(p) for p in (run_dir / "alerts").iterdir()} == before


def test_dry_run_default_passes_dry_run_and_skips_the_post_check(tmp_path, monkeypatch, _restore_main_module):
    commands = []

    def fake(ctx, command, timeout=None):
        commands.append(command)
        ctx.commands.append("run_investigations.py --dry-run")

    monkeypatch.setattr(runner, "run_command", fake)
    result = run_pipeline(_investigation_config(), tmp_path / "run", Options(), _stages(tmp_path), echo=lambda _: None)
    assert result.exit_code == 0 and "--dry-run" in commands[0]
    states = json.loads((tmp_path / "run" / "dashboard" / "check.json").read_text(encoding="utf-8"))["packages"]["t"]["investigation_states"]
    assert states["AI + ATT&CK, checked"] == {"not_generated": 2}  # a folder with only dry-run files reads as not generated


def test_dashboard_check_fails_when_an_evidence_reference_is_missing(tmp_path, _restore_main_module):
    def alerts(ctx):
        out = ctx.run_dir / "alerts"
        out.mkdir(parents=True)
        pd.DataFrame(EVENTS[:2]).to_parquet(out / "events.parquet", index=False)  # drops events the incidents cite
        (out / "incidents.jsonl").write_text("".join(json.dumps(i) + "\n" for i in INCIDENTS), encoding="utf-8")
        return Outcome(DONE, {"package": out})

    real = {s.name: s for s in build_stages({"stages": [{"name": "dashboard", "requires": ["alerts"]}]})}
    config = _investigation_config()
    config["dashboard"]["packages"][0]["investigations"] = "investigate.investigations"
    result = run_pipeline(config, tmp_path / "run", Options(), [Stage("alerts", alerts), real["dashboard"]], echo=lambda _: None)
    assert _status(result)["dashboard"] == "failed" and "missing from events.parquet" in result.failed[0].reason


# ─── Day-16 reproduction ───────────────────────────────────────────────


class _FakeFusion:
    def predict_proba(self, matrix):
        score = matrix[:, 0] / 10.0
        return np.column_stack([1 - score, score])


def _reproduction_frames(n=60):
    rows = pd.DataFrame({"user": [f"U{i}@DOM1" for i in range(n)], "hour": 1 + 15 * 86_400, "day": 16})
    rows["gru_max_event"] = np.linspace(0.0, 9.0, n)
    counts = {c: np.arange(n) % 3 for c in ["n_events", "n_failures", "n_sources", "n_destinations", "n_new_user_source",
                                              "n_new_host_connection", "n_new_user_destination", "n_ntlm", "n_network_logon", "n_logon", "either"]}
    scores = pd.concat([rows, pd.DataFrame(counts)], axis=1)
    scores["is_machine_account"] = False
    scores["fusion"] = scores["gru_max_event"] / 10.0
    units = rows[["user", "hour", "day"]].assign(**{"A:max_event": scores["gru_max_event"] + 1e-3})
    return scores, units, pd.concat([rows[["user", "hour", "day"]], pd.DataFrame(counts), scores[["is_machine_account"]]], axis=1)


def test_reproduction_accepts_small_differences_and_rejects_a_changed_top_set(tmp_path):
    scores, units, counts = _reproduction_frames()
    units.to_parquet(tmp_path / "units.parquet")
    counts.to_parquet(tmp_path / "counts.parquet")
    cfg = {"day": 16, "units": str(tmp_path / "units.parquet"), "counts": str(tmp_path / "counts.parquet"), "gru_tolerance": 5e-3, "top_k": 10}
    ctx = Context({}, tmp_path, Options(), runner.PipelineState(tmp_path / "s.json", "t", tmp_path), "score", PeakMemory(), None, lambda _: None)
    release = type("R", (), {"fusion_model": _FakeFusion()})()
    result = runner.reproduce_day(scores, release, cfg, ctx)
    assert result["ok"] and result["top_alert_sets_equal"] and result["counts_equal"]
    assert 0.0009 < result["median_gru_difference"] < 0.0011 and result["max_gru_difference"] < 5e-3

    units["A:max_event"] = np.where(np.arange(len(units)) == 59, 0.0, units["A:max_event"])  # the top hour drops out
    units.to_parquet(tmp_path / "units.parquet")
    failed = runner.reproduce_day(scores, release, {**cfg, "gru_tolerance": 100.0}, ctx)
    assert failed["top_alert_sets_equal"] is False and failed["ok"] is False

    counts.loc[3, "n_ntlm"] += 1
    counts.to_parquet(tmp_path / "counts.parquet")
    assert runner.reproduce_day(scores, release, cfg, ctx)["counts_equal"] is False


# ─── End to end ────────────────────────────────────────────────────────


@pytest.mark.slow
def test_sample_profile_end_to_end(tmp_path, _restore_main_module):
    config = runner.load_profile("sample")
    run_dir = tmp_path / "sample"
    first = run_pipeline(config, run_dir, Options(), echo=lambda _: None)
    assert first.exit_code == 0, [(r.name, r.reason) for r in first.failed]
    assert all(r.status == "done" for r in first.results)
    check = json.loads((run_dir / "dashboard" / "check.json").read_text(encoding="utf-8"))["packages"]["sample"]
    assert check["incidents"] > 0 and check["unresolved_references"] == 0
    second = run_pipeline(config, run_dir, Options(), echo=lambda _: None)
    assert all(r.status == "skipped" for r in second.results)
