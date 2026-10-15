import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dualscope.handoff.alerts import COUNT_COLUMNS, build_incident_records, group_incidents, select_daily_alerts
from dualscope.pipeline.alerts import build_alert_package
from dualscope.pipeline.scoring import ScoringError, reproduction_check, score_days


class _Untouched:
    """Any attribute access fails, proving the release is not used."""

    def __getattr__(self, name):
        raise AssertionError(f"release touched: {name}")


@pytest.mark.parametrize("days", [[17], [16, 17], range(1, 31), [30]])
def test_test_days_refused_before_loading(tmp_path, days):
    with pytest.raises(ScoringError, match="test days"):
        score_days(tmp_path / "does_not_exist", days, _Untouched())


def test_validation_day_passes_the_guard(tmp_path):
    with pytest.raises(FileNotFoundError):
        score_days(tmp_path / "does_not_exist", [16], _Untouched())


def _hour(day, index):
    return 1 + (day - 1) * 86_400 + index * 3_600


def _scores():
    rows = []
    specs = [
        ("U1@DOM1", 1, 0, 0.9, 2), ("U1@DOM1", 1, 1, 0.8, 1), ("U2@DOM1", 1, 5, 0.7, 3),
        ("C1$@DOM1", 1, 6, 0.1, 1), ("U1@DOM1", 2, 0, 0.6, 2),
    ]
    for i, (user, day, idx, fusion, n) in enumerate(specs):
        rows.append({
            "user": user, "day": day, "hour": _hour(day, idx), "fusion": fusion, "gru_max_event": 0.1 * (i + 1),
            **{c: 0 for c in COUNT_COLUMNS}, "n_events": n, "n_sources": 1, "n_destinations": 1,
            "is_machine_account": user.startswith("C1"),
        })
    return pd.DataFrame(rows)


def _write_events(root, scores):
    for day, rows in scores.groupby("day"):
        events = []
        for _, row in rows.iterrows():
            for k in range(int(row["n_events"])):
                line = len(events) + 1 + 100 * int(day)
                events.append({
                    "source_reference": f"auth.txt:{line}", "source_line": line, "timestamp": int(row["hour"]) + 60 * k,
                    "acting_user": row["user"], "source_user": row["user"], "destination_user": row["user"],
                    "source_computer": "C1", "destination_computer": "C2", "authentication_type": "Kerberos",
                    "logon_type": "Network", "authentication_orientation": "LogOn", "authentication_result": "Success",
                    "is_new_user_source": False, "is_new_host_connection": False, "is_new_user_destination": False,
                    "prior_auth_count_1h": 0, "prior_failure_count_1h": 0, "prior_unique_destinations_24h": 0,
                })
        part = root / "raw" / "events" / f"dataset_day={day}"
        part.mkdir(parents=True)
        pq.write_table(pa.Table.from_pandas(pd.DataFrame(events)), part / "part.parquet")


def test_group_incidents_and_records_defaults_unchanged():
    alerts = select_daily_alerts(_scores(), budget=10)
    ids = group_incidents(alerts)
    assert ids.iloc[0].startswith("INC-TEST-D01-")
    assert group_incidents(alerts, 7_200, "INC-VAL").iloc[0].startswith("INC-VAL-D01-")
    alerts["incident_id"] = ids
    alerts["ground_truth_redteam"] = False
    events = pd.DataFrame({"alert_id": alerts["alert_id"], "timestamp": 1, "source_reference": "auth.txt:1"})
    assert {r["split"] for r in build_incident_records(alerts, events)} == {"test"}
    assert {r["split"] for r in build_incident_records(alerts, events, split="validation")} == {"validation"}


def test_build_alert_package_without_labels(tmp_path):
    scores = _scores()
    _write_events(tmp_path / "features", scores)
    out = tmp_path / "pkg"
    manifest = build_alert_package(
        scores, tmp_path / "features", out, budget=3, split_name="train", id_prefix="INC-DEMO", log=lambda _: None)
    assert {p.name for p in out.iterdir()} == {"alerts.parquet", "events.parquet", "incidents.jsonl", "manifest.json"}
    alerts = pd.read_parquet(out / "alerts.parquet")
    events = pd.read_parquet(out / "events.parquet")
    assert len(alerts) == 4 and not alerts["ground_truth_redteam"].any()
    assert {"user_id", "dataset_day", "fusion_score"} <= set(alerts.columns)
    # Every scored event is kept: events per alert equal n_events.
    per_alert = events.groupby("alert_id").size().reindex(alerts["alert_id"]).to_numpy()
    assert np.array_equal(per_alert, alerts["n_events"].to_numpy())
    incidents = [json.loads(line) for line in (out / "incidents.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(i["incident_id"].startswith("INC-DEMO-") and i["split"] == "train" for i in incidents)
    assert manifest["answer_key"] is False and manifest["counts"]["redteam_alerts"] is None
    assert json.loads((out / "manifest.json").read_text(encoding="utf-8"))["files"]["alerts.parquet"]


def test_build_alert_package_guards(tmp_path):
    scores = _scores()
    _write_events(tmp_path / "features", scores)
    with pytest.raises(PermissionError):
        build_alert_package(scores, tmp_path / "features", tmp_path / "a", budget=3, split_name="test", id_prefix="X")
    out = tmp_path / "b"
    out.mkdir()
    with pytest.raises(FileExistsError):
        build_alert_package(scores, tmp_path / "features", out, budget=3, split_name="train", id_prefix="X")
    bad = scores.copy()
    bad.loc[0, "n_events"] += 1  # the scored count no longer matches the stored events
    with pytest.raises(ValueError, match="n_events"):
        build_alert_package(bad, tmp_path / "features", tmp_path / "c", budget=3, split_name="train", id_prefix="X",
                            log=lambda _: None)


def test_reproduction_check_detects_differences():
    names = ["n_events", "n_failures", "n_sources", "n_destinations", "n_new_user_source", "n_new_host_connection",
             "n_new_user_destination", "n_ntlm", "n_network_logon", "n_logon", "is_machine_account", "either"]
    key = {"user": ["U1"], "hour": [1], "day": [16]}
    mine = pd.DataFrame({**key, "gru_max_event": [0.5], **{c: [1] for c in names}})
    units = pd.DataFrame({**key, "A:max_event": [0.5 + 1e-6]})
    counts_ref = pd.DataFrame({**key, **{c: [1] for c in names}})
    assert reproduction_check(mine, units, counts_ref)["ok"]
    assert not reproduction_check(mine, units.assign(**{"A:max_event": 0.6}), counts_ref)["ok"]
    assert not reproduction_check(mine, units, counts_ref.assign(n_events=2))["counts_equal"]
