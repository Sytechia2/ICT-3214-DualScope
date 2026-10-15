import pandas as pd

from dualscope.handoff.alerts import (
    PRIORITY_ABOVE_CUTOFF,
    PRIORITY_TIED_AT_CUTOFF,
    build_incident_records,
    group_incidents,
    label_alerts,
    select_daily_alerts,
)

COUNTS = {
    "n_events": 2, "n_failures": 0, "n_new_user_source": 0, "n_new_host_connection": 0,
    "n_new_user_destination": 0, "n_ntlm": 0, "n_network_logon": 1, "n_logon": 1,
    "n_sources": 1, "n_destinations": 1,
}


def _scores(rows):
    return pd.DataFrame([{**COUNTS, "user": u, "day": d, "hour": h, "fusion": f, "gru_max_event": g} for u, d, h, f, g in rows])


def _hour(day, index):
    return 1 + (day - 1) * 86_400 + index * 3_600


def test_select_keeps_stored_order_for_ties():
    scores = _scores([
        ("U1@DOM1", 1, _hour(1, 0), 0.5, 0.1),
        ("U2@DOM1", 1, _hour(1, 1), 0.9, 0.9),
        ("U3@DOM1", 1, _hour(1, 2), 0.5, 0.2),
        ("U4@DOM1", 1, _hour(1, 3), 0.5, 0.3),
        ("U5@DOM1", 2, _hour(2, 0), 0.7, 0.5),
    ])
    alerts = select_daily_alerts(scores, budget=3)
    day1 = alerts[alerts["day"] == 1]
    # The tie at 0.5 is broken by stored row order, as in the final test.
    assert list(day1["user"]) == ["U2@DOM1", "U1@DOM1", "U3@DOM1"]
    assert list(day1["rank_in_day"]) == [1, 2, 3]
    assert list(day1["tied_at_cutoff"]) == [False, True, True]
    assert list(alerts["alert_id"]) == ["ALR-D01-01", "ALR-D01-02", "ALR-D01-03", "ALR-D02-01"]
    assert (alerts["window_end"] - alerts["window_start"] == 3_600).all()
    assert day1["gru_percentile_in_day"].iloc[0] == 1.0


def test_group_incidents_merges_hours_within_gap():
    alerts = select_daily_alerts(_scores([
        ("U1@DOM1", 1, _hour(1, 0), 0.9, 0.1),
        ("U1@DOM1", 1, _hour(1, 2), 0.8, 0.1),   # 1-hour gap: same incident
        ("U1@DOM1", 1, _hour(1, 6), 0.7, 0.1),   # 3-hour gap: new incident
        ("U2$@DOM1", 1, _hour(1, 0), 0.6, 0.1),
    ]), budget=10)
    ids = dict(zip(alerts["alert_id"], group_incidents(alerts, max_gap_seconds=7_200)))
    assert ids["ALR-D01-01"] == ids["ALR-D01-02"] == "INC-TEST-D01-U1_DOM1-001"
    assert ids["ALR-D01-03"] == "INC-TEST-D01-U1_DOM1-002"
    assert ids["ALR-D01-04"] == "INC-TEST-D01-U2_DOM1-001"


def test_incident_records_carry_evidence_priority_and_labels():
    alerts = select_daily_alerts(_scores([
        ("U1@DOM1", 1, _hour(1, 0), 0.9, 0.4),
        ("U1@DOM1", 1, _hour(1, 1), 0.5, 0.2),
        ("U2@DOM1", 1, _hour(1, 0), 0.5, 0.1),
    ]), budget=3)
    alerts["incident_id"] = group_incidents(alerts)
    alerts["ground_truth_redteam"] = label_alerts(alerts, {("U2@DOM1", _hour(1, 0))})
    events = pd.DataFrame({
        "alert_id": ["ALR-D01-01", "ALR-D01-01", "ALR-D01-02", "ALR-D01-03"],
        "timestamp": [20, 10, 3_700, 30],
        "source_reference": ["auth.txt:5", "auth.txt:4", "auth.txt:9", "auth.txt:7"],
    })
    graph = pd.DataFrame([{
        "user_id": "U1@DOM1", "dataset_day": 1, "status": "available", "score": 0.95, "is_alert": True,
        "new_edge_count": 2, "degree_growth": 1, "evidence_nodes": ["U1@DOM1", "C1"],
        "top_edges": [{
            "destination_computer": "C1", "edge_raw_score": 0.3, "is_new_edge": True, "success_count": 1,
            "failure_count": 0, "source_lines": [4], "references_truncated": False,
        }],
    }])
    records = {r["user_id"]: r for r in build_incident_records(alerts, events, graph)}

    u1 = records["U1@DOM1"]
    assert u1["source_references"] == ["auth.txt:4", "auth.txt:5", "auth.txt:9"]
    assert u1["evidence_count"] == 3
    assert u1["priority"] == PRIORITY_ABOVE_CUTOFF
    assert (u1["start_time"], u1["end_time"], u1["duration_hours"]) == (_hour(1, 0), _hour(1, 2), 2)
    assert u1["detector_scores"]["graph"] == {"max_score": 0.95, "any_alert": True}
    assert u1["graph_context"]["top_edges"][0]["source_references"] == ["auth.txt:4"]
    assert u1["graph_evidence_nodes"] == ["C1", "U1@DOM1"]
    assert u1["ground_truth_redteam"] is False

    u2 = records["U2@DOM1"]
    assert u2["priority"] == PRIORITY_TIED_AT_CUTOFF
    assert u2["detector_scores"]["graph"] == {"max_score": None, "any_alert": None}
    assert u2["graph_context"]["top_edges"] == []
    assert u2["ground_truth_redteam"] is True and u2["ground_truth_redteam_hours"] == 1
