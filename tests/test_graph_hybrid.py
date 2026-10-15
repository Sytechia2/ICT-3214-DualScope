from dualscope.graph.hybrid import collect_relationship_matches,combine_user_day
from dualscope.graph.threat_history import ThreatHistory


def test_hybrid_keeps_relationship_and_gae_signals_separate():
    record={"dataset_day":17,"user_id":"U1","window_start":1,"window_end":2,"score_available_at":2,"model_version":"gae-v1","score":.7,"is_alert":False}
    match={"timestamp":2,"source_computer":"S1","destination_computer":"D1","source_reference":"auth.txt:2"}
    row=combine_user_day(record,[match])
    assert row["gae_is_alert"] is False
    assert row["base_gae_score"] is None
    assert row["peak_unique_destinations_300s"] is None
    assert row["confirmed_relationship_alert"] is True
    assert row["primary_high_confidence_alert"] is True
    assert row["priority_reason"]=="confirmed_relationship"


def test_hybrid_relationship_evidence_is_causal():
    history=ThreatHistory.from_labels([{"timestamp":1,"user":"U1","source_computer":"S1","destination_computer":"D1"}],frozen_before=10)
    events=[{"timestamp":10,"acting_user":"U1","source_computer":"S1","destination_computer":"D1","source_reference":"auth.txt:7"},{"timestamp":11,"acting_user":"U2","source_computer":"S1","destination_computer":"D1"}]
    matches=collect_relationship_matches(events,history)
    assert list(matches)==["U1"]
    assert matches["U1"][0]["source_reference"]=="auth.txt:7"
