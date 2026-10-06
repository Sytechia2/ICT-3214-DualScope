"""Incident packaging and multi-hour clustering engine (Task 5.4).

Groups related alerts under the same user into coherent incident envelopes.
Maintains full score breakdowns, detector consensus/disagreements,
and links to fine-grained event references for GenAI investigation and analyst triage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence
import pyarrow as pa

from dualscope.fusion.config import IncidentConfig, IncidentPriority
from dualscope.fusion.temporal import TemporalFusedScoreRow


@dataclass(frozen=True)
class IncidentRecord:
    """A multi-hour or single-hour security incident envelope with full provenance."""

    incident_id: str
    user_id: str
    start_time: int
    end_time: int
    duration_seconds: int
    duration_hours: int
    dataset_day: int
    split: str | None

    priority: IncidentPriority
    concordance: str  # concordant_alert | seq_only_alert | graph_only_alert | normal
    max_fused_score: float
    mean_fused_score: float
    fusion_method: str

    # Detector breakdown
    detector_scores: dict[str, Any]
    temporal_context: dict[str, Any]

    # Forensic evidence references
    source_references: list[str]
    evidence_chunk_references: list[str]
    graph_evidence_nodes: list[str]
    evidence_count: int

    # Signature rule match fields (Task 5.4 rule layer)
    is_rule_based_signature: bool = False
    rule_matches: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["priority"] = self.priority.value
        data["is_rule_based_signature"] = self.is_rule_based_signature
        data["rule_matches"] = list(self.rule_matches)
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


INCIDENT_RECORD_SCHEMA = pa.schema(
    [
        pa.field("incident_id", pa.string(), nullable=False),
        pa.field("user_id", pa.string(), nullable=False),
        pa.field("start_time", pa.int64(), nullable=False),
        pa.field("end_time", pa.int64(), nullable=False),
        pa.field("duration_seconds", pa.int64(), nullable=False),
        pa.field("duration_hours", pa.int32(), nullable=False),
        pa.field("dataset_day", pa.int32(), nullable=False),
        pa.field("split", pa.string(), nullable=True),
        pa.field("priority", pa.string(), nullable=False),
        pa.field("concordance", pa.string(), nullable=False),
        pa.field("max_fused_score", pa.float64(), nullable=False),
        pa.field("mean_fused_score", pa.float64(), nullable=False),
        pa.field("fusion_method", pa.string(), nullable=False),
        pa.field("source_references", pa.list_(pa.string()), nullable=False),
        pa.field("evidence_chunk_references", pa.list_(pa.string()), nullable=False),
        pa.field("graph_evidence_nodes", pa.list_(pa.string()), nullable=False),
        pa.field("evidence_count", pa.int32(), nullable=False),
        pa.field("is_rule_based_signature", pa.bool_(), nullable=False),
        pa.field("rule_matches", pa.list_(pa.string()), nullable=False),
    ]
)


class IncidentClusterer:
    """Clusters consecutive or closely spaced anomalous user-hours into incident records."""

    def __init__(
        self,
        config: IncidentConfig | None = None,
        known_signatures: Any | None = None,
    ) -> None:
        self.config = config or IncidentConfig()
        self.known_signatures = known_signatures

    def _determine_priority(
        self,
        max_score: float,
        has_concordant_alert: bool,
        has_temporal_boost: bool,
    ) -> IncidentPriority:
        """Assign operational triage priority based on consensus and calibrated score."""
        if has_concordant_alert or max_score >= self.config.critical_threshold:
            return IncidentPriority.CRITICAL
        if has_temporal_boost or max_score >= self.config.high_threshold:
            return IncidentPriority.HIGH
        if max_score >= self.config.medium_threshold:
            return IncidentPriority.MEDIUM
        return IncidentPriority.LOW

    def _cluster_single_group(
        self, user_id: str, group_rows: list[TemporalFusedScoreRow], incident_seq: int
    ) -> IncidentRecord:
        """Assemble an incident envelope from a consecutive sequence of alert rows."""
        start_time = min(r.window_start for r in group_rows)
        end_time = max(r.window_end for r in group_rows)
        duration_s = end_time - start_time
        duration_h = max(1, math.ceil(duration_s / 3600))
        day = group_rows[0].dataset_day
        split = group_rows[0].split

        scores = [r.fused_score for r in group_rows if r.fused_score is not None]
        max_score = max(scores) if scores else 0.0
        mean_score = sum(scores) / len(scores) if scores else 0.0

        # Check concordance across group
        has_concordant = any(
            r.disagreement_type.value == "concordant_alert" for r in group_rows
        )
        has_seq_only = any(r.disagreement_type.value == "seq_only_alert" for r in group_rows)
        has_graph_only = any(r.disagreement_type.value == "graph_only_alert" for r in group_rows)

        if has_concordant:
            concordance = "concordant_alert"
        elif has_seq_only and has_graph_only:
            concordance = "multi_detector_phased"
        elif has_seq_only:
            concordance = "seq_only_alert"
        elif has_graph_only:
            concordance = "graph_only_alert"
        else:
            concordance = "normal"

        has_boost = any(r.temporal_boost > 0.0 for r in group_rows)

        # Aggregate evidence references
        raw_source_lines: list[int] = []
        chunk_source_lines: list[int] = []
        graph_nodes: set[str] = set()

        for r in group_rows:
            if r.seq_source_lines:
                raw_source_lines.extend(r.seq_source_lines)
                if r.seq_evidence_chunk:
                    offset, length = r.seq_evidence_chunk
                    chunk_source_lines.extend(r.seq_source_lines[offset : offset + length])
            if r.graph_evidence_nodes:
                graph_nodes.update(r.graph_evidence_nodes)

        # Deduplicate while preserving source order
        seen = set()
        ordered_sources = []
        for line in raw_source_lines:
            ref = f"auth.txt:{line}"
            if ref not in seen:
                seen.add(ref)
                ordered_sources.append(ref)

        seen_chunk = set()
        ordered_chunk_refs = []
        for line in chunk_source_lines:
            ref = f"auth.txt:{line}"
            if ref not in seen_chunk:
                seen_chunk.add(ref)
                ordered_chunk_refs.append(ref)

        # Signature / known-attack rule matching (Task 5.4 rule layer)
        matched_rules: list[str] = []
        if self.known_signatures is not None:
            if callable(self.known_signatures):
                matched_rules = list(self.known_signatures(user_id, group_rows))
            elif hasattr(self.known_signatures, "triples"):
                th_triples = self.known_signatures.triples
                for u, s, d in th_triples:
                    if u == user_id:
                        if not graph_nodes or d in graph_nodes:
                            matched_rules.append(f"threat_triple:{u}:{s}->{d}")
            elif isinstance(self.known_signatures, (set, frozenset, list, tuple)):
                sig_set = set(self.known_signatures)
                if user_id in sig_set:
                    matched_rules.append(f"signature_user:{user_id}")
                for node in sorted(graph_nodes):
                    if node in sig_set:
                        matched_rules.append(f"signature_node:{node}")
                    if (user_id, node) in sig_set:
                        matched_rules.append(f"signature_pair:{user_id}->{node}")
                for item in sig_set:
                    if isinstance(item, tuple) and len(item) == 3:
                        u, s, d = item
                        if u == user_id and (not graph_nodes or d in graph_nodes):
                            matched_rules.append(f"threat_triple:{u}:{s}->{d}")

        is_signature = len(matched_rules) > 0
        if is_signature:
            priority = IncidentPriority.CRITICAL
        else:
            priority = self._determine_priority(max_score, has_concordant, has_boost)

        # Build clean incident identifier
        clean_user = user_id.replace("@", "_").replace("$", "")
        split_prefix = split.upper() if split else "LIVE"
        incident_id = f"INC-{split_prefix}-D{day:02d}-{clean_user}-{incident_seq:03d}"

        # Detect lead detector
        lead_dets = [r.lead_detector for r in group_rows if r.lead_detector]
        lead_times = [r.lead_time_seconds for r in group_rows if r.lead_time_seconds is not None]
        overall_lead = lead_dets[0] if lead_dets else None
        overall_lead_time = lead_times[0] if lead_times else None

        # Detector scores summary
        detector_summary = {
            "sequence": {
                "max_score": max((r.seq_score for r in group_rows if r.seq_score is not None), default=None),
                "any_alert": any(r.is_seq_alert for r in group_rows),
            },
            "graph": {
                "max_score": max((r.graph_score for r in group_rows if r.graph_score is not None), default=None),
                "any_alert": any(r.is_graph_alert for r in group_rows),
            },
        }

        temporal_summary = {
            "lead_detector": overall_lead,
            "lead_time_seconds": overall_lead_time,
            "max_temporal_boost": max(r.temporal_boost for r in group_rows),
        }

        return IncidentRecord(
            incident_id=incident_id,
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration_s,
            duration_hours=duration_h,
            dataset_day=day,
            split=split,
            priority=priority,
            concordance=concordance,
            max_fused_score=max_score,
            mean_fused_score=mean_score,
            fusion_method=group_rows[0].fusion_method.value,
            detector_scores=detector_summary,
            temporal_context=temporal_summary,
            source_references=ordered_sources,
            evidence_chunk_references=ordered_chunk_refs,
            graph_evidence_nodes=sorted(graph_nodes),
            evidence_count=len(ordered_sources),
            is_rule_based_signature=is_signature,
            rule_matches=matched_rules,
        )

    def cluster_incidents(
        self, fused_rows: Iterable[TemporalFusedScoreRow]
    ) -> list[IncidentRecord]:
        """Group alerting hours into multi-hour incident records."""
        # 1. Filter eligible rows: alerting rows (or sub-threshold if configured)
        eligible: list[TemporalFusedScoreRow] = []
        for r in fused_rows:
            if r.is_fused_alert is True:
                eligible.append(r)
            elif (
                self.config.include_sub_threshold_investigations
                and r.fused_score is not None
                and r.fused_score >= self.config.medium_threshold
            ):
                eligible.append(r)

        # 2. Group by user
        rows_by_user: dict[str, list[TemporalFusedScoreRow]] = {}
        for r in eligible:
            rows_by_user.setdefault(r.user_id, []).append(r)

        all_incidents: list[IncidentRecord] = []

        # 3. Cluster per user with max_merge_gap_seconds
        for user_id, u_rows in sorted(rows_by_user.items(), key=lambda x: x[0]):
            u_rows.sort(key=lambda r: r.window_start)
            current_cluster: list[TemporalFusedScoreRow] = []
            incident_seq = 1

            for row in u_rows:
                if not current_cluster:
                    current_cluster.append(row)
                else:
                    prev_end = current_cluster[-1].window_end
                    gap = row.window_start - prev_end
                    if gap <= self.config.max_merge_gap_seconds:
                        current_cluster.append(row)
                    else:
                        inc = self._cluster_single_group(user_id, current_cluster, incident_seq)
                        all_incidents.append(inc)
                        incident_seq += 1
                        current_cluster = [row]

            if current_cluster:
                inc = self._cluster_single_group(user_id, current_cluster, incident_seq)
                all_incidents.append(inc)

        all_incidents.sort(key=lambda x: (x.start_time, x.user_id))
        return all_incidents

    def to_arrow_table(self, incidents: Sequence[IncidentRecord]) -> pa.Table:
        """Serialise incident records into a PyArrow Table for dashboard/lakehouse consumption."""
        pylist = []
        for inc in incidents:
            d = inc.to_dict()
            d.pop("detector_scores", None)
            d.pop("temporal_context", None)
            pylist.append(d)
        return pa.Table.from_pylist(pylist, schema=INCIDENT_RECORD_SCHEMA)
