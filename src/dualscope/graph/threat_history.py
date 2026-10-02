"""Causal threat-history rules for the hybrid graph detector.

Confirmed-incident memory is intentionally separate from the unsupervised
graph-autoencoder score. A match is a signature alert, not an anomaly score or
an estimate of attack probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class ThreatHistory:
    """Relationships confirmed malicious strictly before a freeze time."""

    frozen_before: int
    triples: frozenset[tuple[str, str, str]]
    source_computers: frozenset[str]

    @classmethod
    def from_labels(
        cls, labels: Iterable[Mapping[str, object]], *, frozen_before: int
    ) -> "ThreatHistory":
        eligible = [row for row in labels if int(row["timestamp"]) < frozen_before]
        triples = frozenset(
            (
                str(row["user"]),
                str(row["source_computer"]),
                str(row["destination_computer"]),
            )
            for row in eligible
        )
        return cls(
            frozen_before=int(frozen_before),
            triples=triples,
            source_computers=frozenset(source for _, source, _ in triples),
        )

    def matches(self, event: Mapping[str, object]) -> bool:
        """Return whether an event repeats a previously confirmed triple."""
        if int(event["timestamp"]) < self.frozen_before:
            return False
        triple = (
            str(event["acting_user"]),
            str(event["source_computer"]),
            str(event["destination_computer"]),
        )
        return triple in self.triples


def alerted_user_days(
    events: Iterable[Mapping[str, object]], history: ThreatHistory
) -> set[tuple[int, str]]:
    """Return ``(dataset_day, user)`` alerts produced by confirmed triples."""
    alerts: set[tuple[int, str]] = set()
    for event in events:
        if history.matches(event):
            day = int(
                event.get("dataset_day")
                or ((int(event["timestamp"]) - 1) // 86_400 + 1)
            )
            alerts.add((day, str(event["acting_user"])))
    return alerts


def classification_metrics(
    universe: Iterable[tuple[int, str]],
    positives: set[tuple[int, str]],
    alerts: set[tuple[int, str]],
) -> dict[str, float | int]:
    """Evaluate alerts over an explicit user-day universe."""
    units = set(universe)
    predicted = alerts & units
    actual = positives & units
    tp = len(predicted & actual)
    fp = len(predicted - actual)
    fn = len(actual - predicted)
    tn = len(units - predicted - actual)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
