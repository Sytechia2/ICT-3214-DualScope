"""Temporal generalisation checks for graph-detector evaluations."""

from __future__ import annotations

from typing import Mapping


def temporal_overfitting_report(
    development: Mapping[str, float | int],
    holdout: Mapping[str, float | int],
    *,
    max_absolute_f1_drop: float = 0.10,
    max_relative_f1_drop: float = 0.50,
    minimum_holdout_positives: int = 30,
) -> dict[str, object]:
    """Check for a material F1 collapse on a later chronological holdout.

    A failure is evidence consistent with overfitting, not proof of its cause.
    A pass only means this specific temporal-collapse check found no evidence.
    """
    for name, metrics in (("development", development), ("holdout", holdout)):
        for key in ("f1", "precision", "recall", "tp", "fn"):
            if key not in metrics:
                raise ValueError(f"{name} metrics are missing {key!r}")

    development_f1 = float(development["f1"])
    holdout_f1 = float(holdout["f1"])
    absolute_drop = max(0.0, development_f1 - holdout_f1)
    relative_drop = absolute_drop / development_f1 if development_f1 else 0.0
    holdout_positives = int(holdout["tp"]) + int(holdout["fn"])
    enough_positives = holdout_positives >= minimum_holdout_positives
    excessive_drop = (
        absolute_drop > max_absolute_f1_drop
        and relative_drop > max_relative_f1_drop
    )
    detected = excessive_drop or not enough_positives
    reasons: list[str] = []
    if excessive_drop:
        reasons.append(
            "later-holdout F1 dropped beyond both configured absolute and relative limits"
        )
    if not enough_positives:
        reasons.append(
            f"holdout contains only {holdout_positives} positives; at least "
            f"{minimum_holdout_positives} are required"
        )
    if not reasons:
        reasons.append("no material temporal F1 collapse was observed")

    return {
        "overfitting_detected": detected,
        "status": "FAIL" if detected else "PASS",
        "development_f1": development_f1,
        "holdout_f1": holdout_f1,
        "absolute_f1_drop": absolute_drop,
        "relative_f1_drop": relative_drop,
        "holdout_positives": holdout_positives,
        "limits": {
            "max_absolute_f1_drop": max_absolute_f1_drop,
            "max_relative_f1_drop": max_relative_f1_drop,
            "minimum_holdout_positives": minimum_holdout_positives,
        },
        "reasons": reasons,
        "interpretation": (
            "PASS means no temporal performance-collapse evidence was found; "
            "it does not prove that the model is free from overfitting."
        ),
    }
