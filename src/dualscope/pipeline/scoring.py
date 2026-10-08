"""Score user-hours of a feature build with the frozen release model (Task 8.1).

Follows ``scripts/supervised_fusion_final_test.py``: the GRU ``max_event`` score
and the hourly counts are joined per user-hour, put in the final test's fixed
seed-0 shuffled order (so score ties break the same way in ``select_daily_alerts``)
and scored by the fusion model.

Test days 17-30 are refused before anything is loaded: the one-time final test
is spent and those days are not scored again.

Feature builds. ``load_inputs`` accepts a build only if its ``preprocessing.json``
is frozen, was fitted with the same feature configuration and split policy, and
(unless ``allow_pilot_features``) is a production build. A small build made with
``build_lanl_features.py --raw-only`` followed by copying the release
``preprocessing.json`` in and running ``--transform-only`` carries the release
file, so it passes as a production build and needs no flag. ``score_days`` also
requires the build's preprocessing sha256 to equal the release's. Pass
``allow_pilot_features=True`` only for a build whose preprocessing file says it
is a pilot; the sha check still applies, so the GRU always sees the inputs it
was trained on.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from dualscope.fusion.hourly import INPUTS, hourly_counts, is_machine, model_inputs
from dualscope.pipeline.release import REPO_ROOT, Release
from dualscope.sequence.pipeline import SequenceInputs, load_inputs
from dualscope.sequence.scoring import score_day

TEST_DAYS = range(17, 31)
CHECK_DAY = 16
SPLITS_CONFIG = REPO_ROOT / "config" / "lanl_splits.json"
SPLITS_MANIFEST = REPO_ROOT / "data" / "manifests" / "lanl_splits_v1.json"
FEATURE_CONFIG = REPO_ROOT / "config" / "lanl_features_v2.json"
MAX_GRU_DIFFERENCE = 1e-4


class ScoringError(RuntimeError):
    """Scoring refused, or the inputs do not match the release."""


def check_days(days: Iterable[int]) -> list[int]:
    """Sorted unique days; raises if any is a test day (17-30) or none is given."""
    days = sorted({int(d) for d in days})
    blocked = [d for d in days if d in TEST_DAYS]
    if blocked:
        raise ScoringError(f"days {blocked} are test days (17-30); the one-time final test is spent and they are not scored")
    if not days:
        raise ScoringError("no days requested")
    return days


def load_build(
    features_root: str | Path,
    release: Release,
    splits_config: str | Path = SPLITS_CONFIG,
    splits_manifest: str | Path = SPLITS_MANIFEST,
    feature_config: str | Path = FEATURE_CONFIG,
    allow_pilot_features: bool = False,
) -> SequenceInputs:
    """Open a feature build and check its preprocessing file is the release's."""
    inputs = load_inputs(features_root, splits_config, splits_manifest, feature_config, allow_pilot_features=allow_pilot_features)
    if inputs.feature_info()["preprocessing_sha256"] != release.preprocessing_sha256:
        raise ScoringError(f"{inputs.preprocessing_path} differs from the release preprocessing file; the GRU needs the one it was trained with")
    return inputs


def gru_scores(release: Release, inputs: SequenceInputs, days: list[int], log: Callable[[str], None] = print) -> pd.DataFrame:
    """GRU ``max_event`` score for every scorable user-hour of ``days``."""
    parts = []
    started = time.time()
    config = release.gru_config
    for day in inputs.iter_days(days, config.policy):
        available = day.scorable_mask()
        scores = score_day(release.gru_model, day, config.policy.max_sequence_length)
        parts.append(pd.DataFrame({
            "user": day.users[available], "hour": day.hour_starts[available], "day": day.dataset_day,
            "gru_max_event": scores.raw["max_event"][available],
        }))
        log(f"  GRU: day {day.dataset_day} scored, {int(available.sum()):,} of {len(available):,} user-hours available ({time.time() - started:.0f}s)")
    if not parts:
        raise ScoringError(f"no data found for days {days}")
    return pd.concat(parts, ignore_index=True)


def count_days(features_root: str | Path, days: list[int], log: Callable[[str], None] = print) -> pd.DataFrame:
    """Hourly event counts for ``days`` from the build's raw events."""
    dataset = ds.dataset(str(Path(features_root) / "raw" / "events"), format="parquet", partitioning="hive")
    parts = []
    for day in days:
        started = time.time()
        parts.append(hourly_counts(dataset, day))
        log(f"  counts: day {day} ({time.time() - started:.0f}s)")
    return pd.concat(parts, ignore_index=True)


def score_days(
    features_root: str | Path,
    days: Iterable[int],
    release: Release,
    splits_config: str | Path = SPLITS_CONFIG,
    splits_manifest: str | Path = SPLITS_MANIFEST,
    feature_config: str | Path = FEATURE_CONFIG,
    device: str = "cpu",
    log: Callable[[str], None] = print,
    allow_pilot_features: bool = False,
) -> pd.DataFrame:
    """Fusion score for every GRU-scorable user-hour of ``days``.

    Columns: user, hour, day, gru_max_event, the count columns, ``either``,
    ``is_machine_account`` and ``fusion``. Rows are in the final test's seed-0
    shuffled order. Raises ``ScoringError`` for any day in 17-30 before loading
    anything. ``device`` moves the release GRU model; see the module docstring
    for feature builds and ``allow_pilot_features``.
    """
    days = check_days(days)
    inputs = load_build(features_root, release, splits_config, splits_manifest, feature_config, allow_pilot_features)
    release.gru_model.to(device)
    gru = gru_scores(release, inputs, days, log)
    counts = count_days(features_root, days, log)
    frame = gru.merge(counts, on=["user", "hour", "day"], how="inner", validate="1:1")
    if len(frame) != len(gru):
        raise ScoringError("some GRU-scored user-hours have no events")
    if not np.array_equal(is_machine(frame["user"].to_numpy()), frame["is_machine_account"].to_numpy(bool)):
        raise ScoringError("is_machine_account disagrees with the machine-account rule")
    frame = frame.sort_values(["day", "user", "hour"], kind="stable").sample(frac=1.0, random_state=0).reset_index(drop=True)
    frame["fusion"] = release.fusion_model.predict_proba(model_inputs(frame))[:, 1]
    return frame


def reproduction_check(
    scores_day16: pd.DataFrame,
    units_ref: pd.DataFrame,
    counts_ref: pd.DataFrame,
    gru_column: str = "A:max_event",
    tolerance: float = MAX_GRU_DIFFERENCE,
) -> dict[str, Any]:
    """Compare a day-16 scoring with the validation record (the final test's reproduction check).

    ``units_ref`` has user, hour, day and ``gru_column``; ``counts_ref`` has the
    count columns and ``either``. ``ok`` needs the same user-hours, a max GRU
    difference <= ``tolerance`` and equal counts.
    """
    day = CHECK_DAY
    ref = units_ref[units_ref["day"] == day][["user", "hour", "day", gru_column]].merge(
        counts_ref[counts_ref["day"] == day], on=["user", "hour", "day"], validate="1:1")
    mine = scores_day16[scores_day16["day"] == day]
    both = mine.merge(ref, on=["user", "hour", "day"], how="outer", suffixes=("", "_ref"), validate="1:1", indicator=True)
    units_match = bool((both["_merge"] == "both").all())
    result: dict[str, Any] = {
        "day": day, "user_hours": int(len(both)), "units_match": units_match,
        "max_gru_difference": None, "counts_equal": False, "ok": False,
    }
    if units_match:
        columns = [*INPUTS[1:], "either"]
        diff = float(np.max(np.abs(both["gru_max_event"] - both[gru_column])))
        equal = all(np.array_equal(both[c].to_numpy(np.int64), both[f"{c}_ref"].to_numpy(np.int64)) for c in columns)
        result.update(max_gru_difference=diff, counts_equal=equal, ok=bool(diff <= tolerance and equal))
    return result
