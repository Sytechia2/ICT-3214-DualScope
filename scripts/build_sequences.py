"""Task 3.1: construct user-hour sequences and record their manifest.

Streams the Task 2.4 transformed features one dataset day at a time, builds
acting-user/hour sequences, verifies split and hour boundaries for every
event, and records per-split status counts (including warm-up and
insufficient sequences), fitting eligibility, sequence-length distribution and
the chunk counts produced by each candidate maximum length. Optionally writes a
small padded-tensor sample with its evidence references for inspection.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from dualscope.sequence.builder import build_chunks, gather_padded, source_reference  # noqa: E402
from dualscope.sequence.config import SEQUENCE_POLICY_VERSION, SequenceDetectorConfig, input_feature_names  # noqa: E402
from dualscope.sequence.pipeline import git_revision, load_inputs  # noqa: E402
from dualscope.splits import SequenceCandidateRejectionTracker  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-root", type=Path, default=REPO_ROOT / "data/processed/lanl_features_days_01_30")
    parser.add_argument("--splits-config", type=Path, default=REPO_ROOT / "config/lanl_splits.json")
    parser.add_argument("--splits-manifest", type=Path, default=REPO_ROOT / "data/manifests/lanl_splits_v1.json")
    parser.add_argument("--feature-config", type=Path, default=REPO_ROOT / "config/lanl_features.json")
    parser.add_argument("--sequence-config", type=Path, default=REPO_ROOT / "config/sequence_detector.json")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data/manifests/lanl_sequences_v1.json")
    parser.add_argument("--days", type=int, nargs="*", default=None, help="Restrict to these dataset days.")
    parser.add_argument("--sample-output", type=Path, default=None, help="Write a padded-tensor sample (.npz + .json).")
    parser.add_argument("--sample-day", type=int, default=None, help="Day for the tensor sample (default: first validation day).")
    parser.add_argument("--sample-sequences", type=int, default=12)
    parser.add_argument("--allow-pilot-features", action="store_true")
    return parser.parse_args()


def _percentiles(values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {}
    points = [50, 75, 90, 95, 99, 99.9, 100]
    return {f"p{p:g}": float(v) for p, v in zip(points, np.percentile(values, points))}


def write_tensor_sample(day, lengths: int, count: int, output: Path) -> dict:
    """Save padded tensors, masks and evidence references for a few sequences."""
    available = np.flatnonzero(day.scorable_mask())
    rng = np.random.default_rng(0)
    # Seeded random hours, including a few that split into 2-3 chunks so
    # truncation-free splitting is visible without one busy account dominating.
    counts = day.counts[available]
    multi_pool = available[(counts > lengths) & (counts <= 3 * lengths)]
    single_pool = available[counts <= lengths]
    multi = rng.choice(multi_pool, size=min(len(multi_pool), max(1, count // 4)), replace=False)
    single = rng.choice(single_pool, size=min(len(single_pool), count - len(multi)), replace=False)
    chosen = np.sort(np.r_[single, multi])
    chunks = build_chunks(day.offsets, day.counts, lengths, chosen)
    dense, categorical, mask = gather_padded(day.numeric, day.categorical, chunks.start, chunks.length, pad_to=lengths)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output.with_suffix(".npz"), dense=dense, categorical=categorical, mask=mask, lengths=chunks.length)
    sequences = []
    for row, (uh, chunk_index, start, length) in enumerate(zip(chunks.user_hour, chunks.chunk_index, chunks.start, chunks.length)):
        sequences.append(
            {
                "row": row,
                "user_id": str(day.users[uh]),
                "window_start": int(day.hour_starts[uh]),
                "window_end": int(day.hour_starts[uh]) + 3600,
                "split": day.split,
                "hour_events": int(day.counts[uh]),
                "chunk_index": int(chunk_index),
                "chunks_in_hour": int(np.sum(chunks.user_hour == uh)),
                "length": int(length),
                "timestamps": day.timestamps[start : start + length].tolist(),
                "source_references": [source_reference(x) for x in day.source_lines[start : start + length]],
            }
        )
    description = {
        "dataset_day": day.dataset_day,
        "pad_to": lengths,
        "arrays": {
            "dense": "float32 [sequence, step, feature] numeric then binary inputs; zero at padded steps",
            "categorical": "int64 [sequence, step, feature] vocabulary IDs; zero at padded steps",
            "mask": "bool [sequence, step]; True for real events",
            "lengths": "int32 [sequence] real events per sequence",
        },
        "dense_features": input_feature_names()[:8],
        "categorical_features": input_feature_names()[8:],
        "sequences": sequences,
    }
    output.with_suffix(".json").write_text(json.dumps(description, indent=2), encoding="utf-8")
    return {"npz": output.with_suffix(".npz").as_posix(), "json": output.with_suffix(".json").as_posix(), "sequences": len(sequences)}


def main() -> int:
    args = parse_args()
    config = SequenceDetectorConfig.from_file(args.sequence_config)
    inputs = load_inputs(
        args.features_root, args.splits_config, args.splits_manifest, args.feature_config,
        allow_pilot_features=args.allow_pilot_features,
    )
    policy = config.policy
    candidate_lengths = sorted(set(config.search.max_sequence_lengths) | {policy.max_sequence_length})
    days = args.days or sorted(d for name in ("train", "validation", "test") for d in inputs.days_in(name))
    sample_day = args.sample_day or (inputs.days_in("validation") or days)[0]

    tracker = SequenceCandidateRejectionTracker(hour_seconds=policy.hour_seconds)
    splits: dict[str, dict] = {}
    counts_by_split: dict[str, list[np.ndarray]] = {}
    per_day = []
    sample_info = None
    started = time.time()
    for day in inputs.iter_days(days, policy, tracker):
        entry = splits.setdefault(
            day.split,
            {
                "days": [], "events": 0, "user_hours": 0, "status_counts": {}, "status_detail_counts": {},
                "fitting_eligible_user_hours": 0, "fitting_eligible_events": 0,
                "user_hours_touching_excluded_users": 0, "chunks_by_max_length": {str(L): 0 for L in candidate_lengths},
            },
        )
        available = day.scorable_mask()
        entry["days"].append(day.dataset_day)
        entry["events"] += day.n_events
        entry["user_hours"] += day.n_user_hours
        for status, n in zip(*np.unique(day.status, return_counts=True)):
            entry["status_counts"][str(status)] = entry["status_counts"].get(str(status), 0) + int(n)
        for detail, n in zip(*np.unique(day.status_detail[day.status_detail != ""], return_counts=True)):
            entry["status_detail_counts"][str(detail)] = entry["status_detail_counts"].get(str(detail), 0) + int(n)
        entry["fitting_eligible_user_hours"] += int(day.fitting_eligible.sum())
        entry["fitting_eligible_events"] += int(day.counts[day.fitting_eligible].sum())
        entry["user_hours_touching_excluded_users"] += int((day.ineligible_events > 0).sum())
        for L in candidate_lengths:
            entry["chunks_by_max_length"][str(L)] += int(np.sum(-(-day.counts[available] // L)))
        counts_by_split.setdefault(day.split, []).append(day.counts[available])
        per_day.append({"dataset_day": day.dataset_day, "split": day.split, "events": day.n_events, "user_hours": day.n_user_hours, "available_user_hours": int(available.sum())})
        print(f"day {day.dataset_day:02d} [{day.split}] events={day.n_events:,} user-hours={day.n_user_hours:,} available={available.sum():,}")
        if args.sample_output is not None and day.dataset_day == sample_day:
            sample_info = write_tensor_sample(day, policy.max_sequence_length, args.sample_sequences, args.sample_output)

    for name, entry in splits.items():
        lengths = np.concatenate(counts_by_split[name]) if counts_by_split[name] else np.zeros(0)
        entry["available_events_per_user_hour"] = _percentiles(lengths)
        entry["available_user_hours_within_length"] = {
            str(L): float(np.mean(lengths <= L)) if len(lengths) else None for L in candidate_lengths
        }

    manifest = {
        "manifest_version": 1,
        "task": "3.1 — Construct user-hour sequences",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_revision": git_revision(),
        "sequence_policy_version": SEQUENCE_POLICY_VERSION,
        "sequence_policy": {
            **policy.__dict__,
            "acting_user": "acting_user (= source_user, Task 2.2)",
            "unit": "(acting_user, hour_start) with hour_start = 1 + floor((timestamp - 1) / 3600) * 3600",
            "event_order": "timestamp, then source_line (both ascending)",
            "long_hours": "split into ceil(n / L) balanced consecutive chunks (lengths differ by at most one); no event is dropped",
            "padding": "right padding with a boolean mask; padded steps carry zero inputs and zero loss",
            "warm_up": "user-hours whose events lack a complete 24h feature history (dataset day 1) are insufficient_history",
            "fitting_eligibility": "train split, available, and no event whose source or destination user is a training-excluded user",
            "candidate_max_lengths": candidate_lengths,
            "model_inputs": input_feature_names(),
        },
        "inputs": {
            "feature": inputs.feature_info(),
            "split_policy_fingerprint": inputs.split_cfg.fingerprint(),
            "excluded_training_users": inputs.excluded_users,
            "sequence_config_fingerprint": config.fingerprint(),
        },
        "boundary_checks": {
            "every_event_inside_its_split": True,
            "every_event_inside_its_hour": True,
            "tracker": tracker.summary(),
        },
        "splits": splits,
        "days": per_day,
        "tensor_sample": sample_info,
        "labels_used": False,
        "runtime_seconds": round(time.time() - started, 1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Sequence manifest written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
