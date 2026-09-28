"""Shared synthetic data for sequence-detector tests.

The fixture runs deterministic synthetic authentication events through the
real Task 2.4 feature engine and preprocessor, then writes the transformed
feature table and a matching Task 2.2-style ingestion tree so evidence lookups
work. It uses the four-day fixture split policy (train days 1-2, validation
day 3, test day 4).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from dualscope.features.config import FeatureConfig
from dualscope.features.engine import HistoricalFeatureEngine
from dualscope.features.preprocessing import FeaturePreprocessor
from dualscope.splits import SplitConfig, is_fitting_eligible


FIXTURE_SPLITS = Path("data/fixtures/fixture_splits.json")
EXCLUDED_USER = "U9@DOM1"
LONG_HOUR = ("U3@DOM1", 86_400 + 5 * 3_600 + 1)
LONG_HOUR_EVENTS = 70

NORMALIZED_SCHEMA = pa.schema(
    [
        ("timestamp", pa.int64()),
        ("source_user", pa.string()),
        ("destination_user", pa.string()),
        ("source_computer", pa.string()),
        ("destination_computer", pa.string()),
        ("authentication_type", pa.string()),
        ("logon_type", pa.string()),
        ("authentication_orientation", pa.string()),
        ("authentication_result", pa.string()),
        ("acting_user", pa.string()),
        ("exact_duplicate_ordinal", pa.int32()),
        ("source_line", pa.int64()),
        ("source_reference", pa.string()),
        ("raw_record", pa.string()),
        ("dataset_day", pa.int32()),
    ]
)


@dataclass
class SyntheticSequenceData:
    root: Path
    features_dir: Path
    auth_root: Path
    split_cfg: SplitConfig
    preprocessing: dict
    events: list[dict]
    excluded_users: list[str]


def _generate_events(n_users: int = 4, hour_step: int = 3) -> list[dict]:
    rng = random.Random(7)
    pending: list[tuple[int, dict]] = []
    users = [f"U{i}@DOM1" for i in range(1, n_users + 1)]
    for day in range(1, 5):
        for hour in range(0, 24, hour_step):
            base = (day - 1) * 86_400 + hour * 3_600 + 1
            for user in users:
                for _ in range(rng.randint(1, 12)):
                    t = base + rng.randint(0, 3_599)
                    fail = rng.random() < 0.1
                    pending.append(
                        (
                            t,
                            {
                                "source_user": user,
                                "destination_user": user,
                                "source_computer": f"C{rng.randint(1, 4)}",
                                "destination_computer": f"C{rng.randint(1, 6)}",
                                "authentication_type": rng.choice(["Kerberos", "NTLM", "?"]),
                                "logon_type": rng.choice(["Network", "Interactive"]),
                                "authentication_orientation": rng.choice(["LogOn", "LogOff", "TGS"]),
                                "authentication_result": "Fail" if fail else "Success",
                            },
                        )
                    )
    # A long hour, with repeated timestamps, for chunking and ordering checks.
    user, hour_start = LONG_HOUR
    for i in range(LONG_HOUR_EVENTS):
        pending.append(
            (
                hour_start + (i // 3) * 10,
                {
                    "source_user": user,
                    "destination_user": user,
                    "source_computer": "C1",
                    "destination_computer": f"C{1 + i % 5}",
                    "authentication_type": "Kerberos",
                    "logon_type": "Network",
                    "authentication_orientation": "LogOn",
                    "authentication_result": "Success",
                },
            )
        )
    # Training-excluded user acting on day 2, and as a destination of U2.
    for offset, (src, dst) in enumerate([(EXCLUDED_USER, EXCLUDED_USER), ("U2@DOM1", EXCLUDED_USER)]):
        pending.append(
            (
                86_400 + 7 * 3_600 + 100 + offset,
                {
                    "source_user": src,
                    "destination_user": dst,
                    "source_computer": "C2",
                    "destination_computer": "C3",
                    "authentication_type": "NTLM",
                    "logon_type": "Network",
                    "authentication_orientation": "LogOn",
                    "authentication_result": "Success",
                },
            )
        )
    pending.sort(key=lambda item: item[0])
    events = []
    for line, (t, fields) in enumerate(pending, start=1):
        record = ",".join(
            [str(t), fields["source_user"], fields["destination_user"], fields["source_computer"],
             fields["destination_computer"], fields["authentication_type"], fields["logon_type"],
             fields["authentication_orientation"], fields["authentication_result"]]
        )
        events.append(
            {
                "timestamp": t,
                **fields,
                "acting_user": fields["source_user"],
                "exact_duplicate_ordinal": 1,
                "source_line": line,
                "source_reference": f"auth.txt:{line}",
                "raw_record": record,
                "dataset_day": (t - 1) // 86_400 + 1,
            }
        )
    return events


def _write_by_day(table: pa.Table, directory: Path, files_per_day: int = 1) -> list[dict]:
    """Write day partitions; several files per day makes readers see multi-chunk columns."""
    parts = []
    for day in sorted(set(table["dataset_day"].to_pylist())):
        day_rows = table.filter(pc.equal(table["dataset_day"], day))
        size = -(-day_rows.num_rows // files_per_day)
        for index in range(files_per_day):
            part = day_rows.slice(index * size, size)
            relative = Path(f"dataset_day={day:02d}") / f"part-{index:06d}.parquet"
            (directory / relative).parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(part, directory / relative)
            parts.append({"path": relative.as_posix(), "dataset_day": day, "rows": part.num_rows})
    return parts


def build_synthetic_sequence_data(root: Path, n_users: int = 4, hour_step: int = 3) -> SyntheticSequenceData:
    split_cfg = SplitConfig.from_file(FIXTURE_SPLITS)
    events = _generate_events(n_users, hour_step)
    normalized = pa.Table.from_pylist(events, schema=NORMALIZED_SCHEMA)

    auth_root = root / "authentication"
    parts = _write_by_day(normalized, auth_root / "events")
    (auth_root / "summary.json").write_text(
        json.dumps({"source_reference_format": "auth.txt:<source_line>", "parts": parts}), encoding="utf-8"
    )

    feature_cfg = FeatureConfig()
    engine = HistoricalFeatureEngine(feature_cfg)
    batch = normalized.drop_columns(["raw_record"]).to_batches()[0]
    raw = engine.process_batch(batch)
    preprocessor = FeaturePreprocessor(feature_cfg, split_policy_fingerprint=split_cfg.fingerprint(), mode="pilot", is_production=False)
    train = split_cfg.get_split("train")
    in_train = pc.and_(pc.greater_equal(raw["timestamp"], train.timestamp_start), pc.less(raw["timestamp"], train.timestamp_end))
    preprocessor.accumulate_training_batch(raw, pc.and_(in_train, is_fitting_eligible(raw, [EXCLUDED_USER])))
    preprocessor.freeze()
    transformed = pa.Table.from_batches([preprocessor.transform_batch(raw)])
    features_dir = root / "features" / "transformed" / "events"
    _write_by_day(transformed, features_dir, files_per_day=2)
    preprocessor.save(root / "features" / "preprocessing.json")
    return SyntheticSequenceData(
        root=root,
        features_dir=features_dir,
        auth_root=auth_root,
        split_cfg=split_cfg,
        preprocessing=preprocessor.to_dict(),
        events=events,
        excluded_users=[EXCLUDED_USER],
    )


@pytest.fixture(scope="session")
def synthetic_sequences(tmp_path_factory: pytest.TempPathFactory) -> SyntheticSequenceData:
    return build_synthetic_sequence_data(tmp_path_factory.mktemp("sequence_data"))
