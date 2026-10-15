#!/usr/bin/env python3
"""Compare graph-detector metrics across chronological evaluation periods."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dualscope.graph.overfitting import temporal_overfitting_report


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _assert_temporal_order(development: dict, holdout: dict) -> None:
    development_days = development.get("evaluation_days")
    holdout_days = holdout.get("evaluation_days")
    if not development_days or not holdout_days:
        raise ValueError("both reports must contain evaluation_days")
    if int(development_days[1]) >= int(holdout_days[0]):
        raise ValueError("development and holdout periods overlap or are out of order")
    for name, report in (("development", development), ("holdout", holdout)):
        history_days = report.get("history_days")
        if history_days and int(history_days[1]) >= int(report["evaluation_days"][0]):
            raise ValueError(f"{name} history overlaps its evaluation period")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", required=True)
    parser.add_argument("--holdout", required=True)
    parser.add_argument("--max-absolute-f1-drop", type=float, default=0.10)
    parser.add_argument("--max-relative-f1-drop", type=float, default=0.50)
    parser.add_argument("--minimum-holdout-positives", type=int, default=30)
    parser.add_argument("--output")
    parser.add_argument(
        "--fail-on-overfitting",
        action="store_true",
        help="return exit status 1 when the check fails",
    )
    args = parser.parse_args()

    development = _load(args.development)
    holdout = _load(args.holdout)
    _assert_temporal_order(development, holdout)
    result = temporal_overfitting_report(
        development["metrics"],
        holdout["metrics"],
        max_absolute_f1_drop=args.max_absolute_f1_drop,
        max_relative_f1_drop=args.max_relative_f1_drop,
        minimum_holdout_positives=args.minimum_holdout_positives,
    )
    result["development_period"] = development["evaluation_days"]
    result["holdout_period"] = holdout["evaluation_days"]
    text = json.dumps(result, indent=2) + "\n"
    print(text, end="")
    if args.output:
        Path(args.output).write_text(text)
    if args.fail_on_overfitting and result["overfitting_detected"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
