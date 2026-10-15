#!/usr/bin/env python3
"""Run the DualScope pipeline end to end (Task 8.1); see docs/pipeline_runner.md.

Examples:
  python scripts/run_pipeline.py --profile sample
  python scripts/run_pipeline.py --profile full
  python scripts/run_pipeline.py --profile sample --only investigate --force
  python scripts/run_pipeline.py --profile sample --from features
  python scripts/run_pipeline.py --profile full --list-stages
"""

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from dualscope.pipeline.runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
