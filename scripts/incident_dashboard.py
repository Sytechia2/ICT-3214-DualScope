"""Launch with ``streamlit run scripts/incident_dashboard.py`` from the repo root."""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = str(REPOSITORY_ROOT / "src")
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

from dualscope.dashboard.app import run  # noqa: E402


run()
