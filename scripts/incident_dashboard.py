"""Launch with ``streamlit run scripts/incident_dashboard.py`` from the repo root.

To open another run, pass its files after ``--``:

    streamlit run scripts/incident_dashboard.py -- --incidents outputs/pipeline/sample/alerts/incidents.jsonl --investigations outputs/pipeline/sample/investigations

The environment variables DUALSCOPE_INCIDENTS and DUALSCOPE_INVESTIGATIONS do the same.
Without them the dashboard opens the final test alert package and the Gemini investigation run.
"""

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = str(REPOSITORY_ROOT / "src")
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

from dualscope.dashboard.app import run  # noqa: E402


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--incidents", help="incidents.jsonl of the alert package to open")
    parser.add_argument("--investigations", help="folder with the investigation run to show")
    return parser.parse_known_args(sys.argv[1:])[0]  # other arguments (Streamlit's own, pytest's) are ignored


_args = _arguments()
run(_args.incidents, _args.investigations)
