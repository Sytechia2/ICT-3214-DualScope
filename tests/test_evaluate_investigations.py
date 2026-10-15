"""Tests for scripts/evaluate_investigations.py (Task 9.5)."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

from openpyxl import Workbook
import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "evaluate_investigations.py"
_spec = importlib.util.spec_from_file_location("evaluate_investigations", SCRIPT)
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

INCIDENTS = [f"INC-{n}" for n in range(1, 6)]
HEADER = list(ev.COLUMNS)


def label(incident, technique, judgement):
    return [incident, "R", technique, judgement, "", ""]


CONSENSUS = [
    label("INC-1", "T1021.002", "Supported"),
    label("INC-2", "t1078 ", "Supported"),
    label("INC-2", "T1110", "Uncertain"),
    label("INC-3", "", "No supported mapping"),
    label("INC-4", "T1110", "Uncertain"),
    label("INC-5", "T1059", "Supported"),
]
REVIEWER_B = [
    label("INC-1", "T1021.001", "Supported"),
    label("INC-2", "T1110", "Supported"),
    label("INC-3", "T1059", "Supported"),
    label("INC-4", "T1110", "Uncertain"),
    label("INC-5", "T1059", "Supported"),
]


def write_xlsx(path, rows):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "labels"
    sheet.append(HEADER)
    for row in rows:
        sheet.append(row)
    workbook.save(path)
    return path


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        writer.writerows(rows)
    return path


def tech(*ids):
    return [{"technique_id": i} for i in ids]


def gen(incident, mode, ids=None, status="ok"):
    record = {"incident_id": incident, "mode": mode, "status": status}
    if status == "ok":
        record["response"] = {"techniques": tech(*(ids or [])), "no_supported_mapping": not ids}
    return record


def verified(incident, kept=(), removed=(), status="ok"):
    record = {"incident_id": incident, "status": status}
    if status == "ok":
        record["verified"] = {
            "techniques": [{"technique_id": i, "status": s} for i, s in kept],
            "removed_techniques": tech(*removed),
        }
    return record


def write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


@pytest.fixture()
def world(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_jsonl(run_dir / "generated_direct.jsonl", [
        gen("INC-1", "direct", ["T1021", "T1078"]),
        gen("INC-2", "direct", ["T1078", "T1110", "T9999"]),
        gen("INC-3", "direct", ["T1059"]),
        gen("INC-4", "direct", []),
        # INC-5 missing
    ])
    write_jsonl(run_dir / "generated_rag.jsonl", [
        gen("INC-1", "rag", ["T1021", "T1078"]),
        gen("INC-2", "rag", ["T1078", "T1110"]),
        gen("INC-3", "rag", []),
        gen("INC-4", "rag", status="failed"),
        gen("INC-5", "rag", []),
    ])
    write_jsonl(run_dir / "rag_verified.jsonl", [
        verified("INC-1", kept=[("T1021", "Supported")], removed=["T1078"]),
        verified("INC-2", kept=[("T1078", "Supported"), ("T1110", "Uncertain")]),
        verified("INC-3"),
        verified("INC-4", status="failed"),
        verified("INC-5", removed=["T1059"]),
    ])
    sample = tmp_path / "sample.json"
    sample.write_text(json.dumps({"incidents": [{"incident_id": i, "strata": {}} for i in INCIDENTS]}))
    return tmp_path, run_dir, sample


def score(world, labels, reviewers=()):
    _, run_dir, sample = world
    return ev.evaluate(run_dir, sample, labels, reviewers)


def n(metric):
    return metric["numerator"], metric["denominator"]


def test_scores_xlsx_and_csv_agree(world):
    tmp, _, _ = world
    from_xlsx = score(world, write_xlsx(tmp / "c.xlsx", CONSENSUS))
    from_csv = score(world, write_csv(tmp / "c.csv", CONSENSUS))
    assert from_xlsx["modes"] == from_csv["modes"]
    assert from_xlsx["sample"]["incidents"] == 5
    assert from_xlsx["sample"]["reference_no_mapping"] == 1


def test_direct_metrics(world):
    tmp, _, _ = world
    direct = score(world, write_csv(tmp / "c.csv", CONSENSUS))["modes"]["direct"]
    assert n(direct["valid_id_rate"]) == (5, 6)
    assert n(direct["precision_strict"]) == (1, 6)
    assert n(direct["precision_parent"]) == (2, 6)
    assert n(direct["uncertain_matches"]) == (1, 6)
    assert n(direct["unsupported_rate"]) == (3, 6)
    assert n(direct["recall_parent"]) == (2, 3)
    assert n(direct["empty_outputs"]) == (1, 5)  # INC-5 missing
    assert n(direct["abstentions"]["correct"]) == (1, 1)  # INC-4 reference only Uncertain
    assert n(direct["false_mappings_on_no_mapping"]) == (1, 1)


def test_rag_and_verified_metrics(world):
    tmp, _, _ = world
    modes = score(world, write_csv(tmp / "c.csv", CONSENSUS))["modes"]
    rag, ver = modes["rag"], modes["rag_verified"]
    assert n(rag["precision_strict"]) == (1, 4)
    assert n(rag["unsupported_rate"]) == (1, 4)
    assert n(rag["empty_outputs"]) == (1, 5)  # INC-4 failed
    assert n(rag["abstentions"]["correct"]) == (1, 2)  # INC-3
    assert n(rag["abstentions"]["missed"]) == (1, 2)  # INC-5
    assert n(rag["false_mappings_on_no_mapping"]) == (0, 1)
    assert n(ver["precision_parent"]) == (2, 3)
    assert n(ver["unsupported_rate"]) == (0, 3)
    assert n(ver["empty_outputs"]) == (1, 5)


def test_verification_impact(world):
    tmp, _, _ = world
    impact = score(world, write_csv(tmp / "c.csv", CONSENSUS))["verification_impact"]
    assert n(impact["correctly_removed"]) == (1, 2)  # T1078 on INC-1
    assert n(impact["supported_wrongly_removed"]) == (1, 2)  # T1059 on INC-5
    confusion = impact["kept_confusion_verifier_vs_reviewer"]
    assert confusion["Supported"] == {"Supported": 1, "Uncertain": 0, "none": 1}
    assert confusion["Uncertain"] == {"Supported": 0, "Uncertain": 1, "none": 0}


def test_reviewer_agreement(world):
    tmp, _, _ = world
    a = write_xlsx(tmp / "a.xlsx", CONSENSUS)
    b = write_csv(tmp / "b.csv", REVIEWER_B)
    agreement = score(world, a, [a, b])["reviewer_agreement"]
    assert agreement["mean_jaccard_supported_parent"] == pytest.approx(0.6)
    assert n(agreement["any_mapping_agreement"]) == (4, 5)
    assert n(agreement["disagreeing_incidents"]) == (2, 5)


def test_validation_errors():
    def errors(rows):
        with pytest.raises(ev.LabelError) as info:
            ev.build_reference([dict(zip(HEADER, r)) for r in rows], INCIDENTS)
        return str(info.value)

    assert "INC-5" in errors(CONSENSUS[:-1])  # no label row for INC-5
    assert "mixes" in errors(CONSENSUS + [label("INC-3", "T1059", "Supported")])
    assert "bad judgement" in errors(CONSENSUS + [label("INC-1", "T1059", "Maybe")])
    assert "malformed" in errors(CONSENSUS + [label("INC-1", "1059", "Supported")])
    assert "unknown incident" in errors(CONSENSUS + [label("INC-99", "T1059", "Supported")])


def test_cli_writes_outputs(world):
    tmp, run_dir, sample = world
    labels = write_xlsx(tmp / "c.xlsx", CONSENSUS)
    out = tmp / "out"
    args = ["--run-dir", str(run_dir), "--sample", str(sample), "--output-dir", str(out)]
    assert ev.main(args + ["--labels", str(labels)]) == 0
    scores = json.loads((out / "scores.json").read_text())
    assert len(scores["inputs"]["labels"]["sha256"]) == 64
    assert "Precision (strict)" in (out / "report.md").read_text()
    with (out / "per_incident.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5 and "T1021=supported_parent" in rows[0]["direct_verdicts"]
    bad = write_csv(tmp / "bad.csv", CONSENSUS[:-1])
    assert ev.main(args + ["--labels", str(bad)]) == 1
