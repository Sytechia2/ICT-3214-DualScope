"""Frozen graph detector artifact with integrity checks."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib, json
from pathlib import Path
from typing import Any

import numpy as np

from dualscope.graph.config import GraphDetectorConfig, GraphModelSettings, GraphPolicy, GraphTrainingSettings
from dualscope.graph.model import GraphAutoencoder
from dualscope.graph.scoring import score_snapshot
from dualscope.graph.snapshot import GraphSnapshot
from dualscope.graph.training import load_checkpoint
from dualscope.sequence.calibration import QuantileTailCalibrator

FROZEN_FILENAME="frozen_graph_detector.json"

@dataclass
class FrozenGraphDetector:
    model: GraphAutoencoder; config: GraphDetectorConfig; calibrator: QuantileTailCalibrator
    alert_threshold: float; model_version: str; run_id: str
    def score(self,snapshot:GraphSnapshot):
        edge,raw=score_snapshot(self.model,snapshot,self.config.aggregation)
        return edge,raw,self.calibrator.transform(raw)
    @classmethod
    def load(cls,directory:str|Path) -> "FrozenGraphDetector":
        root=Path(directory); record=json.loads((root/FROZEN_FILENAME).read_text())
        payload=dict(record); expected=payload.pop("record_sha256")
        actual=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        if actual!=expected: raise ValueError("frozen graph detector record was modified")
        model,_=load_checkpoint(root/record["checkpoint_file"],record["checkpoint_sha256"])
        return cls(model,GraphDetectorConfig(
            policy=GraphPolicy(**record["config"]["policy"]),
            model=GraphModelSettings(**record["config"]["model"]),
            training=GraphTrainingSettings(**record["config"]["training"]),
            aggregation=record["config"]["aggregation"],maximum_score_age_seconds=record["config"]["maximum_score_age_seconds"],
            top_evidence_edges=record["config"]["top_evidence_edges"]),
            QuantileTailCalibrator.from_dict(record["calibrator"]),float(record["alert_threshold"]),record["model_version"],record["run_id"])

def freeze_detector(directory:str|Path,checkpoint:str|Path,config:GraphDetectorConfig,calibrator:QuantileTailCalibrator,
                    alert_threshold:float,run_id:str,selection:dict[str,Any]) -> dict[str,Any]:
    root=Path(directory); root.mkdir(parents=True,exist_ok=True); checkpoint=Path(checkpoint)
    target=root/"checkpoint.pt"; target.write_bytes(checkpoint.read_bytes())
    sha=hashlib.sha256(target.read_bytes()).hexdigest(); version=f"graph-gae-v1-{config.fingerprint()[:10]}"
    record={"format_version":1,"model_version":version,"run_id":run_id,"checkpoint_file":target.name,
            "checkpoint_sha256":sha,"config":config.to_dict(),"calibrator":calibrator.to_dict(),
            "alert_threshold":alert_threshold,"selection":selection,
            "limitations":["Observed-edge reconstruction is not future-link prediction.","Scores express model deviation, not maliciousness."]}
    record["record_sha256"]=hashlib.sha256(json.dumps(record,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    (root/FROZEN_FILENAME).write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")
    return record
