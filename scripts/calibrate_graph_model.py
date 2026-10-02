#!/usr/bin/env python3
"""Fit validation-only graph score scaling and select a max-F1 alert threshold."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dualscope.graph.config import GraphDetectorConfig
from dualscope.graph.detector import freeze_detector
from dualscope.graph.io import load_snapshot
from dualscope.graph.scoring import score_snapshot
from dualscope.graph.training import load_checkpoint
from dualscope.sequence.calibration import QuantileTailCalibrator,best_f1_threshold
def main():
 p=argparse.ArgumentParser(); p.add_argument("--snapshots",default="data/processed/graph_snapshots_v1"); p.add_argument("--checkpoint",default="models/graph_gae_v1/checkpoint.pt"); p.add_argument("--config",default="config/graph_detector.json"); p.add_argument("--positive-units",help="JSON list of [user_id, window_end] validation positives"); p.add_argument("--output",default="models/graph_detector_v1"); a=p.parse_args()
 cfg=GraphDetectorConfig.from_file(a.config); model,_=load_checkpoint(a.checkpoint); raw=[]; keys=[]
 for path in sorted(Path(a.snapshots).glob("snapshot_day_*.npz")):
  s=load_snapshot(path); day=1+(s.window_end-2)//86400
  if 8<=day<=16:
   _,scores=score_snapshot(model,s,cfg.aggregation); raw.extend(scores); keys.extend((u,s.window_end) for u in s.users)
 raw=np.asarray(raw); cal=QuantileTailCalibrator.fit(raw,min_reference=min(100,len(raw))); norm=cal.transform(raw)
 if a.positive_units:
  positives={tuple(x) for x in json.loads(Path(a.positive_units).read_text())}; labels=np.asarray([k in positives for k in keys]); chosen=best_f1_threshold(labels,norm); threshold=chosen["threshold"]
 else: chosen={"rule":"99.9th validation quantile (labels not supplied)"}; threshold=float(np.quantile(norm,.999))
 freeze_detector(a.output,a.checkpoint,cfg,cal,threshold,"graph-v1",chosen)
if __name__=="__main__": main()
