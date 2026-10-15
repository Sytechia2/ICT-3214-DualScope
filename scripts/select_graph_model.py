#!/usr/bin/env python3
"""Bounded validation-only selection of graph window, latent size and aggregation."""
from __future__ import annotations
import argparse,json,sys
from dataclasses import replace
from pathlib import Path
import numpy as np
import pyarrow.dataset as ds
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dualscope.graph.config import GraphDetectorConfig
from dualscope.graph.detector import freeze_detector
from dualscope.graph.evaluation import label_snapshot_cutoff
from dualscope.graph.io import load_snapshot
from dualscope.graph.scoring import score_snapshot
from dualscope.graph.training import save_checkpoint,train_autoencoder
from dualscope.sequence.calibration import QuantileTailCalibrator,average_precision,best_f1_threshold
from dualscope.splits import SplitConfig

def main():
 p=argparse.ArgumentParser(); p.add_argument("--candidate",action="append",required=True,metavar="SECONDS=SNAPSHOT_DIR"); p.add_argument("--labels",required=True); p.add_argument("--config",default="config/graph_detector.json"); p.add_argument("--splits",default="config/lanl_splits.json"); p.add_argument("--output",default="models/graph_detector_v1"); a=p.parse_args()
 raw_cfg=json.loads(Path(a.config).read_text()); base=GraphDetectorConfig.from_file(a.config); split_cfg=SplitConfig.from_file(a.splits); validation=split_cfg.get_split("validation")
 label_rows=ds.dataset(a.labels,format="parquet",partitioning="hive").to_table(columns=["timestamp","user"],filter=(ds.field("timestamp")>=validation.timestamp_start)&(ds.field("timestamp")<validation.timestamp_end)).to_pylist()
 candidates=[]; root=Path(a.output); trials=root/"trials"; trials.mkdir(parents=True,exist_ok=True)
 for item in a.candidate:
  seconds_s,directory=item.split("=",1); seconds=int(seconds_s)
  train_paths=sorted(Path(directory).glob("fitting_snapshot_day_0[2-7].npz")); val_paths=[x for x in sorted(Path(directory).glob("snapshot_day_*.npz")) if 8<=1+(load_snapshot(x).window_end-2)//86400<=16]
  if not train_paths or not val_paths: raise ValueError(f"candidate {seconds} lacks training or validation snapshots")
  train=[load_snapshot(x) for x in train_paths]; val=[load_snapshot(x) for x in val_paths]
  positives={(str(x["user"]),label_snapshot_cutoff(int(x["timestamp"]),stride_seconds=base.policy.stride_seconds)) for x in label_rows}
  for latent in raw_cfg["search"]["latent_sizes"]:
   cfg=replace(base,policy=replace(base.policy,window_seconds=seconds),model=replace(base.model,latent_size=int(latent)))
   model,history=train_autoencoder(train,cfg.model,cfg.training)
   checkpoint=trials/f"w{seconds}-z{latent}.pt"; sha=save_checkpoint(checkpoint,model,{"history":history,"window_seconds":seconds,"latent_size":latent})
   for aggregation in raw_cfg["search"]["aggregations"]:
    scores=[]; keys=[]
    for snapshot in val:
     _,user_scores=score_snapshot(model,snapshot,aggregation); scores.extend(user_scores); keys.extend((u,snapshot.window_end) for u in snapshot.users)
    scores=np.asarray(scores); calibrator=QuantileTailCalibrator.fit(scores,min_reference=min(100,len(scores))); normal=calibrator.transform(scores); labels=np.asarray([k in positives for k in keys],dtype=bool)
    ap=average_precision(labels,normal); threshold=best_f1_threshold(labels,normal) if labels.any() else {"threshold":float(np.quantile(normal,.999)),"f1":None}
    candidates.append({"window_seconds":seconds,"latent_size":latent,"aggregation":aggregation,"average_precision":ap,"threshold":threshold,"checkpoint":str(checkpoint),"checkpoint_sha256":sha,"calibrator":calibrator.to_dict()})
 best=max(candidates,key=lambda x:x["average_precision"] if x["average_precision"] is not None else -1); selected=replace(base,policy=replace(base.policy,window_seconds=best["window_seconds"]),model=replace(base.model,latent_size=best["latent_size"]),aggregation=best["aggregation"])
 record=freeze_detector(root,best["checkpoint"],selected,QuantileTailCalibrator.from_dict(best["calibrator"]),best["threshold"]["threshold"],"graph-selection-v1",{"selection_split":"validation","metric":"average_precision","winner":{k:v for k,v in best.items() if k not in {"calibrator"}}})
 (root/"selection.json").write_text(json.dumps({"rule":"highest validation average precision; test labels unused","candidates":candidates,"selected_model_version":record["model_version"]},indent=2)+"\n")
if __name__=="__main__": main()
