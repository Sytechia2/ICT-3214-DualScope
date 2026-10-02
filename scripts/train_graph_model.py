#!/usr/bin/env python3
"""Train Task 4.2 on training-split graph snapshots."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dualscope.graph.config import GraphDetectorConfig
from dualscope.graph.io import load_snapshot
from dualscope.graph.training import save_checkpoint,train_autoencoder
def main():
 p=argparse.ArgumentParser(); p.add_argument("--snapshots",default="data/processed/graph_snapshots_v1"); p.add_argument("--config",default="config/graph_detector.json"); p.add_argument("--output",default="models/graph_gae_v1"); a=p.parse_args(); cfg=GraphDetectorConfig.from_file(a.config)
 paths=sorted(Path(a.snapshots).glob("fitting_snapshot_day_0[2-7].npz")); snapshots=[load_snapshot(x) for x in paths]
 root=Path(a.output); root.mkdir(parents=True,exist_ok=True); progress=root/"training_progress.jsonl"
 progress.write_text("")
 def log(record):
  with progress.open("a",encoding="utf-8") as handle:
   handle.write(json.dumps(record,sort_keys=True)+"\n"); handle.flush()
  print(json.dumps(record,sort_keys=True),flush=True)
 model,history=train_autoencoder(snapshots,cfg.model,cfg.training,log=log)
 sha=save_checkpoint(root/"checkpoint.pt",model,{"config":cfg.to_dict(),"training_snapshots":[p.name for p in paths],"history":history})
 (root/"training_log.json").write_text(json.dumps({"checkpoint_sha256":sha,"history":history},indent=2)+"\n")
if __name__=="__main__": main()
