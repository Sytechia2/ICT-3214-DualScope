#!/usr/bin/env python3
"""Export frozen Task 4.4 scores and relationship evidence."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dualscope.graph.detector import FrozenGraphDetector
from dualscope.graph.export import build_score_table,write_score_table
from dualscope.graph.io import load_snapshot
from dualscope.splits import SplitConfig
def main():
 p=argparse.ArgumentParser(); p.add_argument("--snapshots",default="data/processed/graph_snapshots_v1"); p.add_argument("--detector",default="models/graph_detector_v1"); p.add_argument("--splits",default="config/lanl_splits.json"); p.add_argument("--output",default="outputs/graph_scores_v1"); a=p.parse_args(); detector=FrozenGraphDetector.load(a.detector); splits=SplitConfig.from_file(a.splits)
 for path in sorted(Path(a.snapshots).glob("snapshot_day_*.npz")):
  s=load_snapshot(path); day=1+(s.window_end-2)//86400; split=splits.find_split_for_timestamp(s.window_end-1)
  edge,raw,norm=detector.score(s); status="insufficient_history" if s.window_end<=splits.timestamp_start_inclusive+detector.config.policy.warmup_seconds else "available"
  table=build_score_table(s,edge,raw,norm,model_version=detector.model_version,run_id=detector.run_id,split=split.name if split else None,aggregation=detector.config.aggregation,alert_threshold=detector.alert_threshold,top_edges=detector.config.top_evidence_edges,status=status,status_detail="warm-up history incomplete" if status!="available" else "")
  write_score_table(table,a.output,day)
if __name__=="__main__": main()
