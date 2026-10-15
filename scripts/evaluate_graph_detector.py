#!/usr/bin/env python3
"""Report precision/recall/F1, PR-AUC and ROC-AUC for frozen graph scores."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import pyarrow.dataset as ds
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dualscope.graph.evaluation import evaluate_records
from dualscope.splits import SplitConfig
def main():
 p=argparse.ArgumentParser(); p.add_argument("--scores",required=True); p.add_argument("--labels",required=True); p.add_argument("--split",choices=["validation","test"],required=True); p.add_argument("--splits",default="config/lanl_splits.json"); p.add_argument("--output"); a=p.parse_args()
 scores=ds.dataset(a.scores,format="parquet",partitioning="hive").to_table(filter=ds.field("split")==a.split).to_pylist()
 interval=SplitConfig.from_file(a.splits).get_split(a.split)
 labels=ds.dataset(a.labels,format="parquet",partitioning="hive").to_table(columns=["timestamp","user"],filter=(ds.field("timestamp")>=interval.timestamp_start)&(ds.field("timestamp")<interval.timestamp_end)).to_pylist()
 threshold=next((float(r["alert_threshold"]) for r in scores if r.get("alert_threshold") is not None),None)
 if threshold is None: raise ValueError("score rows do not contain a frozen alert threshold")
 result={"split":a.split,"metrics":evaluate_records(scores,labels,threshold=threshold),"note":"accuracy intentionally omitted because labels are highly imbalanced"}
 text=json.dumps(result,indent=2)+"\n"; print(text,end="")
 if a.output: Path(a.output).write_text(text)
if __name__=="__main__": main()
