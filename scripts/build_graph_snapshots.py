#!/usr/bin/env python3
"""Build causal rolling graph snapshots from Task 2.4 raw feature rows."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dualscope.graph.config import GraphDetectorConfig
from dualscope.graph.io import iter_events,save_snapshot
from dualscope.graph.snapshot import build_snapshot
from dualscope.splits import SplitConfig

def main():
 p=argparse.ArgumentParser(); p.add_argument("--features",default="data/raw/lanl_features_days_01_30/raw/events"); p.add_argument("--config",default="config/graph_detector.json"); p.add_argument("--splits",default="config/lanl_splits.json"); p.add_argument("--splits-manifest",default="data/manifests/lanl_splits_v1.json"); p.add_argument("--output",default="data/processed/graph_snapshots_v1"); p.add_argument("--end-day",type=int,default=30); a=p.parse_args()
 cfg=GraphDetectorConfig.from_file(a.config); splits=SplitConfig.from_file(a.splits); root=Path(a.output); first_seen={}; fit_first_seen={}
 split_manifest=json.loads(Path(a.splits_manifest).read_text()); excluded=set(split_manifest["training_exclusions"]["excluded_users"])
 manifest={"config":cfg.to_dict(),"training_excluded_users":sorted(excluded),"snapshots":[]}
 origin=splits.timestamp_start_inclusive
 for cutoff in range(origin+cfg.policy.window_seconds, min(splits.timestamp_end_exclusive,origin+a.end_day*86400)+1,cfg.policy.stride_seconds):
  started=time.time(); window_start=cutoff-cfg.policy.window_seconds
  seen={edge for edge,t in first_seen.items() if t < window_start}; neighbours={}
  for user,computer in seen: neighbours.setdefault(user,set()).add(computer)
  def capture(events,state):
   for row in events:
    edge=(str(row["acting_user"]),str(row["destination_computer"])); state[edge]=min(state.get(edge,int(row["timestamp"])),int(row["timestamp"])); yield row
  snap=build_snapshot(capture(iter_events(a.features,window_start,cutoff),first_seen),cutoff,cfg.policy,seen_edges=seen,prior_neighbours=neighbours)
  day=1+(cutoff-2)//86400; path=save_snapshot(snap,root/f"snapshot_day_{day:02d}.npz")
  if day <= splits.get_split("train").dataset_day_end:
   fit_seen={edge for edge,t in fit_first_seen.items() if t < window_start}; fit_neighbours={}
   for user,computer in fit_seen: fit_neighbours.setdefault(user,set()).add(computer)
   fit=build_snapshot(capture(iter_events(a.features,window_start,cutoff,excluded),fit_first_seen),cutoff,cfg.policy,seen_edges=fit_seen,prior_neighbours=fit_neighbours)
   save_snapshot(fit,root/f"fitting_snapshot_day_{day:02d}.npz")
  manifest["snapshots"].append({"day":day,"path":path.name,"window_start":snap.window_start,"window_end":cutoff,"score_available_at":cutoff,"nodes":snap.n_nodes,"edges":snap.n_edges,"seconds":round(time.time()-started,2)})
 root.mkdir(parents=True,exist_ok=True); (root/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
if __name__=="__main__": main()
