"""Compact graph snapshot persistence and partitioned feature readers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow.dataset as ds

from dualscope.graph.snapshot import GraphSnapshot

GRAPH_COLUMNS=["timestamp","source_user","destination_user","acting_user","destination_computer","authentication_result","source_line","source_reference"]

def iter_events(features_dir:str|Path,start:int,end:int, excluded_users:set[str] | None=None) -> Iterator[dict]:
    dataset=ds.dataset(str(features_dir),format="parquet",partitioning="hive")
    scanner=dataset.scanner(columns=GRAPH_COLUMNS,filter=(ds.field("timestamp")>=start)&(ds.field("timestamp")<end),batch_size=262_144)
    excluded_users=excluded_users or set()
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            if row["source_user"] not in excluded_users and row["destination_user"] not in excluded_users:
                yield row

def save_snapshot(snapshot:GraphSnapshot,path:str|Path) -> Path:
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    offsets=np.r_[0,np.cumsum([len(x) for x in snapshot.source_lines])]
    lines=np.asarray([v for group in snapshot.source_lines for v in group],dtype=np.int64)
    np.savez_compressed(path,window_start=snapshot.window_start,window_end=snapshot.window_end,
        score_available_at=snapshot.score_available_at,users=np.asarray(snapshot.users),computers=np.asarray(snapshot.computers),
        edge_users=snapshot.edge_users,edge_computers=snapshot.edge_computers,success_counts=snapshot.success_counts,
        failure_counts=snapshot.failure_counts,edge_weights=snapshot.edge_weights,source_line_offsets=offsets,
        source_lines=lines,source_reference_counts=snapshot.source_reference_counts,new_edges=snapshot.new_edges,
        prior_user_degrees=snapshot.prior_user_degrees,current_user_degrees=snapshot.current_user_degrees,
        peak_user_auth_count_60s=snapshot.peak_user_auth_count_60s,
        peak_user_auth_count_300s=snapshot.peak_user_auth_count_300s,
        peak_user_unique_destinations_300s=snapshot.peak_user_unique_destinations_300s)
    return path

def load_snapshot(path:str|Path) -> GraphSnapshot:
    with np.load(path,allow_pickle=False) as d:
        # NPZ members are individually compressed. Cache each member once: indexing
        # ``d["source_lines"]`` inside the edge loop would decompress the complete
        # array for every edge and make production snapshots effectively unloadable.
        off=d["source_line_offsets"]
        lines=d["source_lines"]
        refs=tuple(tuple(int(x) for x in lines[off[i]:off[i+1]]) for i in range(len(off)-1))
        n_users=len(d["users"])
        zeros=np.zeros(n_users,dtype=np.int64)
        return GraphSnapshot(int(d["window_start"]),int(d["window_end"]),int(d["score_available_at"]),
            tuple(d["users"].tolist()),tuple(d["computers"].tolist()),d["edge_users"],d["edge_computers"],
            d["success_counts"],d["failure_counts"],d["edge_weights"],refs,d["source_reference_counts"],
            d["new_edges"],d["prior_user_degrees"],d["current_user_degrees"],
            d["peak_user_auth_count_60s"] if "peak_user_auth_count_60s" in d else zeros,
            d["peak_user_auth_count_300s"] if "peak_user_auth_count_300s" in d else zeros,
            d["peak_user_unique_destinations_300s"] if "peak_user_unique_destinations_300s" in d else zeros)
