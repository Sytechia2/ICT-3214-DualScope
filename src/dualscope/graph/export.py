"""Common graph score schema and evidence export (Task 4.4)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from dualscope.graph.snapshot import GraphSnapshot


EDGE_EVIDENCE = pa.struct([
    ("destination_computer", pa.string()), ("edge_raw_score", pa.float64()),
    ("edge_weight", pa.float64()), ("success_count", pa.int64()), ("failure_count", pa.int64()),
    ("is_new_edge", pa.bool_()), ("source_lines", pa.list_(pa.int64())),
    ("source_reference_count", pa.int64()), ("references_truncated", pa.bool_()),
])
GRAPH_SCORE_SCHEMA = pa.schema([
    ("detector",pa.string()), ("model_version",pa.string()), ("run_id",pa.string()),
    ("user_id",pa.string()), ("window_start",pa.int64()), ("window_end",pa.int64()),
    ("score_available_at",pa.int64()), ("dataset_day",pa.int32()), ("split",pa.string()),
    ("status",pa.string()), ("status_detail",pa.string()), ("raw_score",pa.float64()),
    ("score",pa.float64()), ("alert_threshold",pa.float64()), ("is_alert",pa.bool_()),
    ("aggregation",pa.string()), ("n_edges",pa.int32()), ("new_edge_count",pa.int32()),
    ("prior_degree",pa.int32()), ("current_degree",pa.int32()), ("degree_growth",pa.int32()),
    ("peak_auth_count_60s",pa.int32()), ("peak_auth_count_300s",pa.int32()),
    ("peak_unique_destinations_300s",pa.int32()),
    ("evidence_nodes",pa.list_(pa.string())), ("top_edges",pa.list_(EDGE_EVIDENCE)),
], metadata={"schema_version":"1.0.0","score_semantics":"0-1 validation-relative rarity; not an attack probability",
             "reference_format":"auth.txt:<source_line>"})


def build_score_table(snapshot: GraphSnapshot, edge_raw: np.ndarray, user_raw: np.ndarray,
                      normalised: np.ndarray, *, model_version: str, run_id: str, split: str,
                      aggregation: str, alert_threshold: float, top_edges: int=5,
                      status: str="available", status_detail: str="") -> pa.Table:
    rows=[]
    for u,user in enumerate(snapshot.users):
        idx=np.flatnonzero(snapshot.edge_users==u)
        order=idx[np.argsort(-edge_raw[idx],kind="stable")][:top_edges]
        available=status=="available" and np.isfinite(user_raw[u]) and np.isfinite(normalised[u])
        evidence=[]
        for i in order:
            lines=list(snapshot.source_lines[i]); count=int(snapshot.source_reference_counts[i])
            evidence.append({"destination_computer":snapshot.computers[int(snapshot.edge_computers[i])],
                "edge_raw_score":float(edge_raw[i]),"edge_weight":float(snapshot.edge_weights[i]),
                "success_count":int(snapshot.success_counts[i]),"failure_count":int(snapshot.failure_counts[i]),
                "is_new_edge":bool(snapshot.new_edges[i]),"source_lines":lines,"source_reference_count":count,
                "references_truncated":count>len(lines)})
        score=float(normalised[u]) if available else None
        rows.append({"detector":"graph_autoencoder","model_version":model_version,"run_id":run_id,
            "user_id":user,"window_start":snapshot.window_start,"window_end":snapshot.window_end,
            "score_available_at":snapshot.score_available_at,"dataset_day":1+(snapshot.window_end-2)//86400,
            "split":split,"status":status,"status_detail":status_detail,"raw_score":float(user_raw[u]) if available else None,
            "score":score,"alert_threshold":alert_threshold,"is_alert":score>=alert_threshold if available else None,
            "aggregation":aggregation,"n_edges":int(len(idx)),"new_edge_count":int(snapshot.new_edges[idx].sum()),
            "prior_degree":int(snapshot.prior_user_degrees[u]),"current_degree":int(snapshot.current_user_degrees[u]),
            "degree_growth":int(snapshot.current_user_degrees[u]-snapshot.prior_user_degrees[u]),
            "peak_auth_count_60s":int(snapshot.peak_user_auth_count_60s[u]),
            "peak_auth_count_300s":int(snapshot.peak_user_auth_count_300s[u]),
            "peak_unique_destinations_300s":int(snapshot.peak_user_unique_destinations_300s[u]),
            "evidence_nodes":[user]+[e["destination_computer"] for e in evidence],"top_edges":evidence})
    return pa.Table.from_pylist(rows,schema=GRAPH_SCORE_SCHEMA)


def write_score_table(table: pa.Table, output_dir: str | Path, day: int) -> Path:
    path=Path(output_dir)/f"dataset_day={day:02d}"/"part-000000.parquet"; path.parent.mkdir(parents=True,exist_ok=True)
    pq.write_table(table.drop_columns(["dataset_day"]),path,compression="zstd"); return path


class GraphScoreStore:
    def __init__(self, scores_dir: str | Path) -> None:
        self.dataset=ds.dataset(str(scores_dir),format="parquet",partitioning="hive")
    def table(self, columns=None, filter=None): return self.dataset.to_table(columns=columns,filter=filter)
    def get(self,user_id:str,window_end:int) -> dict[str,Any] | None:
        t=self.table(filter=(ds.field("user_id")==user_id)&(ds.field("window_end")==window_end))
        if t.num_rows>1: raise ValueError("duplicate graph score")
        return t.to_pylist()[0] if t.num_rows else None
