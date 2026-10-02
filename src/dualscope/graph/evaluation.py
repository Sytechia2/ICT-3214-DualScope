"""Graph-window label alignment and imbalanced classification evaluation."""
from __future__ import annotations
import numpy as np
from dualscope.sequence.calibration import evaluation_summary

def label_snapshot_cutoff(timestamp:int, *, origin:int=1, stride_seconds:int=86_400) -> int:
    """First causal snapshot cutoff strictly after an event timestamp."""
    return origin + ((int(timestamp)-origin)//stride_seconds + 1)*stride_seconds

def evaluate_records(records:list[dict], labels:list[dict], *, threshold:float,
                     origin:int=1, stride_seconds:int=86_400) -> dict:
    available=[r for r in records if r.get("status")=="available" and r.get("score") is not None]
    positives={(str(x["user"]),label_snapshot_cutoff(int(x["timestamp"]),origin=origin,stride_seconds=stride_seconds)) for x in labels}
    y=np.asarray([(str(r["user_id"]),int(r["window_end"])) in positives for r in available],dtype=bool)
    scores=np.asarray([r["score"] for r in available],dtype=np.float64)
    return evaluation_summary(y,scores,len(positives),threshold)
