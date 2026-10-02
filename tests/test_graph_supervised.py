import numpy as np
import pandas as pd
import pytest
from dualscope.graph.supervised import FEATURE_COLUMNS,best_f1_threshold,classification_report,feature_matrix


def test_supervised_feature_matrix_is_finite_and_log_scales_counts():
    row={name:1.0 for name in FEATURE_COLUMNS};row["n_edges"]=99
    matrix=feature_matrix(pd.DataFrame([row]))
    assert matrix.shape==(1,len(FEATURE_COLUMNS))
    assert matrix[0,0]==1 and matrix[0,2]==pytest.approx(np.log1p(99))


def test_supervised_threshold_and_report():
    y=[True,False,True,False];scores=[.9,.2,.8,.1];threshold=best_f1_threshold(y,scores);report=classification_report(y,scores,threshold)
    assert report["f1"]==1.0 and report["tp"]==2 and report["fp"]==0
