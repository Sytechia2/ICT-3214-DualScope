#!/usr/bin/env python3
"""Train and freeze Option A using only graph-derived user-day features."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import joblib
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier,RandomForestClassifier,ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dualscope.graph.supervised import FEATURE_COLUMNS,best_f1_threshold,classification_report,feature_matrix


def main():
    p=argparse.ArgumentParser();p.add_argument("--baseline-scores",default="outputs/graph_evaluation/scores");p.add_argument("--burst-scores",default="outputs/graph_evaluation/burst_scores");p.add_argument("--source-features",default="outputs/graph_evaluation/improvement_trials/source_host_daily_features.parquet");p.add_argument("--labels",default="outputs/graph_evaluation/redteam_labels");p.add_argument("--output",default="outputs/graph_evaluation/supervised_detector");a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    base_cols=["user_id","dataset_day","score","n_edges","new_edge_count","prior_degree","current_degree","degree_growth"]
    burst_cols=["user_id","dataset_day","score","peak_auth_count_60s","peak_auth_count_300s","peak_unique_destinations_300s"]
    base=ds.dataset(a.baseline_scores,format="parquet",partitioning="hive").to_table(columns=base_cols).to_pandas().rename(columns={"score":"baseline_score"})
    burst=ds.dataset(a.burst_scores,format="parquet",partitioning="hive").to_table(columns=burst_cols).to_pandas().rename(columns={"score":"burst_score"})
    source=pq.read_table(a.source_features).to_pandas();frame=base.merge(burst,on=["user_id","dataset_day"]).merge(source,on=["user_id","dataset_day"])
    labels=ds.dataset(a.labels,format="parquet",partitioning="hive").to_table(columns=["user","dataset_day"]).to_pandas().drop_duplicates().rename(columns={"user":"user_id"});labels["label"]=True
    frame=frame.merge(labels,on=["user_id","dataset_day"],how="left");frame["label"]=frame["label"].astype("boolean").fillna(False).astype(bool)
    train=frame[frame.dataset_day.between(8,12)];validation=frame[frame.dataset_day.between(13,16)];test=frame[frame.dataset_day.between(17,30)]
    x_train=feature_matrix(train);y_train=train.label.to_numpy();x_val=feature_matrix(validation);y_val=validation.label.to_numpy();x_test=feature_matrix(test);y_test=test.label.to_numpy()
    candidates={
      "logistic":make_pipeline(StandardScaler(),LogisticRegression(class_weight="balanced",C=.1,max_iter=500,random_state=42)),
      "hist_gradient_boosting":HistGradientBoostingClassifier(max_iter=200,max_leaf_nodes=15,learning_rate=.05,l2_regularization=5,class_weight="balanced",random_state=42),
      "random_forest":RandomForestClassifier(n_estimators=400,max_depth=10,min_samples_leaf=3,class_weight="balanced_subsample",n_jobs=-1,random_state=42,max_features=.8),
      "extra_trees":ExtraTreesClassifier(n_estimators=400,max_depth=12,min_samples_leaf=3,class_weight="balanced",n_jobs=-1,random_state=42,max_features=.8),
    }
    rows=[];best=None
    for name,model in candidates.items():
      model.fit(x_train,y_train);val_scores=model.predict_proba(x_val)[:,1];threshold=best_f1_threshold(y_val,val_scores);val_report=classification_report(y_val,val_scores,threshold)
      row={"model":name,"validation":val_report};rows.append(row);print(json.dumps(row),flush=True)
      key=(val_report["average_precision"],val_report["f1"])
      if best is None or key>best[0]:best=(key,name,model,threshold,val_report)
    _,name,model,threshold,val_report=best;test_scores=model.predict_proba(x_test)[:,1];test_report=classification_report(y_test,test_scores,threshold)
    artifact={"method":"supervised classifier over graph-derived features","training_days":[8,12],"validation_days":[13,16],"test_days":[17,30],"selection_metric":"validation average precision; validation F1 threshold","selected_model":name,"features":list(FEATURE_COLUMNS),"training_units":len(train),"training_positives":int(y_train.sum()),"validation":val_report,"test":test_report,"retrospective_warning":"The test period had been inspected in earlier project experiments; use a fresh holdout for an unbiased final claim.","candidates":rows}
    joblib.dump({"model":model,"threshold":threshold,"features":FEATURE_COLUMNS},out/"supervised_graph_detector.joblib");(out/"metrics.json").write_text(json.dumps(artifact,indent=2)+"\n");print(json.dumps(artifact,indent=2))


if __name__=="__main__":main()
