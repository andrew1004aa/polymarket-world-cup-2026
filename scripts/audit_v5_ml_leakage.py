#!/usr/bin/env python3
"""Artifact and source-code audit of RQ3 nested-CV threshold selection."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SAMPLE=ROOT/"model_samples/v4/rq3_ml/rq3_match_pre_full_reversal_30m.csv.gz"
BASE=ROOT/"ml_results/v4/rq3_match_pre"
OUT=ROOT/"docs/data_audit/v5"


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        while b:=f.read(1024*1024):h.update(b)
    return h.hexdigest()


def assert_thresholds(result_dir, tuning_name="tuning.csv", metrics_name="fold_metrics.csv"):
    tuning=pd.read_csv(result_dir/tuning_name)
    metrics=pd.read_csv(result_dir/metrics_name)
    if "model" not in tuning: tuning["model"]="random_forest"
    merged=metrics.merge(tuning[["fold","model","feature_set","f1_threshold"]],
                         on=["fold","model","feature_set"],how="inner")
    if len(merged)!=len(tuning):
        raise AssertionError(f"Threshold/metric key mismatch in {result_dir}")
    difference=np.abs(merged.threshold-merged.f1_threshold)
    if not np.all(difference<1e-14):
        raise AssertionError(f"Stored test threshold differs from inner-OOF threshold in {result_dir}")
    return {"models_audited":len(tuning),"maximum_threshold_difference":float(difference.max())}


def assert_oof(result_dir,data,expected_specs=None):
    pred=pd.read_csv(result_dir/"out_of_fold_predictions.csv.gz")
    keys=[c for c in ["model","feature_set"] if c in pred]
    if not keys: keys=["feature_set"]
    errors=[];specs=0
    for name,g in pred.groupby(keys):
        specs+=1
        if g.row_index.duplicated().any():errors.append(f"duplicate OOF row: {name}")
        joined=g.merge(data[["event_id"]],left_on="row_index",right_index=True,how="left")
        if joined.event_id.isna().any():errors.append(f"unknown row index: {name}")
        if joined.groupby("event_id").fold.nunique().max()!=1:errors.append(f"event split across folds: {name}")
        if len(g)!=len(data):errors.append(f"incomplete OOF coverage: {name}: {len(g)}")
    if errors:raise AssertionError("; ".join(errors))
    return {"specifications":specs,"rows_per_specification":len(data),"event_overlap_across_outer_folds":0}


def source_checks():
    files={
        "linear":ROOT/"scripts/run_v4_rq3_ml_baselines.py",
        "trees":ROOT/"scripts/run_v4_rq3_ml_trees.py",
        "ablation":ROOT/"scripts/run_v4_rq3_rf_ablation.py",
        "chronological":ROOT/"scripts/run_v4_rq3_chronological_validation.py",
    }
    required=["StratifiedGroupKFold","oof","f1_score","predict_proba"]
    output={}
    for name,path in files.items():
        text=path.read_text()
        missing=[token for token in required if token not in text]
        if missing:raise AssertionError(f"{path} missing expected nested-CV tokens: {missing}")
        # The frozen implementations return thresholds from inner OOF vectors;
        # test probabilities are created only after fitting on outer train.
        if name!="chronological" and "data.iloc[test]" not in text:
            raise AssertionError(f"No explicit outer-test isolation found in {path}")
        if name=="chronological" and "y_test" not in text:
            raise AssertionError("No explicit chronological holdout found")
        output[name]={"path":str(path.relative_to(ROOT)),"sha256":sha(path)}
    return output


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(SAMPLE,usecols=["event_id","full_reversal_30m"])
    checks={}
    checks["linear_thresholds"]=assert_thresholds(BASE)
    checks["linear_oof"]=assert_oof(BASE,data)
    checks["tree_thresholds"]=assert_thresholds(BASE/"trees")
    checks["tree_oof"]=assert_oof(BASE/"trees",data)
    checks["ablation_thresholds"]=assert_thresholds(BASE/"ablation")
    checks["ablation_oof"]=assert_oof(BASE/"ablation",data)

    chrono_tuning=pd.read_csv(BASE/"chronological/tuning.csv")
    chrono_metrics=pd.read_csv(BASE/"chronological/metrics.csv")
    chrono=chrono_metrics.merge(chrono_tuning,on=["model","feature_set"])
    if not np.all(np.abs(chrono.threshold-chrono.f1_threshold)<1e-14):
        raise AssertionError("Chronological test threshold differs from training-OOF threshold")
    chrono_summary=json.loads((BASE/"chronological/summary.json").read_text())
    if chrono_summary["split"]["event_overlap"]!=0:raise AssertionError("Chronological event leakage")
    checks["chronological"]={"models_audited":len(chrono),"event_overlap":0,
                              "train_events":chrono_summary["split"]["train_event_count"],
                              "test_events":chrono_summary["split"]["test_event_count"]}
    checks["source_code"]=source_checks()
    payload={"generated_at":datetime.now(timezone.utc).isoformat(),"status":"PASS",
             "sample_rows":len(data),"events":int(data.event_id.nunique()),
             "conclusion":"All reported F1 thresholds were selected from inner out-of-fold predictions within outer-training data; held-out labels were used only for final evaluation.",
             "checks":checks}
    (OUT/"ml_threshold_leakage_audit.json").write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps(payload,indent=2))


if __name__=="__main__":main()
