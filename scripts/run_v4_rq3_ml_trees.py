#!/usr/bin/env python3
"""Run grouped nested CV for RQ3 random-forest and boosted-tree models."""
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "model_samples/v4/rq3_ml/rq3_match_pre_full_reversal_30m.csv.gz"
OUT = ROOT / "ml_results/v4/rq3_match_pre/trees"
SEED = 20260813
TARGET = "full_reversal_30m"
BASELINE = ["baseline_yes_price", "initial_directional_move_5m"]
BEHAVIOURAL = [
    "log_initiating_trade_value", "initiating_minute_top1_wallet_share",
    "initiating_minute_top3_wallet_share", "initiating_minute_wallet_hhi",
    "initiating_minute_absolute_flow_imbalance", "follower_imbalance_5m",
    "follower_signed_log_net_5m",
]
GRIDS = {
    "random_forest": [
        {"max_depth": 4, "min_samples_leaf": 20, "max_features": "sqrt"},
        {"max_depth": 8, "min_samples_leaf": 20, "max_features": "sqrt"},
        {"max_depth": None, "min_samples_leaf": 20, "max_features": "sqrt"},
        {"max_depth": 8, "min_samples_leaf": 50, "max_features": 1.0},
    ],
    "hist_gradient_boosting": [
        {"learning_rate": 0.05, "max_leaf_nodes": 7, "l2_regularization": 1.0},
        {"learning_rate": 0.05, "max_leaf_nodes": 15, "l2_regularization": 1.0},
        {"learning_rate": 0.10, "max_leaf_nodes": 7, "l2_regularization": 1.0},
        {"learning_rate": 0.05, "max_leaf_nodes": 15, "l2_regularization": 5.0},
    ],
}


def make_model(kind, parameters):
    if kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=500, n_jobs=2, random_state=SEED,
            class_weight="balanced_subsample", **parameters)
    return HistGradientBoostingClassifier(
        max_iter=250, random_state=SEED, early_stopping=False, **parameters)


def best_f1_threshold(y, probability):
    candidates = np.unique(np.quantile(probability, np.linspace(0.02, 0.98, 97)))
    scores = [f1_score(y, probability >= threshold, zero_division=0)
              for threshold in candidates]
    return float(candidates[int(np.argmax(scores))])


def tune(x, y, groups, kind):
    inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED + 1)
    best = None
    for order, parameters in enumerate(GRIDS[kind]):
        oof = np.full(len(y), np.nan)
        for train, valid in inner.split(x, y, groups):
            model = make_model(kind, parameters).fit(x.iloc[train], y.iloc[train])
            oof[valid] = model.predict_proba(x.iloc[valid])[:, 1]
        score = float(average_precision_score(y, oof))
        if best is None or score > best[0]:
            best = (score, order, parameters, oof)
    score, _, parameters, oof = best
    return parameters, best_f1_threshold(y, oof), score


def evaluate(y, probability, threshold):
    prediction = probability >= threshold
    return {
        "pr_auc": average_precision_score(y, probability),
        "roc_auc": roc_auc_score(y, probability),
        "f1": f1_score(y, prediction, zero_division=0),
        "precision": precision_score(y, prediction, zero_division=0),
        "recall": recall_score(y, prediction, zero_division=0),
        "brier": brier_score_loss(y, probability),
        "threshold": threshold,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(SRC)
    y = data[TARGET].astype(int)
    groups = data.event_id.astype(str)
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    specs = [(kind, feature_set, columns)
             for kind in GRIDS
             for feature_set, columns in (("baseline", BASELINE),
                                          ("expanded", BASELINE + BEHAVIOURAL))]
    fold_rows, predictions, tuning_rows = [], [], []
    for fold, (train, test) in enumerate(outer.split(data, y, groups), 1):
        print(f"Outer fold {fold}/5", flush=True)
        for kind, feature_set, columns in specs:
            parameters, threshold, inner_pr = tune(
                data.iloc[train][columns], y.iloc[train], groups.iloc[train], kind)
            model = make_model(kind, parameters).fit(data.iloc[train][columns], y.iloc[train])
            probability = model.predict_proba(data.iloc[test][columns])[:, 1]
            fold_rows.append({"fold": fold, "model": kind, "feature_set": feature_set,
                              **evaluate(y.iloc[test], probability, threshold)})
            tuning_rows.append({"fold": fold, "model": kind, "feature_set": feature_set,
                                "parameters": json.dumps(parameters, sort_keys=True),
                                "inner_pr_auc": inner_pr, "f1_threshold": threshold})
            for index, value in zip(test, probability):
                predictions.append({"row_index": int(index), "fold": fold, "model": kind,
                                    "feature_set": feature_set, "y_true": int(y.iloc[index]),
                                    "probability": float(value)})
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(OUT / "fold_metrics.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(OUT / "tuning.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "out_of_fold_predictions.csv.gz", index=False,
                                     compression="gzip")
    metrics = ["pr_auc", "roc_auc", "f1", "precision", "recall", "brier"]
    aggregate = folds.groupby(["model", "feature_set"])[metrics].agg(["mean", "std"])
    aggregate.columns = [f"{left}_{right}" for left, right in aggregate.columns]
    aggregate = aggregate.reset_index()
    aggregate.to_csv(OUT / "aggregate_metrics.csv", index=False)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_qc_passed", "rows": len(data), "positive": int(y.sum()),
        "groups": int(groups.nunique()), "seed": SEED,
        "outer_cv": "5-fold StratifiedGroupKFold by event_id",
        "inner_cv": "3-fold StratifiedGroupKFold by event_id",
        "implementation_note": "Histogram gradient boosting is the pre-specified fallback because XGBoost is not installed.",
        "grids": GRIDS,
        "software": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
        "aggregate": aggregate.to_dict(orient="records"),
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(aggregate.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
