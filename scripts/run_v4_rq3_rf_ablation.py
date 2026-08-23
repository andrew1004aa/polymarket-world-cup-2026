#!/usr/bin/env python3
"""Grouped nested-CV feature-family ablations for the RQ3 random forest."""
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "model_samples/v4/rq3_ml/rq3_match_pre_full_reversal_30m.csv.gz"
OUT = ROOT / "ml_results/v4/rq3_match_pre/ablation"
TREE_RESULTS = ROOT / "ml_results/v4/rq3_match_pre/trees"
SEED = 20260813
TARGET = "full_reversal_30m"
BASELINE = ["baseline_yes_price", "initial_directional_move_5m"]
FAMILIES = {
    "trade_size": ["log_initiating_trade_value"],
    "concentration": [
        "initiating_minute_top1_wallet_share", "initiating_minute_top3_wallet_share",
        "initiating_minute_wallet_hhi", "initiating_minute_absolute_flow_imbalance",
    ],
    "following": ["follower_imbalance_5m", "follower_signed_log_net_5m"],
}
ALL_INCREMENTAL = sum(FAMILIES.values(), [])
GRID = [
    {"max_depth": 4, "min_samples_leaf": 20, "max_features": "sqrt"},
    {"max_depth": 8, "min_samples_leaf": 20, "max_features": "sqrt"},
    {"max_depth": None, "min_samples_leaf": 20, "max_features": "sqrt"},
    {"max_depth": 8, "min_samples_leaf": 50, "max_features": 1.0},
]


def make_model(parameters):
    return RandomForestClassifier(
        n_estimators=500, n_jobs=2, random_state=SEED,
        class_weight="balanced_subsample", **parameters)


def threshold(y, probability):
    candidates = np.unique(np.quantile(probability, np.linspace(0.02, 0.98, 97)))
    scores = [f1_score(y, probability >= value, zero_division=0) for value in candidates]
    return float(candidates[int(np.argmax(scores))])


def tune(x, y, groups):
    inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED + 1)
    best = None
    for order, parameters in enumerate(GRID):
        oof = np.full(len(y), np.nan)
        for train, valid in inner.split(x, y, groups):
            model = make_model(parameters).fit(x.iloc[train], y.iloc[train])
            oof[valid] = model.predict_proba(x.iloc[valid])[:, 1]
        score = float(average_precision_score(y, oof))
        if best is None or score > best[0]:
            best = (score, -order, parameters, oof)
    score, _, parameters, oof = best
    return parameters, threshold(y, oof), score


def metrics(y, probability, cutoff):
    predicted = probability >= cutoff
    return {
        "pr_auc": average_precision_score(y, probability),
        "roc_auc": roc_auc_score(y, probability),
        "f1": f1_score(y, predicted, zero_division=0),
        "precision": precision_score(y, predicted, zero_division=0),
        "recall": recall_score(y, predicted, zero_division=0),
        "brier": brier_score_loss(y, probability), "threshold": cutoff,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(SRC)
    y, groups = data[TARGET].astype(int), data.event_id.astype(str)
    specifications = {
        f"without_{family}": BASELINE + [feature for feature in ALL_INCREMENTAL
                                         if feature not in removed]
        for family, removed in FAMILIES.items()
    }
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_rows, tuning_rows, predictions = [], [], []
    for fold, (train, test) in enumerate(outer.split(data, y, groups), 1):
        print(f"Outer fold {fold}/5", flush=True)
        for specification, columns in specifications.items():
            parameters, cutoff, inner_pr = tune(
                data.iloc[train][columns], y.iloc[train], groups.iloc[train])
            model = make_model(parameters).fit(data.iloc[train][columns], y.iloc[train])
            probability = model.predict_proba(data.iloc[test][columns])[:, 1]
            fold_rows.append({"fold": fold, "model": "random_forest",
                              "feature_set": specification,
                              **metrics(y.iloc[test], probability, cutoff)})
            tuning_rows.append({"fold": fold, "feature_set": specification,
                                "parameters": json.dumps(parameters, sort_keys=True),
                                "inner_pr_auc": inner_pr, "f1_threshold": cutoff})
            for index, value in zip(test, probability):
                predictions.append({"row_index": int(index), "fold": fold,
                                    "feature_set": specification,
                                    "y_true": int(y.iloc[index]), "probability": float(value)})
    folds = pd.DataFrame(fold_rows)
    # Reuse the already-completed full and baseline RF folds produced with the same split and grid.
    original = pd.read_csv(TREE_RESULTS / "fold_metrics.csv")
    original = original[original.model.eq("random_forest")].copy()
    folds = pd.concat([original, folds], ignore_index=True)
    folds.to_csv(OUT / "fold_metrics.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(OUT / "tuning.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "out_of_fold_predictions.csv.gz", index=False,
                                     compression="gzip")
    names = ["pr_auc", "roc_auc", "f1", "precision", "recall", "brier"]
    aggregate = folds.groupby("feature_set")[names].agg(["mean", "std"])
    aggregate.columns = [f"{left}_{right}" for left, right in aggregate.columns]
    aggregate = aggregate.reset_index()
    full = aggregate[aggregate.feature_set.eq("expanded")].iloc[0]
    for metric in ("pr_auc", "roc_auc", "f1"):
        aggregate[f"loss_vs_full_{metric}"] = full[f"{metric}_mean"] - aggregate[f"{metric}_mean"]
    aggregate["brier_loss_vs_full"] = aggregate.brier_mean - full.brier_mean
    aggregate.to_csv(OUT / "aggregate_metrics.csv", index=False)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_qc_passed", "rows": len(data), "groups": int(groups.nunique()),
        "seed": SEED, "model": "random_forest", "families": FAMILIES,
        "outer_cv": "5-fold StratifiedGroupKFold by event_id",
        "inner_cv": "3-fold StratifiedGroupKFold by event_id",
        "software": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
        "aggregate": aggregate.to_dict(orient="records"),
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(aggregate.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
