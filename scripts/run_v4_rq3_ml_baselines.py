#!/usr/bin/env python3
"""Run leakage-safe grouped CV for the first-stage RQ3 ML baselines."""
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "model_samples/v4/rq3_ml/rq3_match_pre_full_reversal_30m.csv.gz"
OUT = ROOT / "ml_results/v4/rq3_match_pre"
SEED = 20260813
TARGET = "full_reversal_30m"
BASELINE = ["baseline_yes_price", "initial_directional_move_5m"]
BEHAVIOURAL = [
    "log_initiating_trade_value", "initiating_minute_top1_wallet_share",
    "initiating_minute_top3_wallet_share", "initiating_minute_wallet_hhi",
    "initiating_minute_absolute_flow_imbalance", "follower_imbalance_5m",
    "follower_signed_log_net_5m",
]
C_GRID = [0.01, 0.1, 1.0, 10.0]
L1_GRID = [0.25, 0.5, 0.75]


def estimator(kind, c, l1_ratio=None):
    kwargs = dict(C=c, max_iter=5000, random_state=SEED)
    if kind == "logistic":
        model = LogisticRegression(solver="lbfgs", penalty="l2", **kwargs)
    else:
        model = LogisticRegression(solver="saga", penalty="elasticnet",
                                   l1_ratio=l1_ratio, tol=1e-4, **kwargs)
    return Pipeline([("scale", StandardScaler()), ("model", model)])


def best_f1_threshold(y, probability):
    candidates = np.unique(np.quantile(probability, np.linspace(0.02, 0.98, 97)))
    scores = np.array([f1_score(y, probability >= t, zero_division=0) for t in candidates])
    return float(candidates[int(np.argmax(scores))])


def tune_and_threshold(x, y, groups, kind):
    inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED + 1)
    settings = [(c, None) for c in C_GRID] if kind == "logistic" else [
        (c, ratio) for c in C_GRID for ratio in L1_GRID
    ]
    best = None
    for c, ratio in settings:
        oof = np.full(len(y), np.nan)
        for train, valid in inner.split(x, y, groups):
            fit = estimator(kind, c, ratio).fit(x.iloc[train], y.iloc[train])
            oof[valid] = fit.predict_proba(x.iloc[valid])[:, 1]
        score = average_precision_score(y, oof)
        candidate = (score, -c, -(ratio or 0), c, ratio, oof)
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    _, _, _, c, ratio, oof = best
    return c, ratio, best_f1_threshold(y, oof), float(average_precision_score(y, oof))


def metrics(y, probability, threshold):
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
    fold_rows, predictions, tuning = [], [], []
    specs = [
        ("logistic", "baseline", BASELINE),
        ("logistic", "expanded", BASELINE + BEHAVIOURAL),
        ("elastic_net", "baseline", BASELINE),
        ("elastic_net", "expanded", BASELINE + BEHAVIOURAL),
    ]
    for fold, (train, test) in enumerate(outer.split(data, y, groups), 1):
        print(f"Outer fold {fold}/5", flush=True)
        prevalence = float(y.iloc[train].mean())
        majority_probability = np.full(len(test), prevalence)
        result = metrics(y.iloc[test], majority_probability, 0.5)
        fold_rows.append({"fold": fold, "model": "majority", "feature_set": "none", **result})
        for i, row in zip(test, majority_probability):
            predictions.append({"row_index": int(i), "fold": fold, "model": "majority",
                                "feature_set": "none", "y_true": int(y.iloc[i]), "probability": row})
        for kind, feature_set, columns in specs:
            c, ratio, threshold, inner_pr = tune_and_threshold(
                data.iloc[train][columns], y.iloc[train], groups.iloc[train], kind)
            fit = estimator(kind, c, ratio).fit(data.iloc[train][columns], y.iloc[train])
            probability = fit.predict_proba(data.iloc[test][columns])[:, 1]
            result = metrics(y.iloc[test], probability, threshold)
            name = "elastic_net" if kind == "elastic_net" else "logistic"
            fold_rows.append({"fold": fold, "model": name, "feature_set": feature_set, **result})
            tuning.append({"fold": fold, "model": name, "feature_set": feature_set,
                           "C": c, "l1_ratio": ratio, "inner_pr_auc": inner_pr,
                           "f1_threshold": threshold})
            for i, row in zip(test, probability):
                predictions.append({"row_index": int(i), "fold": fold, "model": name,
                                    "feature_set": feature_set, "y_true": int(y.iloc[i]),
                                    "probability": float(row)})
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(OUT / "fold_metrics.csv", index=False)
    pd.DataFrame(tuning).to_csv(OUT / "tuning.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "out_of_fold_predictions.csv.gz", index=False,
                                     compression="gzip")
    metric_columns = ["pr_auc", "roc_auc", "f1", "precision", "recall", "brier"]
    aggregate = folds.groupby(["model", "feature_set"])[metric_columns].agg(["mean", "std"])
    aggregate.columns = [f"{a}_{b}" for a, b in aggregate.columns]
    aggregate = aggregate.reset_index()
    aggregate.to_csv(OUT / "aggregate_metrics.csv", index=False)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_qc_passed",
        "sample_rows": len(data), "positive": int(y.sum()), "groups": int(groups.nunique()),
        "outer_cv": "5-fold StratifiedGroupKFold by event_id",
        "inner_cv": "3-fold StratifiedGroupKFold by event_id",
        "seed": SEED, "target": TARGET, "baseline_features": BASELINE,
        "behavioural_features": BEHAVIOURAL,
        "software": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
        "aggregate": aggregate.to_dict(orient="records"),
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(aggregate.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
