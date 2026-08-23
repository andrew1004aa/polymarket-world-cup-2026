#!/usr/bin/env python3
"""Chronological holdout robustness test for RQ3 H3c."""
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "model_samples/v4/rq3_ml/rq3_match_pre_full_reversal_30m.csv.gz"
MAPPING = ROOT / "regression_ready/tables/event_market_mapping.csv"
OUT = ROOT / "ml_results/v4/rq3_match_pre/chronological"
SEED = 20260813
TARGET = "full_reversal_30m"
BASELINE = ["baseline_yes_price", "initial_directional_move_5m"]
INCREMENTAL = [
    "log_initiating_trade_value", "initiating_minute_top1_wallet_share",
    "initiating_minute_top3_wallet_share", "initiating_minute_wallet_hhi",
    "initiating_minute_absolute_flow_imbalance", "follower_imbalance_5m",
    "follower_signed_log_net_5m",
]
LOGISTIC_GRID = [0.01, 0.1, 1.0, 10.0]
RF_GRID = [
    {"max_depth": 4, "min_samples_leaf": 20, "max_features": "sqrt"},
    {"max_depth": 8, "min_samples_leaf": 20, "max_features": "sqrt"},
    {"max_depth": None, "min_samples_leaf": 20, "max_features": "sqrt"},
    {"max_depth": 8, "min_samples_leaf": 50, "max_features": 1.0},
]


def make_model(kind, parameters):
    if kind == "logistic":
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=parameters["C"], solver="lbfgs",
                                         penalty="l2", max_iter=5000,
                                         random_state=SEED)),
        ])
    return RandomForestClassifier(
        n_estimators=500, n_jobs=2, random_state=SEED,
        class_weight="balanced_subsample", **parameters)


def candidate_grid(kind):
    return [{"C": value} for value in LOGISTIC_GRID] if kind == "logistic" else RF_GRID


def best_threshold(y, probability):
    candidates = np.unique(np.quantile(probability, np.linspace(0.02, 0.98, 97)))
    values = [f1_score(y, probability >= cutoff, zero_division=0) for cutoff in candidates]
    return float(candidates[int(np.argmax(values))])


def tune(x, y, groups, kind):
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED + 1)
    best = None
    for order, parameters in enumerate(candidate_grid(kind)):
        oof = np.full(len(y), np.nan)
        for fit_index, valid_index in splitter.split(x, y, groups):
            model = make_model(kind, parameters).fit(x.iloc[fit_index], y.iloc[fit_index])
            oof[valid_index] = model.predict_proba(x.iloc[valid_index])[:, 1]
        score = float(average_precision_score(y, oof))
        if best is None or score > best[0]:
            best = (score, -order, parameters, oof)
    score, _, parameters, oof = best
    return parameters, best_threshold(y, oof), score


def evaluate(y, probability, cutoff):
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
    mapping = pd.read_csv(MAPPING, usecols=["event_id", "actual_kickoff_utc"]).drop_duplicates()
    mapping["actual_kickoff_utc"] = pd.to_datetime(mapping.actual_kickoff_utc, utc=True)
    events = mapping.sort_values(["actual_kickoff_utc", "event_id"]).reset_index(drop=True)
    number_train = int(np.floor(len(events) * 0.75))
    train_events = set(pd.to_numeric(events.iloc[:number_train].event_id).astype(int))
    test_events = set(pd.to_numeric(events.iloc[number_train:].event_id).astype(int))
    event_number = pd.to_numeric(data.event_id, errors="raise").astype(int)
    train = data[event_number.isin(train_events)].copy()
    test = data[event_number.isin(test_events)].copy()
    if train_events.intersection(test_events):
        raise SystemExit("Event leakage between chronological train and test")
    if not len(train) or not len(test) or len(train) + len(test) != len(data):
        raise SystemExit("Chronological split failed row-coverage QC")
    y_train, y_test = train[TARGET].astype(int), test[TARGET].astype(int)
    groups_train = train.event_id.astype(str)
    results, tuning, predictions = [], [], []
    for kind in ("logistic", "random_forest"):
        for feature_set, columns in (("baseline", BASELINE),
                                     ("expanded", BASELINE + INCREMENTAL)):
            print(f"Tuning {kind} {feature_set}", flush=True)
            parameters, cutoff, inner_pr = tune(train[columns], y_train, groups_train, kind)
            model = make_model(kind, parameters).fit(train[columns], y_train)
            probability = model.predict_proba(test[columns])[:, 1]
            results.append({"model": kind, "feature_set": feature_set,
                            **evaluate(y_test, probability, cutoff)})
            tuning.append({"model": kind, "feature_set": feature_set,
                           "parameters": json.dumps(parameters, sort_keys=True),
                           "inner_pr_auc": inner_pr, "f1_threshold": cutoff})
            for index, value in zip(test.index, probability):
                predictions.append({"row_index": int(index), "model": kind,
                                    "feature_set": feature_set, "event_id": data.loc[index, "event_id"],
                                    "y_true": int(data.loc[index, TARGET]),
                                    "probability": float(value)})
    result = pd.DataFrame(results)
    result.to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(tuning).to_csv(OUT / "tuning.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "predictions.csv.gz", index=False, compression="gzip")
    split = {
        "train_event_count": len(train_events), "test_event_count": len(test_events),
        "train_rows": len(train), "test_rows": len(test),
        "train_positive": int(y_train.sum()), "test_positive": int(y_test.sum()),
        "train_positive_share": float(y_train.mean()), "test_positive_share": float(y_test.mean()),
        "train_first_kickoff": events.iloc[0].actual_kickoff_utc.isoformat(),
        "train_last_kickoff": events.iloc[number_train - 1].actual_kickoff_utc.isoformat(),
        "test_first_kickoff": events.iloc[number_train].actual_kickoff_utc.isoformat(),
        "test_last_kickoff": events.iloc[-1].actual_kickoff_utc.isoformat(),
        "event_overlap": len(train_events.intersection(test_events)),
    }
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_qc_passed", "split": split, "seed": SEED,
        "selection": "earliest 75% of matches train; latest 25% holdout",
        "tuning": "3-fold StratifiedGroupKFold by event_id within training period only",
        "software": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
        "metrics": result.to_dict(orient="records"),
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(split, indent=2), flush=True)
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
