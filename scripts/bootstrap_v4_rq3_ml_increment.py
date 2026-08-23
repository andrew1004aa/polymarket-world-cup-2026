#!/usr/bin/env python3
"""Paired match-cluster bootstrap for expanded versus baseline RQ3 models."""
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             f1_score, roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "model_samples/v4/rq3_ml/rq3_match_pre_full_reversal_30m.csv.gz"
OUT = ROOT / "ml_results/v4/rq3_match_pre/bootstrap"
SOURCES = [
    ROOT / "ml_results/v4/rq3_match_pre",
    ROOT / "ml_results/v4/rq3_match_pre/trees",
]
SEED = 20260814
REPLICATES = 4999
MODELS = ["logistic", "elastic_net", "hist_gradient_boosting", "random_forest"]


def metric_values(y, probability, prediction):
    return {
        "pr_auc": average_precision_score(y, probability),
        "roc_auc": roc_auc_score(y, probability),
        "f1": f1_score(y, prediction, zero_division=0),
        "brier": brier_score_loss(y, probability),
    }


def load_predictions():
    frames = []
    for source in SOURCES:
        predictions = pd.read_csv(source / "out_of_fold_predictions.csv.gz")
        thresholds = pd.read_csv(source / "fold_metrics.csv")[
            ["fold", "model", "feature_set", "threshold"]
        ]
        predictions = predictions.merge(thresholds, on=["fold", "model", "feature_set"],
                                        validate="many_to_one")
        frames.append(predictions)
    data = pd.concat(frames, ignore_index=True)
    sample = pd.read_csv(SAMPLE, usecols=["event_id"])
    sample.index.name = "row_index"
    sample = sample.reset_index()
    data = data.merge(sample, on="row_index", validate="many_to_one")
    data["prediction"] = (data.probability >= data.threshold).astype("int8")
    return data


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_predictions()
    rng = np.random.default_rng(SEED)
    estimates, draws = [], []
    for model_name in MODELS:
        model = data[data.model.eq(model_name)]
        wide = model.pivot(index="row_index", columns="feature_set",
                           values=["y_true", "probability", "prediction", "event_id"])
        if len(wide) != 5843 or not np.array_equal(
                wide[("y_true", "baseline")], wide[("y_true", "expanded")]):
            raise SystemExit(f"Prediction pairing QC failed for {model_name}")
        y = wide[("y_true", "baseline")].to_numpy(dtype=int)
        base_probability = wide[("probability", "baseline")].to_numpy(float)
        full_probability = wide[("probability", "expanded")].to_numpy(float)
        base_prediction = wide[("prediction", "baseline")].to_numpy(dtype=int)
        full_prediction = wide[("prediction", "expanded")].to_numpy(dtype=int)
        event = wide[("event_id", "baseline")].to_numpy()
        unique_events = np.unique(event)
        event_indices = {value: np.flatnonzero(event == value) for value in unique_events}
        baseline = metric_values(y, base_probability, base_prediction)
        expanded = metric_values(y, full_probability, full_prediction)
        observed = {name: (baseline[name] - expanded[name] if name == "brier"
                           else expanded[name] - baseline[name]) for name in baseline}
        bootstrap = {name: np.empty(REPLICATES) for name in observed}
        for replicate in range(REPLICATES):
            sampled_events = rng.choice(unique_events, size=len(unique_events), replace=True)
            indices = np.concatenate([event_indices[value] for value in sampled_events])
            b = metric_values(y[indices], base_probability[indices], base_prediction[indices])
            e = metric_values(y[indices], full_probability[indices], full_prediction[indices])
            for name in observed:
                bootstrap[name][replicate] = (b[name] - e[name] if name == "brier"
                                               else e[name] - b[name])
        for name, value in observed.items():
            values = bootstrap[name]
            row = {
                "model": model_name, "metric": name,
                "baseline": baseline[name], "expanded": expanded[name],
                "increment": value,
                "ci_2_5": float(np.quantile(values, 0.025)),
                "ci_97_5": float(np.quantile(values, 0.975)),
                "bootstrap_se": float(values.std(ddof=1)),
                "one_sided_p_increment_le_zero": float((1 + np.sum(values <= 0)) /
                                                         (REPLICATES + 1)),
            }
            estimates.append(row)
            draws.extend({"model": model_name, "metric": name,
                          "replicate": i + 1, "increment": draw}
                         for i, draw in enumerate(values))
        print(f"Completed {model_name}", flush=True)
    estimates = pd.DataFrame(estimates)
    estimates.to_csv(OUT / "confidence_intervals.csv", index=False)
    pd.DataFrame(draws).to_csv(OUT / "bootstrap_draws.csv.gz", index=False, compression="gzip")
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_qc_passed", "replicates": REPLICATES, "seed": SEED,
        "resampling_unit": "FIFA match event_id; paired baseline and expanded predictions",
        "interval": "2.5th and 97.5th percentile cluster-bootstrap interval",
        "metric_direction": "positive increments favour expanded; Brier is baseline minus expanded",
        "software": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
        "results": estimates.to_dict(orient="records"),
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(estimates.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
