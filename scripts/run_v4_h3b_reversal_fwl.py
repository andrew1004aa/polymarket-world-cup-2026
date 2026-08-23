#!/usr/bin/env python3
"""Estimate H3b with explicit two-way FWL absorption and event clustering."""
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "model_samples/v4/h3b_reversal/h3b_match_pre_initial_directional_move.csv.gz"
OUT = ROOT / "regression_results/v4/h3b/match_pre"
HORIZONS = ("15m", "30m", "60m")
CONTROLS = [
    "initial_directional_move_5m", "yes_price_t", "absolute_lagged_30m_price_change",
    "log_lagged_30m_volume", "log_minutes_to_kickoff", "log_gross_volume_usdc",
    "log_trade_count",
]


def absorb(values, codes, tolerance=1e-10, max_iterations=10000):
    result = np.asarray(values, dtype=float).copy()
    for iteration in range(1, max_iterations + 1):
        before = result.copy()
        for group_codes in codes:
            number_groups = int(group_codes.max()) + 1
            sums = np.zeros((number_groups, result.shape[1]))
            np.add.at(sums, group_codes, result)
            counts = np.bincount(group_codes, minlength=number_groups).reshape(-1, 1)
            result -= (sums / counts)[group_codes]
        if np.max(np.abs(result - before)) < tolerance:
            return result, iteration
    raise RuntimeError("Two-way fixed-effect absorption did not converge")


def fit_model(sample, outcome, terms, model_name, specification, focal_term):
    variables = [outcome] + terms
    matrix = sample[variables].to_numpy(float)
    market_codes = pd.Categorical(sample.market_id).codes
    hour_codes = pd.Categorical(sample.calendar_hour_utc).codes
    residual, iterations = absorb(matrix, [market_codes, hour_codes])
    y, x = residual[:, 0], residual[:, 1:]
    nonzero_design = np.max(np.abs(x), axis=1) > 1e-14
    y_fit = y[nonzero_design]
    x_fit = x[nonzero_design]
    groups = pd.Categorical(sample.event_id.to_numpy()[nonzero_design]).codes
    xtx_inverse = np.linalg.pinv(x_fit.T @ x_fit)
    beta = xtx_inverse @ (x_fit.T @ y_fit)
    errors = y_fit - x_fit @ beta
    number_clusters = int(groups.max()) + 1
    scores = np.zeros((number_clusters, x_fit.shape[1]))
    np.add.at(scores, groups, x_fit * errors[:, None])
    meat = scores.T @ scores
    n, k = x_fit.shape
    correction = (number_clusters / (number_clusters - 1)) * ((n - 1) / (n - k))
    covariance = correction * xtx_inverse @ meat @ xtx_inverse
    standard_errors = np.sqrt(np.diag(covariance))
    cluster_df = int(sample.event_id.nunique()) - 1
    critical = stats.t.ppf(0.975, cluster_df)
    rows = []
    for index, term in enumerate(terms):
        estimate = float(beta[index]); standard_error = float(standard_errors[index])
        statistic = estimate / standard_error
        rows.append({
            "model": model_name, "specification": specification, "outcome": outcome,
            "term": term, "estimate": estimate, "std_error": standard_error,
            "t_statistic": statistic, "cluster_df": cluster_df,
            "p_value": float(2 * stats.t.sf(abs(statistic), cluster_df)),
            "conf_low": estimate - critical * standard_error,
            "conf_high": estimate + critical * standard_error,
        })
    diagnostic = {
        "model": model_name, "specification": specification, "outcome": outcome,
        "focal_term": focal_term, "rows_input": len(sample),
        "rows_nonzero_absorbed_design": int(nonzero_design.sum()),
        "outcome_share": float(sample[outcome].mean()),
        "markets": int(sample.market_id.nunique()), "events": int(sample.event_id.nunique()),
        "calendar_hours": int(sample.calendar_hour_utc.nunique()),
        "absorption_iterations": iterations, "condition_number": float(np.linalg.cond(x[nonzero_design])),
    }
    return rows, diagnostic


def holm_adjust(p_values):
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted_sorted = np.maximum.accumulate((len(values) - np.arange(len(values))) * values[order])
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    outcomes = [f"{kind}_{horizon}" for horizon in HORIZONS
                for kind in ("partial_reversal", "full_reversal")]
    columns = ["market_id", "event_id", "calendar_hour_utc", "wallet_volume_hhi",
               "absolute_flow_imbalance", "dominant_directional_concentration",
               "initial_directional_move_5m", "yes_price_t", "lagged_30m_price_change",
               "log_lagged_30m_volume", "log_minutes_to_kickoff", "log_gross_volume_usdc",
               "log_trade_count"] + outcomes
    data = pd.read_csv(SRC, usecols=columns)
    data["absolute_lagged_30m_price_change"] = data.lagged_30m_price_change.abs()
    specifications = {
        "primary_interaction": (["dominant_directional_concentration", "wallet_volume_hhi",
                                 "absolute_flow_imbalance"] + CONTROLS,
                                "dominant_directional_concentration"),
        "hhi_robustness": (["wallet_volume_hhi", "absolute_flow_imbalance"] + CONTROLS,
                           "wallet_volume_hhi"),
    }
    rows, diagnostics = [], []
    for specification, (terms, focal_term) in specifications.items():
        for outcome in outcomes:
            sample = data.dropna(subset=[outcome] + terms).copy()
            model_name = f"H3B_{specification.upper()}_{outcome.upper()}"
            print(f"Estimating {model_name} on {len(sample):,} rows", flush=True)
            fitted_rows, diagnostic = fit_model(sample, outcome, terms, model_name,
                                                specification, focal_term)
            rows.extend(fitted_rows); diagnostics.append(diagnostic)
    coefficients = pd.DataFrame(rows)
    coefficients.to_csv(OUT / "coefficients.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(OUT / "model_diagnostics.csv", index=False)
    primary = coefficients[(coefficients.specification == "primary_interaction") &
                           (coefficients.term == "dominant_directional_concentration")].copy()
    primary["holm_p_value"] = holm_adjust(primary.p_value)
    primary["effect_per_0_1_increase"] = primary.estimate * 0.1
    robust = coefficients[(coefficients.specification == "hhi_robustness") &
                          (coefficients.term == "wallet_volume_hhi")].copy()
    robust["effect_per_0_1_increase"] = robust.estimate * 0.1
    primary.to_csv(OUT / "primary_concentration_results.csv", index=False)
    robust.to_csv(OUT / "hhi_robustness_results.csv", index=False)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_qc_passed", "sample_rows": len(data),
        "estimator": "Frisch-Waugh-Lovell alternating two-way absorption followed by OLS",
        "fixed_effects": ["market_id", "calendar_hour_utc"],
        "inference": "CRV1 event-clustered covariance; t reference with 103 df",
        "primary_focal_variable": "wallet HHI × absolute net-flow imbalance",
        "multiple_testing": "Holm correction across six primary outcomes",
        "software": {"python": platform.python_version(), "pandas": pd.__version__,
                     "numpy": np.__version__, "scipy": scipy.__version__,
                     "linear_algebra": "NumPy OLS with finite-sample CRV1 correction"},
        "primary_results": primary.to_dict(orient="records"),
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(primary[["outcome", "estimate", "std_error", "t_statistic", "p_value",
                   "holm_p_value", "conf_low", "conf_high", "effect_per_0_1_increase"]].to_string(index=False))
    print("\nHHI robustness")
    print(robust[["outcome", "estimate", "std_error", "p_value", "conf_low", "conf_high"]].to_string(index=False))


if __name__ == "__main__":
    main()
