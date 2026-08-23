#!/usr/bin/env python3
"""Estimate 2x2 Large x high-activity-wallet reversal models on frozen H3b sample."""

from __future__ import annotations

import hashlib
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
REVERSAL = ROOT / "model_samples/v4/h3b_reversal/h3b_match_pre_initial_directional_move.csv.gz"
PANEL = ROOT / "model_samples/v4/large_wallet_2x2/large_wallet_2x2_primary_prematch.csv.gz"
SAMPLE_OUT = ROOT / "model_samples/v4/large_wallet_2x2/reversal_2x2_sample.csv.gz"
OUT = ROOT / "regression_results/v4/large_wallet_2x2_reversal"
GROUPS = ("large_high_activity", "large_other", "ordinary_high_activity", "ordinary_other")
FILTER_TO_PANEL = False
HORIZONS = ("15m", "30m", "60m")
CONTROLS = ["initial_directional_move_5m", "yes_price_t",
            "absolute_lagged_30m_price_change", "log_lagged_30m_volume",
            "log_minutes_to_kickoff", "log_gross_volume_usdc", "log_trade_count"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def absorb(values: np.ndarray, codes: list[np.ndarray], tolerance=1e-10, max_iterations=10000):
    result = np.asarray(values, dtype=float).copy()
    for iteration in range(1, max_iterations + 1):
        before = result.copy()
        for group_codes in codes:
            groups = int(group_codes.max()) + 1
            sums = np.zeros((groups, result.shape[1]))
            np.add.at(sums, group_codes, result)
            counts = np.bincount(group_codes, minlength=groups).reshape(-1, 1)
            result -= (sums / counts)[group_codes]
        if np.max(np.abs(result - before)) < tolerance:
            return result, iteration
    raise RuntimeError("Two-way fixed-effect absorption did not converge")


def contrast(label: str, weights: dict[str, float], beta: np.ndarray, covariance: np.ndarray,
             terms: list[str], clusters: int, model: str, outcome: str) -> dict:
    index = {term: i for i, term in enumerate(terms)}
    w = np.zeros(len(terms))
    for term, value in weights.items():
        w[index[term]] = value
    estimate = float(w @ beta)
    variance = float(w @ covariance @ w)
    if not math.isfinite(variance) or variance <= 0:
        raise RuntimeError(f"Invalid variance for {model}/{label}: {variance}")
    se = math.sqrt(variance); statistic = estimate / se; df = clusters - 1
    critical = float(stats.t.ppf(.975, df))
    return {"model": model, "outcome": outcome, "contrast": label,
            "estimate": estimate, "std_error": se, "t_statistic": statistic,
            "cluster_df": df, "p_value": float(2 * stats.t.sf(abs(statistic), df)),
            "conf_low": estimate - critical * se, "conf_high": estimate + critical * se,
            "weights": json.dumps(weights, sort_keys=True)}


def fit(sample: pd.DataFrame, outcome: str):
    exposures = [f"aligned_flow_{g}" for g in GROUPS]
    terms = exposures + CONTROLS
    matrix = sample[[outcome] + terms].to_numpy(float)
    codes = [pd.Categorical(sample.market_id).codes, pd.Categorical(sample.calendar_hour_utc).codes]
    residual, iterations = absorb(matrix, codes)
    y, x = residual[:, 0], residual[:, 1:]
    keep = np.max(np.abs(x), axis=1) > 1e-14
    y, x = y[keep], x[keep]
    groups = pd.Categorical(sample.event_id.to_numpy()[keep]).codes
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ (x.T @ y)
    errors = y - x @ beta
    clusters = int(groups.max()) + 1
    scores = np.zeros((clusters, x.shape[1]))
    np.add.at(scores, groups, x * errors[:, None])
    n, k = x.shape
    correction = (clusters / (clusters - 1)) * ((n - 1) / (n - k))
    covariance = correction * xtx_inv @ (scores.T @ scores) @ xtx_inv
    se = np.sqrt(np.diag(covariance)); df = clusters - 1
    critical = float(stats.t.ppf(.975, df))
    model = f"REVERSAL_2X2_{outcome.upper()}"
    coefficients = []
    for i, term in enumerate(terms):
        statistic = float(beta[i] / se[i])
        coefficients.append({"model": model, "outcome": outcome, "term": term,
            "estimate": float(beta[i]), "std_error": float(se[i]), "t_statistic": statistic,
            "cluster_df": df, "p_value": float(2 * stats.t.sf(abs(statistic), df)),
            "conf_low": float(beta[i] - critical * se[i]),
            "conf_high": float(beta[i] + critical * se[i])})
    lh, lo, oh, oo = exposures
    specifications = {
        "high_activity_effect_among_large": {lh: 1, lo: -1},
        "high_activity_effect_among_ordinary": {oh: 1, oo: -1},
        "large_effect_among_high_activity": {lh: 1, oh: -1},
        "large_effect_among_other": {lo: 1, oo: -1},
        "difference_in_differences": {lh: 1, lo: -1, oh: -1, oo: 1},
        "average_high_activity_effect": {lh: .5, lo: -.5, oh: .5, oo: -.5},
        "average_large_effect": {lh: .5, lo: .5, oh: -.5, oo: -.5},
    }
    tests = [contrast(label, weights, beta, covariance, terms, clusters, model, outcome)
             for label, weights in specifications.items()]
    diagnostic = {"model": model, "outcome": outcome, "rows": int(len(sample)),
                  "rows_nonzero_absorbed_design": int(keep.sum()),
                  "outcome_rate": float(sample[outcome].mean()),
                  "markets": int(sample.market_id.nunique()),
                  "events": int(sample.event_id.nunique()),
                  "calendar_hours": int(sample.calendar_hour_utc.nunique()),
                  "absorption_iterations": iterations,
                  "condition_number": float(np.linalg.cond(x))}
    return coefficients, tests, diagnostic


def holm(values: pd.Series) -> np.ndarray:
    p = values.to_numpy(float); order = np.argsort(p)
    adjusted_sorted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    adjusted = np.empty_like(p); adjusted[order] = np.minimum(adjusted_sorted, 1)
    return adjusted


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True); SAMPLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    log = {"version": "v4_large_wallet_2x2_reversal_20260815", "status": "running",
           "started_at": now(), "reversal_sha256": sha256(REVERSAL), "panel_sha256": sha256(PANEL)}
    (OUT / "run_log.json").write_text(json.dumps(log, indent=2) + "\n")
    print("Loading frozen reversal and 2x2 panels", flush=True)
    reversal = pd.read_csv(REVERSAL)
    panel_cols = ["model_sample_record_id"] + [f"{g}_signed_log_net_flow" for g in GROUPS]
    panel = pd.read_csv(PANEL, usecols=panel_cols)
    data = reversal.merge(panel, on="model_sample_record_id", how="inner", validate="one_to_one")
    if ((not FILTER_TO_PANEL and len(data) != len(reversal)) or
            data.model_sample_record_id.duplicated().any()):
        raise RuntimeError("2x2 merge failed one-to-one reversal-sample reconciliation")
    data["absolute_lagged_30m_price_change"] = data.lagged_30m_price_change.abs()
    for g in GROUPS:
        data[f"aligned_flow_{g}"] = data.trade_direction * data[f"{g}_signed_log_net_flow"]
    keep = ["model_sample_record_id", "market_id", "event_id", "calendar_hour_utc",
            "trade_direction"] + [f"aligned_flow_{g}" for g in GROUPS] + CONTROLS
    outcomes = [f"{kind}_reversal_{h}" for h in HORIZONS for kind in ("partial", "full")]
    keep += outcomes
    data[keep].to_csv(SAMPLE_OUT, index=False, compression="gzip")
    coefficients, tests, diagnostics = [], [], []
    for outcome in outcomes:
        sample = data.dropna(subset=[outcome] + [f"aligned_flow_{g}" for g in GROUPS] + CONTROLS)
        print(f"Estimating {outcome} on {len(sample):,} rows", flush=True)
        c, t, d = fit(sample, outcome); coefficients += c; tests += t; diagnostics.append(d)
    coef = pd.DataFrame(coefficients); con = pd.DataFrame(tests); diag = pd.DataFrame(diagnostics)
    con["holm_p_value_within_contrast_family"] = con.groupby("contrast")["p_value"].transform(holm)
    con["is_primary_30m_full_reversal"] = con.outcome.eq("full_reversal_30m")
    coef.to_csv(OUT / "coefficients.csv", index=False)
    con.to_csv(OUT / "contrasts.csv", index=False)
    diag.to_csv(OUT / "model_diagnostics.csv", index=False)
    payload = {"generated_at": now(), "status": "complete_qc_passed",
        "definition": "conditional on non-zero net flow followed by a same-direction five-minute price move",
        "primary_outcome": "full_reversal_30m", "secondary_outcomes": [x for x in outcomes if x != "full_reversal_30m"],
        "exposure": "trade_direction multiplied by each group-specific signed-log net flow",
        "sample_rows": int(len(data)), "markets": int(data.market_id.nunique()), "events": int(data.event_id.nunique()),
        "fixed_effects": ["market_id", "calendar_hour_utc"], "cluster": "event_id",
        "multiplicity": "Holm within each contrast family across six reversal outcomes",
        "sample_output": str(SAMPLE_OUT.relative_to(ROOT)), "sample_sha256": sha256(SAMPLE_OUT),
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__}}
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    log.update({"status": "complete_qc_passed", "completed_at": now(), "outputs":
                ["coefficients.csv", "contrasts.csv", "model_diagnostics.csv", "summary.json"]})
    (OUT / "run_log.json").write_text(json.dumps(log, indent=2) + "\n")
    print(con[con.is_primary_30m_full_reversal][["contrast", "estimate", "std_error", "p_value",
          "holm_p_value_within_contrast_family", "conf_low", "conf_high"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
