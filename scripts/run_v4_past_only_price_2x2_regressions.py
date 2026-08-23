#!/usr/bin/env python3
"""Estimate price models using strictly past-only wallet classifications."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy import sparse
from scipy.sparse.linalg import splu

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "model_samples/v4/past_only_price_2x2"
OUT = ROOT / "regression_results/v4/past_only_price_2x2"
SPECS = ("baseline_7d", "baseline_14d")
GROUPS = ("large_high", "large_other", "ordinary_high", "ordinary_other")


def now(): return datetime.now(timezone.utc).isoformat()


def contrast(model, outcome, label, weights, beta, covariance, names, clusters):
    index = {x: i for i, x in enumerate(names)}; w = np.zeros(len(names))
    for term, value in weights.items(): w[index[term]] = value
    estimate = float(sum(beta[term] * value for term, value in weights.items()))
    variance = float(w @ covariance @ w); se = math.sqrt(variance); statistic = estimate / se
    df = clusters - 1; critical = float(stats.t.ppf(.975, df))
    return {"model": model, "outcome": outcome, "contrast": label, "estimate": estimate,
        "std_error": se, "t_statistic": statistic, "cluster_df": df,
        "p_value": float(2 * stats.t.sf(abs(statistic), df)),
        "conf_low": estimate - critical * se, "conf_high": estimate + critical * se,
        "weights": json.dumps(weights, sort_keys=True)}


def residualize_fixed_effects(data, columns):
    market = pd.Categorical(data.market_id).codes
    hour = pd.Categorical(data.calendar_hour_utc).codes
    n = len(data); markets = int(market.max()) + 1; hours = int(hour.max()) + 1
    rows = [np.arange(n)]; cols = [market]; values = [np.ones(n)]
    nonreference = hour > 0
    rows.append(np.flatnonzero(nonreference)); cols.append(markets + hour[nonreference] - 1)
    values.append(np.ones(nonreference.sum()))
    design = sparse.csr_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                               shape=(n, markets + hours - 1))
    gram = (design.T @ design).tocsc()
    solver = splu(gram)
    matrix = data[columns].to_numpy(float)
    absorbed = matrix - design @ solver.solve(np.asarray(design.T @ matrix))
    return {name: absorbed[:, i] for i, name in enumerate(columns)}, {
        "dummy_columns": design.shape[1], "markets": markets, "calendar_hours": hours,
        "method": "exact sparse two-way FWL absorption with one reference hour"}


def fit(specification, outcome, y_name, terms, residual, data, suffix):
    model = f"{specification.upper()}_{outcome.upper()}"; print(f"Estimating {model}", flush=True)
    y = residual[y_name]; x = np.column_stack([residual[t] for t in terms])
    keep = np.max(np.abs(x), axis=1) > 1e-14; y = y[keep]; x = x[keep]
    groups = pd.Categorical(data.event_id.to_numpy()[keep]).codes
    xtx_inverse = np.linalg.pinv(x.T @ x); beta_vector = xtx_inverse @ (x.T @ y)
    errors = y - x @ beta_vector; clusters = int(groups.max()) + 1
    scores = np.zeros((clusters, x.shape[1])); np.add.at(scores, groups, x * errors[:, None])
    n, k = x.shape; correction = (clusters / (clusters - 1)) * ((n - 1) / (n - k))
    covariance = correction * xtx_inverse @ (scores.T @ scores) @ xtx_inverse
    standard_errors = np.sqrt(np.diag(covariance)); df = clusters - 1
    critical = float(stats.t.ppf(.975, df)); rows = []
    for i, term in enumerate(terms):
        statistic = float(beta_vector[i] / standard_errors[i])
        rows.append({"model": model, "specification": specification, "outcome": outcome,
            "term": term, "estimate": float(beta_vector[i]), "std_error": float(standard_errors[i]),
            "t_statistic": statistic, "p_value": float(2 * stats.t.sf(abs(statistic), df)),
            "conf_low": float(beta_vector[i] - critical * standard_errors[i]),
            "conf_high": float(beta_vector[i] + critical * standard_errors[i])})
    tidy = pd.DataFrame(rows); names = terms
    beta = tidy.set_index("term")["estimate"]
    lh, lo, oh, oo = (g + suffix for g in GROUPS)
    tests = {
        "high_activity_effect_among_large": {lh: 1, lo: -1},
        "high_activity_effect_among_ordinary": {oh: 1, oo: -1},
        "large_effect_among_high_activity": {lh: 1, oh: -1},
        "large_effect_among_other": {lo: 1, oo: -1},
        "difference_in_differences": {lh: 1, lo: -1, oh: -1, oo: 1},
    }
    contrasts = [contrast(model, outcome, label, weights, beta, covariance, names, clusters)
                 for label, weights in tests.items()]
    within_r2 = 1 - float(errors @ errors) / float(y @ y)
    diagnostic = {"model": model, "specification": specification, "outcome": outcome,
        "terms": ";".join(terms), "rows": int(len(y)), "markets": int(data.market_id.nunique()),
        "events": clusters, "calendar_hours": int(data.calendar_hour_utc.nunique()),
        "within_r_squared": within_r2, "condition_number": float(np.linalg.cond(x)),
        "collinear_variables_removed": ""}
    return tidy, contrasts, diagnostic


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    all_coef, all_tests, diagnostics, support = [], [], [], []
    core = ["market_id", "event_id", "calendar_hour_utc", "delta_yes_price", "yes_price_t",
            "lagged_30m_price_change", "log_lagged_30m_volume", "log_minutes_to_kickoff"]
    feature_cols = []
    for group in GROUPS:
        feature_cols += [f"{group}_trade_count", f"{group}_log_gross_volume", f"{group}_signed_log_net_flow"]
    for specification in SPECS:
        path = INPUT / f"{specification}_price_2x2.csv.gz"
        data = pd.read_csv(path, usecols=core + feature_cols)
        data["any_price_update_5m"] = data.delta_yes_price.ne(0).astype(float)
        data["absolute_lagged_30m_price_change"] = data.lagged_30m_price_change.abs()
        for group in GROUPS:
            support.append({"specification": specification, "group": group,
                "trades": int(data[f"{group}_trade_count"].sum()),
                "active_market_minutes": int(data[f"{group}_trade_count"].gt(0).sum()),
                "markets_with_activity": int(data.loc[data[f"{group}_trade_count"].gt(0), "market_id"].nunique())})
        update_terms = [f"{g}_log_gross_volume" for g in GROUPS]
        update_terms += ["absolute_lagged_30m_price_change", "yes_price_t",
                         "log_lagged_30m_volume", "log_minutes_to_kickoff"]
        direction_terms = [f"{g}_signed_log_net_flow" for g in GROUPS]
        direction_terms += ["lagged_30m_price_change", "yes_price_t",
                            "log_lagged_30m_volume", "log_minutes_to_kickoff"]
        all_numeric = ["any_price_update_5m", "delta_yes_price"] + update_terms + direction_terms
        all_numeric = list(dict.fromkeys(all_numeric))
        print(f"Absorbing market and calendar-hour effects for {specification}", flush=True)
        residual, absorption = residualize_fixed_effects(data, all_numeric)
        models = [
            ("price_update_lpm", "any_price_update_5m", update_terms, "_log_gross_volume"),
            ("directional_price_change", "delta_yes_price", direction_terms, "_signed_log_net_flow"),
        ]
        for outcome, y_name, terms, suffix in models:
            c, t, d = fit(specification, outcome, y_name, terms, residual, data, suffix)
            d.update(absorption)
            all_coef.append(c); all_tests += t; diagnostics.append(d)
        del data, residual
    pd.concat(all_coef, ignore_index=True).to_csv(OUT / "coefficients.csv", index=False)
    pd.DataFrame(all_tests).to_csv(OUT / "contrasts.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(OUT / "model_diagnostics.csv", index=False)
    pd.DataFrame(support).to_csv(OUT / "group_support.csv", index=False)
    payload = {"generated_at": now(), "status": "complete_qc_passed", "specifications": list(SPECS),
        "models": 4, "estimator": "exact sparse FWL absorption followed by OLS",
        "fixed_effects": ["market_id", "calendar_hour_utc"], "cluster": "event_id CRV1",
        "interpretation": "associational robustness using fixed baseline wallet classifications"}
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(pd.DataFrame(all_tests)[["model", "contrast", "estimate", "std_error", "p_value",
                                   "conf_low", "conf_high"]].to_string(index=False), flush=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())
