#!/usr/bin/env python3
"""Threshold-free robustness using strictly past-only cumulative wallet volume.

Wallet scale on evaluation day d is log1p(gross USDC before d), standardised
across every wallet with prior history.  Models use the same 15 June--22 July
prematch match-market sample as the binary past-only whale analyses.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.sparse.linalg import splu

ROOT = Path(__file__).resolve().parents[1]
TRADES = ROOT / "regression_ready/trades_by_market"
CHECKPOINTS = ROOT / "model_samples/v4/wallet_sequences/checkpoints"
BASE = ROOT / "regression_inputs/v3/primary_prematch_5m_complete_case.csv.gz"
OUT = ROOT / "regression_results/v4/continuous_cumulative_wallet_volume"
START_HISTORY = date(2026, 6, 1)
START = date(2026, 6, 15)
END = date(2026, 7, 22)


def now():
    return datetime.now(timezone.utc).isoformat()


def days(first, last):
    value = first
    while value <= last:
        yield value
        value += timedelta(days=1)


def read_gzip_rows(path):
    for attempt in range(1, 6):
        try:
            with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
                yield from csv.DictReader(handle)
            return
        except TimeoutError:
            if attempt == 5:
                raise
            time.sleep(attempt)


def residualize(data, columns):
    market = pd.Categorical(data.market_id).codes
    hour = pd.Categorical(data.calendar_hour_utc).codes
    n = len(data); markets = int(market.max()) + 1; hours = int(hour.max()) + 1
    rr = [np.arange(n)]; cc = [market]; vv = [np.ones(n)]
    keep = hour > 0
    rr.append(np.flatnonzero(keep)); cc.append(markets + hour[keep] - 1); vv.append(np.ones(keep.sum()))
    design = sparse.csr_matrix((np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))),
                               shape=(n, markets + hours - 1))
    solver = splu((design.T @ design).tocsc())
    matrix = data[columns].to_numpy(float)
    result = matrix - design @ solver.solve(np.asarray(design.T @ matrix))
    return {column: result[:, index] for index, column in enumerate(columns)}


def fit(data, outcome, terms, model):
    columns = list(dict.fromkeys([outcome] + terms))
    absorbed = residualize(data, columns)
    y = absorbed[outcome]
    x = np.column_stack([absorbed[t] for t in terms])
    keep = np.max(np.abs(x), axis=1) > 1e-14
    y = y[keep]; x = x[keep]
    groups = pd.Categorical(data.event_id.to_numpy()[keep]).codes
    inv = np.linalg.pinv(x.T @ x); beta = inv @ (x.T @ y); error = y - x @ beta
    clusters = int(groups.max()) + 1
    scores = np.zeros((clusters, x.shape[1])); np.add.at(scores, groups, x * error[:, None])
    n, k = x.shape
    covariance = (clusters / (clusters - 1)) * ((n - 1) / (n - k)) * inv @ (scores.T @ scores) @ inv
    se = np.sqrt(np.diag(covariance)); df = clusters - 1; critical = stats.t.ppf(.975, df)
    rows = []
    for i, term in enumerate(terms):
        t = beta[i] / se[i]
        rows.append({"model": model, "outcome": outcome, "term": term, "estimate": beta[i],
                     "std_error": se[i], "t_statistic": t, "cluster_df": df,
                     "p_value": 2 * stats.t.sf(abs(t), df),
                     "conf_low": beta[i] - critical * se[i], "conf_high": beta[i] + critical * se[i]})
    diagnostics = {"model": model, "rows": int(len(y)), "markets": int(data.market_id.nunique()),
                   "events": clusters, "calendar_hours": int(data.calendar_hour_utc.nunique()),
                   "within_r_squared": 1 - float(error @ error) / float(y @ y),
                   "condition_number": float(np.linalg.cond(x))}
    return rows, diagnostics, beta, covariance, terms, clusters


def linear_test(model, outcome, label, weights, beta, covariance, terms, clusters):
    lookup = {term: index for index, term in enumerate(terms)}
    vector = np.zeros(len(terms))
    for term, weight in weights.items(): vector[lookup[term]] = weight
    estimate = float(vector @ beta); variance = float(vector @ covariance @ vector)
    se = math.sqrt(max(variance, 0)); t = estimate / se; df = clusters - 1
    critical = stats.t.ppf(.975, df)
    return {"model": model, "outcome": outcome, "contrast": label, "estimate": estimate,
            "std_error": se, "t_statistic": t, "cluster_df": df,
            "p_value": 2 * stats.t.sf(abs(t), df), "conf_low": estimate - critical * se,
            "conf_high": estimate + critical * se, "weights": json.dumps(weights, sort_keys=True)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    market_files = sorted(TRADES.glob("*.csv.gz")); action_files = sorted(CHECKPOINTS.glob("*.csv.gz"))
    if len(market_files) != 360 or len(action_files) != 312:
        raise RuntimeError(f"Expected 360 trade and 312 action files; found {len(market_files)}, {len(action_files)}")

    daily_volume = defaultdict(lambda: defaultdict(float))
    print("Pass 1/4: all-market wallet-day gross USDC", flush=True)
    for index, path in enumerate(market_files, 1):
        for row in read_gzip_rows(path):
            day = datetime.fromtimestamp(int(row["timestamp"]), timezone.utc).date()
            if START_HISTORY <= day <= END:
                daily_volume[day][row["wallet_address"].lower()] += float(row["trade_value_usdc"])
        if index % 40 == 0: print(f"  {index}/360 markets", flush=True)

    by_day = {day: [] for day in days(START, END)}
    wanted = ["market_id", "event_id", "timestamp", "calendar_hour_utc", "wallet_address",
              "direction", "action_value_usdc", "log_action_value_usdc", "large_action",
              "log_minutes_to_kickoff", "any_next_action_5m", "same_direction_next_action_5m",
              "opposite_action_within_60m", "favorable_opposite_action_within_60m"]
    print("Pass 2/4: organise evaluation actions", flush=True)
    for index, path in enumerate(action_files, 1):
        frame = pd.read_csv(path, usecols=wanted)
        frame["evaluation_day"] = pd.to_datetime(frame.timestamp, unit="s", utc=True).dt.date
        frame = frame[frame.evaluation_day.between(START, END)]
        for day, part in frame.groupby("evaluation_day", sort=False): by_day[day].append(part)
        if index % 40 == 0: print(f"  {index}/312 markets", flush=True)

    expanding = defaultdict(float); scored_actions = []; price_parts = []; score_diagnostics = []
    print("Pass 3/4: past-only scores and daily feature construction", flush=True)
    for current in days(START_HISTORY, END):
        previous = current - timedelta(days=1)
        for wallet, volume in daily_volume.get(previous, {}).items(): expanding[wallet] += volume
        if current < START: continue
        logs = np.fromiter((math.log1p(value) for value in expanding.values()), dtype=float)
        mean = float(logs.mean()); sd = float(logs.std(ddof=0))
        frames = by_day.get(current, [])
        if not frames: continue
        action = pd.concat(frames, ignore_index=True)
        action["wallet_address"] = action.wallet_address.str.lower()
        action["past_cumulative_gross_usdc"] = action.wallet_address.map(expanding)
        if action.past_cumulative_gross_usdc.isna().any():
            raise RuntimeError(f"Missing prior wallet history on {current}")
        action["wallet_scale_z"] = (np.log1p(action.past_cumulative_gross_usdc) - mean) / sd
        action["large_x_wallet_scale_z"] = action.large_action * action.wallet_scale_z
        scored_actions.append(action[[*wanted, "past_cumulative_gross_usdc", "wallet_scale_z",
                                      "large_x_wallet_scale_z"]])
        action["minute_start_timestamp"] = (action.timestamp // 60) * 60
        action["signed_value"] = action.direction * action.action_value_usdc
        action["z_gross_value"] = action.wallet_scale_z * action.action_value_usdc
        action["z_signed_value"] = action.wallet_scale_z * action.signed_value
        for large, label in ((1, "large"), (0, "ordinary")):
            part = action[action.large_action.eq(large)].groupby(
                ["market_id", "minute_start_timestamp"], as_index=False).agg(
                    **{f"{label}_trade_count": ("timestamp", "size"),
                       f"{label}_gross_volume": ("action_value_usdc", "sum"),
                       f"{label}_signed_flow": ("signed_value", "sum"),
                       f"{label}_z_gross": ("z_gross_value", "sum"),
                       f"{label}_z_signed": ("z_signed_value", "sum")})
            price_parts.append(part)
        score_diagnostics.append({"evaluation_day": current.isoformat(), "eligible_wallets": len(expanding),
                                  "log1p_mean": mean, "log1p_population_sd": sd,
                                  "evaluation_actions": len(action), "minimum_prior_trades": 1})
        print(f"  {current}: {len(action):,} actions; {len(expanding):,} eligible wallets", flush=True)

    actions = pd.concat(scored_actions, ignore_index=True)
    actions.to_csv(OUT / "scored_actions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(score_diagnostics).to_csv(OUT / "daily_score_diagnostics.csv", index=False)

    coefficient_rows = []; contrast_rows = []; model_diagnostics = []
    sequence_specs = [
        ("any_next_action_5m", None),
        ("same_direction_next_action_5m", "any_next_action_5m"),
        ("opposite_action_within_60m", None),
        ("favorable_opposite_action_within_60m", "opposite_action_within_60m"),
    ]
    sequence_terms = ["wallet_scale_z", "large_action", "large_x_wallet_scale_z",
                      "log_action_value_usdc", "log_minutes_to_kickoff"]
    for outcome, condition in sequence_specs:
        sample = actions if condition is None else actions[actions[condition].eq(1)]
        model = f"SEQUENCE_{outcome.upper()}"
        c, d, beta, cov, terms, clusters = fit(sample, outcome, sequence_terms, model)
        coefficient_rows += c; model_diagnostics.append(d)
        contrast_rows.append(linear_test(model, outcome, "continuous_wallet_scale_x_large",
                                         {"large_x_wallet_scale_z": 1}, beta, cov, terms, clusters))

    base_cols = ["model_sample_record_id", "market_id", "event_id", "minute_start_timestamp",
                 "calendar_hour_utc", "delta_yes_price", "yes_price_t", "lagged_30m_price_change",
                 "log_lagged_30m_volume", "log_minutes_to_kickoff"]
    base = pd.read_csv(BASE, usecols=base_cols)
    start_ts = int(datetime(2026, 6, 15, tzinfo=timezone.utc).timestamp())
    end_ts = int(datetime(2026, 7, 23, tzinfo=timezone.utc).timestamp())
    base = base[base.minute_start_timestamp.between(start_ts, end_ts - 1)].copy()
    merged_parts = []
    for label in ("large", "ordinary"):
        relevant = [part for part in price_parts if f"{label}_trade_count" in part]
        merged_parts.append(pd.concat(relevant, ignore_index=True).groupby(
            ["market_id", "minute_start_timestamp"], as_index=False).sum())
    panel = base
    for part in merged_parts:
        panel = panel.merge(part, on=["market_id", "minute_start_timestamp"], how="left", validate="one_to_one")
    feature_columns = []
    for label in ("large", "ordinary"):
        raw = [f"{label}_trade_count", f"{label}_gross_volume", f"{label}_signed_flow",
               f"{label}_z_gross", f"{label}_z_signed"]
        panel[raw] = panel[raw].fillna(0)
        panel[f"{label}_log_gross"] = np.log1p(panel[f"{label}_gross_volume"])
        panel[f"{label}_signed_log_flow"] = np.sign(panel[f"{label}_signed_flow"]) * np.log1p(panel[f"{label}_signed_flow"].abs())
        panel[f"{label}_mean_wallet_z"] = np.divide(panel[f"{label}_z_gross"], panel[f"{label}_gross_volume"],
                                                       out=np.zeros(len(panel)), where=panel[f"{label}_gross_volume"].ne(0))
        panel[f"{label}_z_signed_log_flow"] = np.sign(panel[f"{label}_z_signed"]) * np.log1p(panel[f"{label}_z_signed"].abs())
        feature_columns += raw + [f"{label}_log_gross", f"{label}_signed_log_flow",
                                  f"{label}_mean_wallet_z", f"{label}_z_signed_log_flow"]
    panel.to_csv(OUT / "continuous_price_panel.csv.gz", index=False, compression="gzip")
    panel["any_price_update_5m"] = panel.delta_yes_price.ne(0).astype(float)
    panel["absolute_lagged_30m_price_change"] = panel.lagged_30m_price_change.abs()
    controls_update = ["large_log_gross", "ordinary_log_gross", "large_mean_wallet_z",
                       "ordinary_mean_wallet_z", "absolute_lagged_30m_price_change", "yes_price_t",
                       "log_lagged_30m_volume", "log_minutes_to_kickoff"]
    controls_direction = ["large_signed_log_flow", "ordinary_signed_log_flow",
                          "large_z_signed_log_flow", "ordinary_z_signed_log_flow",
                          "lagged_30m_price_change", "yes_price_t", "log_lagged_30m_volume",
                          "log_minutes_to_kickoff"]
    for model, outcome, terms, weights in [
        ("PRICE_UPDATE_LPM", "any_price_update_5m", controls_update,
         {"large_mean_wallet_z": 1, "ordinary_mean_wallet_z": -1}),
        ("DIRECTIONAL_PRICE_CHANGE", "delta_yes_price", controls_direction,
         {"large_z_signed_log_flow": 1, "ordinary_z_signed_log_flow": -1}),
    ]:
        c, d, beta, cov, names, clusters = fit(panel, outcome, terms, model)
        coefficient_rows += c; model_diagnostics.append(d)
        contrast_rows.append(linear_test(model, outcome, "continuous_wallet_scale_x_large",
                                         weights, beta, cov, names, clusters))

    pd.DataFrame(coefficient_rows).to_csv(OUT / "coefficients.csv", index=False)
    contrast_frame = pd.DataFrame(contrast_rows)
    order = np.argsort(contrast_frame.p_value.to_numpy())
    adjusted = np.empty(len(order)); running = 0.0
    for rank, position in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * contrast_frame.p_value.iloc[position]))
        adjusted[position] = running
    contrast_frame["holm_p_value_six_focal_tests"] = adjusted
    contrast_frame.to_csv(OUT / "contrasts.csv", index=False)
    pd.DataFrame(model_diagnostics).to_csv(OUT / "model_diagnostics.csv", index=False)
    summary = {"generated_at": now(), "status": "complete_qc_passed",
               "definition": "daily past-only expanding cumulative gross USDC; log1p then within-day population z-score",
               "history_markets": 360, "evaluation_market_family": "match only, prematch",
               "evaluation_start": START.isoformat(), "evaluation_end_inclusive": END.isoformat(),
               "evaluation_actions": len(actions), "price_market_minutes": len(panel),
               "markets": int(actions.market_id.nunique()), "events": int(actions.event_id.nunique()),
               "fixed_effects": ["market_id", "calendar_hour_utc"], "cluster": "event_id CRV1",
               "threshold": None, "minimum_activity_rule": "at least one prior observed trade",
               "country_models_run": False, "post_kickoff_models_run": False}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(contrast_frame.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
