#!/usr/bin/env python3
"""Build past-only threshold sensitivity features at P95/P97.5/P99/P99.5."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "regression_inputs/v5/strict_timing/strict_timing_past_only_p99.csv.gz"
TRADES = ROOT / "regression_ready/trades_by_market"
OUT = ROOT / "regression_inputs/v6/threshold_sensitivity"
OUTPUT = OUT / "past_only_threshold_sensitivity.csv.gz"
START = pd.Timestamp("2026-06-01", tz="UTC")
MIN_HISTORY = 1000
QUANTILES = {"p95": .95, "p975": .975, "p99": .99, "p995": .995}


def direction(side: pd.Series, outcome: pd.Series) -> np.ndarray:
    return np.where(side.str.upper().eq("BUY").eq(outcome.str.upper().eq("YES")), 1.0, -1.0)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    keep = ["market_id", "event_id", "calendar_hour_utc", "minute_start_timestamp",
            "v5_delta_5m", "v5_baseline_price", "v5_lagged_30m_price_change",
            "log_lagged_30m_volume", "log_minutes_to_kickoff", "v5_strict0_5m"]
    base = pd.read_csv(BASE, usecols=keep, dtype={"market_id": str})
    base["evaluation_day"] = pd.to_datetime(base.minute_start_timestamp, unit="s", utc=True).dt.floor("D")
    feature_frames = []; threshold_rows = []
    markets = sorted(base.market_id.unique(), key=int)
    for index, market_id in enumerate(markets, 1):
        required = base.loc[base.market_id.eq(market_id), ["minute_start_timestamp", "evaluation_day"]]
        required_minutes = set(required.minute_start_timestamp.astype(int))
        eval_days = sorted(required.evaluation_day.unique())
        trades = pd.read_csv(TRADES / f"{market_id}.csv.gz",
                             usecols=["timestamp", "side", "outcome", "trade_value_usdc"])
        trades["timestamp"] = trades.timestamp.astype("int64")
        trades["evaluation_day"] = pd.to_datetime(trades.timestamp, unit="s", utc=True).dt.floor("D")
        trades["minute_start_timestamp"] = (trades.timestamp // 60) * 60
        trades["signed_value"] = direction(trades.side, trades.outcome) * trades.trade_value_usdc
        thresholds = {}
        for day in eval_days:
            exp_values = trades.loc[(trades.evaluation_day >= START) & (trades.evaluation_day < day), "trade_value_usdc"].to_numpy()
            rolling_start = day - pd.Timedelta(days=7)
            roll_values = trades.loc[(trades.evaluation_day >= rolling_start) & (trades.evaluation_day < day), "trade_value_usdc"].to_numpy()
            thresholds[day] = {}
            for spec, values, history_start in (("expanding", exp_values, START), ("rolling7", roll_values, rolling_start)):
                thresholds[day][spec] = {}
                for label, q in QUANTILES.items():
                    value = float(np.quantile(values, q, method="linear")) if len(values) >= MIN_HISTORY else np.nan
                    thresholds[day][spec][label] = value
                    threshold_rows.append({"market_id": market_id, "evaluation_day": str(day.date()),
                                           "specification": spec, "percentile": label,
                                           "history_start": str(history_start.date()),
                                           "history_end_exclusive": str(day.date()),
                                           "history_trades": len(values), "threshold_usdc": value,
                                           "eligible": int(np.isfinite(value))})
        relevant = trades[trades.minute_start_timestamp.isin(required_minutes)].copy()
        market_features = required[["minute_start_timestamp"]].drop_duplicates().copy()
        for spec in ("expanding", "rolling7"):
            for label in QUANTILES:
                column = f"{spec}_{label}_threshold_usdc"
                relevant[column] = relevant.evaluation_day.map({d: thresholds[d][spec][label] for d in eval_days})
                eligible = relevant[column].notna(); large = eligible & relevant.trade_value_usdc.ge(relevant[column])
                relevant[f"{spec}_{label}_large_signed"] = np.where(large, relevant.signed_value, 0.0)
                relevant[f"{spec}_{label}_ordinary_signed"] = np.where(eligible & ~large, relevant.signed_value, 0.0)
                grouped = relevant.groupby("minute_start_timestamp", as_index=False).agg(
                    **{f"{spec}_{label}_large_flow": (f"{spec}_{label}_large_signed", "sum"),
                       f"{spec}_{label}_ordinary_flow": (f"{spec}_{label}_ordinary_signed", "sum"),
                       f"{spec}_{label}_large_trades": (f"{spec}_{label}_large_signed", lambda x: int(np.count_nonzero(x)))})
                market_features = market_features.merge(grouped, on="minute_start_timestamp", how="left", validate="one_to_one")
                threshold_map = {d: thresholds[d][spec][label] for d in eval_days}
                day_map = required.drop_duplicates("minute_start_timestamp").set_index("minute_start_timestamp").evaluation_day
                market_features[f"{spec}_{label}_threshold_usdc"] = market_features.minute_start_timestamp.map(day_map).map(threshold_map)
                market_features[f"{spec}_{label}_eligible"] = market_features[f"{spec}_{label}_threshold_usdc"].notna().astype("int8")
        market_features["market_id"] = market_id; feature_frames.append(market_features)
        if index % 20 == 0 or index == len(markets): print(f"[{index}/{len(markets)}] {market_id}", flush=True)
    features = pd.concat(feature_frames, ignore_index=True)
    output = base.merge(features, on=["market_id", "minute_start_timestamp"], how="left", validate="one_to_one")
    for spec in ("expanding", "rolling7"):
        for label in QUANTILES:
            for group in ("large", "ordinary"):
                col = f"{spec}_{label}_{group}_flow"; output[col] = output[col].fillna(0.0)
    output.to_csv(OUTPUT, index=False, compression={"method": "gzip", "compresslevel": 6})
    pd.DataFrame(threshold_rows).to_csv(OUT / "daily_thresholds.csv.gz", index=False, compression="gzip")
    summary = {"generated_at": datetime.now(timezone.utc).isoformat(), "status": "built_qc_passed",
               "rows": len(output), "markets": len(markets), "minimum_prior_trades": MIN_HISTORY,
               "percentiles": QUANTILES}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

