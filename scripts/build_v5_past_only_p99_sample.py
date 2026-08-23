#!/usr/bin/env python3
"""Build expanding and rolling past-only P99 H2 model samples."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "regression_inputs/v3/primary_prematch_5m_complete_case.csv.gz"
TRADES = ROOT / "regression_ready/trades_by_market"
OUT = ROOT / "regression_inputs/v5/past_only_p99"
MIN_HISTORY = 1000
START = pd.Timestamp("2026-06-01", tz="UTC")


def now(): return datetime.now(timezone.utc).isoformat()


def signed_direction(side, outcome):
    return np.where((side.str.upper().eq("BUY")) == (outcome.str.upper().eq("YES")), 1.0, -1.0)


def threshold(values):
    return float(np.quantile(values, .99, method="linear")) if len(values) >= MIN_HISTORY else np.nan


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(BASE, dtype={"market_id": str})
    base["evaluation_day"] = pd.to_datetime(base.minute_start_timestamp, unit="s", utc=True).dt.floor("D")
    base["common_eligible"] = 0
    feature_frames = []
    threshold_rows = []
    markets = sorted(base.market_id.unique(), key=int)
    if len(markets) != 312: raise RuntimeError(f"Expected 312 match markets, found {len(markets)}")

    for index, market_id in enumerate(markets, 1):
        required = base.loc[base.market_id.eq(market_id), ["minute_start_timestamp", "evaluation_day"]]
        required_minutes = set(required.minute_start_timestamp.astype(int))
        eval_days = sorted(required.evaluation_day.unique())
        path = TRADES / f"{market_id}.csv.gz"
        trades = pd.read_csv(path, usecols=["timestamp", "side", "outcome", "trade_value_usdc"])
        trades["timestamp"] = trades.timestamp.astype("int64")
        trades["evaluation_day"] = pd.to_datetime(trades.timestamp, unit="s", utc=True).dt.floor("D")
        trades["minute_start_timestamp"] = (trades.timestamp // 60) * 60
        trades["signed_value"] = signed_direction(trades.side, trades.outcome) * trades.trade_value_usdc

        thresholds = {}
        for day in eval_days:
            expanding_values = trades.loc[(trades.evaluation_day >= START) & (trades.evaluation_day < day), "trade_value_usdc"].to_numpy()
            rolling_start = day - pd.Timedelta(days=7)
            rolling_values = trades.loc[(trades.evaluation_day >= rolling_start) & (trades.evaluation_day < day), "trade_value_usdc"].to_numpy()
            exp = threshold(expanding_values); roll = threshold(rolling_values)
            thresholds[day] = {"expanding": exp, "rolling7": roll}
            threshold_rows += [
                {"market_id": market_id, "evaluation_day": str(day.date()), "specification": "expanding",
                 "history_start": str(START.date()), "history_end_exclusive": str(day.date()),
                 "history_trades": len(expanding_values), "p99_threshold_usdc": exp,
                 "eligible": int(np.isfinite(exp))},
                {"market_id": market_id, "evaluation_day": str(day.date()), "specification": "rolling7",
                 "history_start": str(rolling_start.date()), "history_end_exclusive": str(day.date()),
                 "history_trades": len(rolling_values), "p99_threshold_usdc": roll,
                 "eligible": int(np.isfinite(roll))},
            ]

        relevant = trades[trades.minute_start_timestamp.isin(required_minutes)].copy()
        rows = []
        for specification in ("expanding", "rolling7"):
            relevant[f"{specification}_threshold"] = relevant.evaluation_day.map(
                {day: value[specification] for day, value in thresholds.items()})
            eligible = relevant[f"{specification}_threshold"].notna()
            large = eligible & relevant.trade_value_usdc.ge(relevant[f"{specification}_threshold"])
            relevant[f"{specification}_large_signed"] = np.where(large, relevant.signed_value, 0.0)
            relevant[f"{specification}_ordinary_signed"] = np.where(eligible & ~large, relevant.signed_value, 0.0)
            relevant[f"{specification}_large_count"] = large.astype(int)
            relevant[f"{specification}_ordinary_count"] = (eligible & ~large).astype(int)
            grouped = relevant.groupby("minute_start_timestamp", as_index=False).agg(
                **{f"{specification}_large_net_flow_usdc": (f"{specification}_large_signed", "sum"),
                   f"{specification}_ordinary_net_flow_usdc": (f"{specification}_ordinary_signed", "sum"),
                   f"{specification}_large_trade_count": (f"{specification}_large_count", "sum"),
                   f"{specification}_ordinary_trade_count": (f"{specification}_ordinary_count", "sum")})
            rows.append(grouped)
        market_features = rows[0].merge(rows[1], on="minute_start_timestamp", how="outer", validate="one_to_one")
        market_features["market_id"] = market_id
        feature_frames.append(market_features)
        if index % 25 == 0 or index == len(markets): print(f"[{index}/{len(markets)}] {market_id}", flush=True)

    features = pd.concat(feature_frames, ignore_index=True)
    output = base.merge(features, on=["market_id", "minute_start_timestamp"], how="left", validate="one_to_one")
    for specification in ("expanding", "rolling7"):
        threshold_map = {(row["market_id"], row["evaluation_day"]): row["p99_threshold_usdc"]
                         for row in threshold_rows if row["specification"] == specification}
        keys = zip(output.market_id, output.evaluation_day.dt.strftime("%Y-%m-%d"))
        output[f"{specification}_p99_threshold_usdc"] = [threshold_map.get(key, np.nan) for key in keys]
        output[f"{specification}_eligible"] = output[f"{specification}_p99_threshold_usdc"].notna().astype(int)
        for suffix in ("large_net_flow_usdc", "ordinary_net_flow_usdc", "large_trade_count", "ordinary_trade_count"):
            column = f"{specification}_{suffix}"; output[column] = output[column].fillna(0)
        for label in ("large", "ordinary"):
            net = output[f"{specification}_{label}_net_flow_usdc"]
            output[f"{specification}_signed_log_{label}_net_flow"] = np.sign(net) * np.log1p(net.abs())
    output["common_eligible"] = output.expanding_eligible * output.rolling7_eligible
    if output.model_sample_record_id.duplicated().any(): raise RuntimeError("Duplicate model record IDs")
    thresholds_frame = pd.DataFrame(threshold_rows)
    thresholds_frame.to_csv(OUT / "daily_thresholds.csv.gz", index=False, compression="gzip")
    output.to_csv(OUT / "past_only_p99_primary_sample.csv.gz", index=False, compression="gzip")
    summary = {"generated_at": now(), "status": "built_qc_passed", "source_rows": len(base),
               "markets": len(markets), "minimum_prior_trades": MIN_HISTORY,
               "expanding_eligible_rows": int(output.expanding_eligible.sum()),
               "rolling7_eligible_rows": int(output.rolling7_eligible.sum()),
               "common_eligible_rows": int(output.common_eligible.sum()),
               "expanding_large_trades": int(output.expanding_large_trade_count.sum()),
               "rolling7_large_trades": int(output.rolling7_large_trade_count.sum()),
               "missing_core_values": int(output[["delta_yes_price", "yes_price_t"]].isna().sum().sum())}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__": raise SystemExit(main())
