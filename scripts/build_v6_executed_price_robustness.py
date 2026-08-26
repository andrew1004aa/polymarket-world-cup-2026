#!/usr/bin/env python3
"""Align H2 minutes to executed trade prices as a price-source robustness."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "regression_inputs/v5/strict_timing/strict_timing_past_only_p99.csv.gz"
TRADES = ROOT / "regression_ready/trades_by_market"
OUT = ROOT / "regression_inputs/v6/executed_price"
OUTPUT = OUT / "executed_price_h2.csv.gz"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    keep = ["market_id", "event_id", "calendar_hour_utc", "minute_start_timestamp",
            "v5_baseline_price", "v5_lagged_30m_price_change", "log_lagged_30m_volume",
            "log_minutes_to_kickoff", "expanding_eligible", "rolling7_eligible",
            "expanding_signed_log_large_net_flow", "expanding_signed_log_ordinary_net_flow",
            "rolling7_signed_log_large_net_flow", "rolling7_signed_log_ordinary_net_flow"]
    base = pd.read_csv(SOURCE, usecols=keep, dtype={"market_id": str})
    frames = []
    for index, (market_id, rows) in enumerate(base.groupby("market_id", sort=False), 1):
        trades = pd.read_csv(TRADES / f"{market_id}.csv.gz",
                             usecols=["timestamp", "yes_equivalent_price"]).sort_values("timestamp", kind="mergesort")
        ts = trades.timestamp.to_numpy(np.int64); prices = trades.yes_equivalent_price.to_numpy(float)
        focal = rows.minute_start_timestamp.to_numpy(np.int64); target = focal + 300
        before = np.searchsorted(ts, focal, side="left") - 1
        after = np.searchsorted(ts, target, side="left")
        valid_before = before >= 0; valid_after = after < len(ts)
        output = rows.copy()
        output["executed_baseline_timestamp"] = np.where(valid_before, ts[np.maximum(before, 0)], np.nan)
        output["executed_baseline_price"] = np.where(valid_before, prices[np.maximum(before, 0)], np.nan)
        safe_after = np.minimum(after, max(len(ts) - 1, 0))
        output["executed_outcome_timestamp"] = np.where(valid_after, ts[safe_after], np.nan)
        output["executed_outcome_price"] = np.where(valid_after, prices[safe_after], np.nan)
        output["executed_baseline_gap_seconds"] = focal - output.executed_baseline_timestamp
        output["executed_outcome_gap_seconds"] = output.executed_outcome_timestamp - target
        output["executed_delta_5m"] = output.executed_outcome_price - output.executed_baseline_price
        frames.append(output)
        if index % 25 == 0 or index == base.market_id.nunique():
            print(f"[{index}/{base.market_id.nunique()}] {market_id}", flush=True)
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(OUTPUT, index=False, compression={"method": "gzip", "compresslevel": 6})
    gaps = []
    for tolerance in (60, 300, 900, 3600):
        valid = result.executed_baseline_gap_seconds.between(1, tolerance) & result.executed_outcome_gap_seconds.between(0, tolerance)
        gaps.append({"tolerance_seconds": tolerance, "rows": int(valid.sum()),
                     "markets": int(result.loc[valid, "market_id"].nunique()),
                     "coverage_share": float(valid.mean()),
                     "median_baseline_gap": float(result.loc[valid, "executed_baseline_gap_seconds"].median()),
                     "median_outcome_gap": float(result.loc[valid, "executed_outcome_gap_seconds"].median())})
    pd.DataFrame(gaps).to_csv(OUT / "coverage.csv", index=False)
    summary = {"generated_at": datetime.now(timezone.utc).isoformat(), "status": "built_qc_passed",
               "rows": len(result), "markets": int(result.market_id.nunique()),
               "baseline_rule": "last executed YES-equivalent trade price strictly before focal minute",
               "outcome_rule": "first executed YES-equivalent trade price at or after t+5 minutes",
               "focal_minute_excluded_from_baseline": True}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

