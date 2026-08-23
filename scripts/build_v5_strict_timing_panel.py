#!/usr/bin/env python3
"""Re-align V5 prices so all baselines precede the flow window."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "regression_inputs/v5/past_only_p99/past_only_p99_primary_sample.csv.gz"
PRICE_MANIFEST = ROOT / "regression_ready/price_files.csv"
OUT = ROOT / "regression_inputs/v5/strict_timing"
OUTPUT = OUT / "strict_timing_past_only_p99.csv.gz"
HORIZONS = (1, 5, 15)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def align(prices_t: np.ndarray, prices_p: np.ndarray, targets: np.ndarray, prior: bool):
    side = "left" if prior else "left"
    positions = np.searchsorted(prices_t, targets, side=side)
    if prior:
        positions -= 1  # strictly earlier than target
    valid = (positions >= 0) & (positions < len(prices_t))
    timestamps = np.full(len(targets), np.nan)
    values = np.full(len(targets), np.nan)
    timestamps[valid] = prices_t[positions[valid]]
    values[valid] = prices_p[positions[valid]]
    gaps = targets - timestamps if prior else timestamps - targets
    return timestamps, values, gaps


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(SOURCE)
    manifest = pd.read_csv(PRICE_MANIFEST)
    price_paths = {str(row.market_id): ROOT / row.path for row in manifest.itertuples()}

    # Activity is defined ex ante from the full prematch flow panel, before any
    # price-gap filtering.
    volume = data.groupby("market_id", observed=True).gross_volume_usdc.sum()
    median_volume = float(volume.median())
    data["active_market_above_median"] = data.market_id.map(volume.gt(median_volume)).astype("int8")

    result_columns = ["v5_baseline_timestamp", "v5_baseline_price", "v5_baseline_gap_seconds",
                      "v5_lag30_timestamp", "v5_lag30_price", "v5_lag30_gap_seconds",
                      "v5_lagged_30m_price_change"]
    for h in HORIZONS:
        result_columns += [f"v5_post_{h}m_timestamp", f"v5_post_{h}m_price",
                           f"v5_post_{h}m_gap_seconds", f"v5_delta_{h}m",
                           f"v5_strict0_{h}m", f"v5_strict1_{h}m"]
    for column in result_columns:
        data[column] = np.nan

    for number, (market_id, indexes) in enumerate(data.groupby("market_id", sort=True).groups.items(), 1):
        key = str(int(market_id))
        path = price_paths[key]
        prices = pd.read_csv(path, usecols=["outcome", "timestamp", "yes_equivalent_price"])
        prices = prices.loc[prices.outcome.str.upper().eq("YES"), ["timestamp", "yes_equivalent_price"]]
        prices = prices.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
        pt = prices.timestamp.to_numpy(np.int64); pp = prices.yes_equivalent_price.to_numpy(float)
        loc = np.asarray(indexes); t = data.loc[loc, "minute_start_timestamp"].to_numpy(np.int64)

        bts, bp, bg = align(pt, pp, t, prior=True)
        lts, lp, lg = align(pt, pp, t - 1800, prior=True)
        data.loc[loc, "v5_baseline_timestamp"] = bts
        data.loc[loc, "v5_baseline_price"] = bp
        data.loc[loc, "v5_baseline_gap_seconds"] = bg
        data.loc[loc, "v5_lag30_timestamp"] = lts
        data.loc[loc, "v5_lag30_price"] = lp
        data.loc[loc, "v5_lag30_gap_seconds"] = lg
        data.loc[loc, "v5_lagged_30m_price_change"] = bp - lp

        for h in HORIZONS:
            target = t + h * 60
            fts, fp, fg = align(pt, pp, target, prior=False)
            data.loc[loc, f"v5_post_{h}m_timestamp"] = fts
            data.loc[loc, f"v5_post_{h}m_price"] = fp
            data.loc[loc, f"v5_post_{h}m_gap_seconds"] = fg
            data.loc[loc, f"v5_delta_{h}m"] = fp - bp
            # '0 minute' means both absolute gaps are in [0, 60) seconds;
            # '1 minute' permits [0, 120). No interpolation or forward fill.
            data.loc[loc, f"v5_strict0_{h}m"] = ((bg >= 0) & (bg < 60) & (fg >= 0) & (fg < 60)).astype("int8")
            data.loc[loc, f"v5_strict1_{h}m"] = ((bg >= 0) & (bg < 120) & (fg >= 0) & (fg < 120)).astype("int8")
        if number == 1 or number % 25 == 0 or number == data.market_id.nunique():
            print(f"Aligned {number}/{data.market_id.nunique()} markets", flush=True)

    data.to_csv(OUTPUT, index=False, compression={"method": "gzip", "compresslevel": 6})
    gap_rows = []
    for h in HORIZONS:
        for label in ("strict0", "strict1"):
            mask = data[f"v5_{label}_{h}m"].eq(1)
            gap_rows.append({"horizon_minutes": h, "sample": label, "rows": int(mask.sum()),
                             "markets": int(data.loc[mask, "market_id"].nunique()),
                             "events": int(data.loc[mask, "event_id"].nunique()),
                             "zero_change_rows": int(data.loc[mask, f"v5_delta_{h}m"].eq(0).sum()),
                             "zero_change_rate": float(data.loc[mask, f"v5_delta_{h}m"].eq(0).mean())})
    pd.DataFrame(gap_rows).to_csv(OUT / "strict_sample_coverage.csv", index=False)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "status": "complete_qc_passed",
        "source": str(SOURCE.relative_to(ROOT)), "source_sha256": sha256(SOURCE),
        "rows": len(data), "markets": int(data.market_id.nunique()), "events": int(data.event_id.nunique()),
        "market_volume_median_usdc": median_volume,
        "active_markets": int(volume.gt(median_volume).sum()),
        "timing": {"flow": "[t,t+1 minute)", "baseline": "latest price strictly before t",
                   "post": "first price at or after t+h", "strict0": "both gaps <60 seconds",
                   "strict1": "both gaps <120 seconds", "interpolation": False},
        "output": str(OUTPUT.relative_to(ROOT)), "output_sha256": sha256(OUTPUT),
        "coverage": gap_rows,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
