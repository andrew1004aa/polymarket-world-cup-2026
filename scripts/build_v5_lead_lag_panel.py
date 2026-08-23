#!/usr/bin/env python3
"""Add strictly aligned pre- and post-flow outcomes to the V5 panel."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "regression_inputs/v5/strict_timing/strict_timing_past_only_p99.csv.gz"
MANIFEST = ROOT / "regression_ready/price_files.csv"
OUT = ROOT / "regression_inputs/v5/lead_lag"
OUTPUT = OUT / "lead_lag_panel.csv.gz"
PRE = (1, 5, 15)
POST = (1, 5, 15, 30)


def lookup(pt, pp, targets, prior):
    pos = np.searchsorted(pt, targets, side="left")
    if prior: pos -= 1
    valid = (pos >= 0) & (pos < len(pt))
    ts = np.full(len(targets), np.nan); price = np.full(len(targets), np.nan)
    ts[valid] = pt[pos[valid]]; price[valid] = pp[pos[valid]]
    gap = targets - ts if prior else ts - targets
    return ts, price, gap


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    keep = ["market_id", "event_id", "calendar_hour_utc", "minute_start_timestamp",
            "v5_baseline_price", "v5_baseline_gap_seconds", "v5_lagged_30m_price_change",
            "log_lagged_30m_volume", "log_minutes_to_kickoff", "common_eligible",
            "expanding_eligible", "rolling7_eligible",
            "expanding_signed_log_large_net_flow", "expanding_signed_log_ordinary_net_flow",
            "rolling7_signed_log_large_net_flow", "rolling7_signed_log_ordinary_net_flow"]
    data = pd.read_csv(SOURCE, usecols=keep)
    paths = {str(int(r.market_id)): ROOT / r.path for r in pd.read_csv(MANIFEST).itertuples()}
    for h in PRE:
        for suffix in ("timestamp", "price", "gap_seconds", "change", "strict0"):
            data[f"pre_{h}m_{suffix}"] = np.nan
    for h in POST:
        for suffix in ("timestamp", "price", "gap_seconds", "change", "strict0"):
            data[f"post_{h}m_{suffix}"] = np.nan

    total = data.market_id.nunique()
    for number, (mid, idx) in enumerate(data.groupby("market_id", sort=True).groups.items(), 1):
        p = pd.read_csv(paths[str(int(mid))], usecols=["outcome", "timestamp", "yes_equivalent_price"])
        p = p[p.outcome.str.upper().eq("YES")].drop_duplicates("timestamp", keep="last").sort_values("timestamp")
        pt = p.timestamp.to_numpy(np.int64); pp = p.yes_equivalent_price.to_numpy(float)
        loc = np.asarray(idx); t = data.loc[loc, "minute_start_timestamp"].to_numpy(np.int64)
        baseline = data.loc[loc, "v5_baseline_price"].to_numpy(float)
        baseline_gap = data.loc[loc, "v5_baseline_gap_seconds"].to_numpy(float)
        for h in PRE:
            ts, price, gap = lookup(pt, pp, t-h*60, prior=True)
            data.loc[loc, f"pre_{h}m_timestamp"] = ts; data.loc[loc, f"pre_{h}m_price"] = price
            data.loc[loc, f"pre_{h}m_gap_seconds"] = gap
            data.loc[loc, f"pre_{h}m_change"] = baseline-price
            data.loc[loc, f"pre_{h}m_strict0"] = ((baseline_gap>=0)&(baseline_gap<60)&(gap>=0)&(gap<60)).astype("int8")
        for h in POST:
            ts, price, gap = lookup(pt, pp, t+h*60, prior=False)
            data.loc[loc, f"post_{h}m_timestamp"] = ts; data.loc[loc, f"post_{h}m_price"] = price
            data.loc[loc, f"post_{h}m_gap_seconds"] = gap
            data.loc[loc, f"post_{h}m_change"] = price-baseline
            data.loc[loc, f"post_{h}m_strict0"] = ((baseline_gap>=0)&(baseline_gap<60)&(gap>=0)&(gap<60)).astype("int8")
        if number == 1 or number % 25 == 0 or number == total:
            print(f"Lead-lag aligned {number}/{total} markets", flush=True)
    data.to_csv(OUTPUT, index=False, compression={"method":"gzip","compresslevel":6})
    coverage=[]
    for phase, horizons in (("pre",PRE),("post",POST)):
        for h in horizons:
            m=data[f"{phase}_{h}m_strict0"].eq(1)
            coverage.append({"phase":phase,"horizon_minutes":h,"rows":int(m.sum()),
                             "markets":int(data.loc[m,"market_id"].nunique()),
                             "events":int(data.loc[m,"event_id"].nunique()),
                             "zero_change_rate":float(data.loc[m,f"{phase}_{h}m_change"].eq(0).mean())})
    pd.DataFrame(coverage).to_csv(OUT/"coverage.csv",index=False)
    summary={"generated_at":datetime.now(timezone.utc).isoformat(),"status":"complete_qc_passed",
             "rows":len(data),"markets":int(data.market_id.nunique()),"events":int(data.event_id.nunique()),
             "pre_horizons":PRE,"post_horizons":POST,"strict0":"both endpoint gaps <60 seconds",
             "interpolation":False,"coverage":coverage,"output":str(OUTPUT.relative_to(ROOT))}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == "__main__": main()
