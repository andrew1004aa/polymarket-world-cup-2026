#!/usr/bin/env python3
"""Build resumable within-market P99/P95 threshold diagnostics for 360 markets."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/"regression_ready/trade_files.csv"
OUT=ROOT/"large_trade_diagnostics/v3"
CHECKPOINTS=OUT/"market_checkpoints"
START=1780272000
END=1785542400


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        while chunk:=f.read(1024*1024): h.update(chunk)
    return h.hexdigest()


def atomic_json(path:Path,obj:object)->None:
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+"\n"); tmp.replace(path)


def atomic_csv(path:Path,frame:pd.DataFrame)->None:
    tmp=path.with_suffix(path.suffix+".tmp"); frame.to_csv(tmp,index=False); tmp.replace(path)


def checkpoint(row,force:bool)->dict:
    target=CHECKPOINTS/f"{int(row.market_id)}.json"
    source=ROOT/row.path
    if target.exists() and not force:
        prior=json.loads(target.read_text())
        if prior.get("source_sha256")==row.sha256 and prior.get("status")=="complete": return prior
    data=pd.read_csv(source,usecols=["timestamp","trade_value_usdc"])
    data=data[(data.timestamp>=START)&(data.timestamp<END)]
    values=data.trade_value_usdc.to_numpy(dtype=float)
    if len(values)==0 or not np.isfinite(values).all() or (values<0).any():
        raise RuntimeError(f"Invalid trade values for market {row.market_id}")
    p95=float(np.quantile(values,.95,method="linear")); p99=float(np.quantile(values,.99,method="linear"))
    p95_mask=values>=p95; p99_mask=values>=p99
    result={"status":"complete","market_id":int(row.market_id),"source_path":row.path,
        "source_sha256":row.sha256,"trade_count":int(len(values)),"total_value_usdc":float(values.sum()),
        "p95_threshold_usdc":p95,"p99_threshold_usdc":p99,
        "p95_trade_count":int(p95_mask.sum()),"p99_trade_count":int(p99_mask.sum()),
        "p95_actual_share":float(p95_mask.mean()),"p99_actual_share":float(p99_mask.mean()),
        "p95_tie_count":int((values==p95).sum()),"p99_tie_count":int((values==p99).sum()),
        "p95_value_usdc":float(values[p95_mask].sum()),"p99_value_usdc":float(values[p99_mask].sum()),
        "p99_value_share":float(values[p99_mask].sum()/values.sum()),
        "extreme_1pct_market_volume_threshold_usdc":float(values.sum()*.01),
        "max_trade_value_usdc":float(values.max()),"completed_at":datetime.now(timezone.utc).isoformat()}
    atomic_json(target,result); return result


def main()->int:
    parser=argparse.ArgumentParser(); parser.add_argument("--force",action="store_true"); parser.add_argument("--build-only",action="store_true")
    args=parser.parse_args(); OUT.mkdir(parents=True,exist_ok=True); CHECKPOINTS.mkdir(parents=True,exist_ok=True)
    manifest=pd.read_csv(MANIFEST)
    if len(manifest)!=360: raise SystemExit(f"Expected 360 markets, observed {len(manifest)}")
    completed=[]
    if not args.build_only:
        for i,row in enumerate(manifest.itertuples(),1):
            completed.append(checkpoint(row,args.force))
            if i%10==0 or i==360: print(f"{i}/360 thresholds checkpointed",flush=True)
    else:
        for row in manifest.itertuples():
            path=CHECKPOINTS/f"{int(row.market_id)}.json"
            if not path.exists(): raise SystemExit(f"Missing checkpoint: {path}")
            completed.append(json.loads(path.read_text()))
    frame=pd.DataFrame(completed).merge(
        pd.read_csv(ROOT/"regression_ready/tables/markets.csv",usecols=["market_id","market_type","question"]),
        on="market_id",how="left",validate="one_to_one")
    columns=["market_id","market_type","question","trade_count","total_value_usdc","p99_threshold_usdc",
             "p99_trade_count","p99_actual_share","p99_tie_count","p99_value_usdc","p99_value_share",
             "p95_threshold_usdc","p95_trade_count","p95_actual_share","p95_tie_count","p95_value_usdc",
             "extreme_1pct_market_volume_threshold_usdc","max_trade_value_usdc","source_path","source_sha256"]
    frame=frame[columns].sort_values(["market_type","market_id"])
    atomic_csv(OUT/"market_thresholds.csv",frame)
    summary={"version":"v3","status":"threshold_diagnostics_complete_no_regression","generated_at":datetime.now(timezone.utc).isoformat(),
        "markets":len(frame),"match_markets":int((frame.market_type=="match").sum()),
        "outright_markets":int((frame.market_type=="outright").sum()),
        "eligible_trades":int(frame.trade_count.sum()),"minimum_market_trades":int(frame.trade_count.min()),
        "markets_under_1000_trades":int((frame.trade_count<1000).sum()),
        "markets_under_5_p99_trades":int((frame.p99_trade_count<5).sum()),
        "markets_with_no_p99_trades":int((frame.p99_trade_count==0).sum()),
        "p99_trades":int(frame.p99_trade_count.sum()),
        "p99_actual_trade_share":float(frame.p99_trade_count.sum()/frame.trade_count.sum()),
        "p99_value_share":float(frame.p99_value_usdc.sum()/frame.total_value_usdc.sum()),
        "p99_tie_markets":int((frame.p99_tie_count>0).sum()),
        "threshold_table_sha256":sha256(OUT/"market_thresholds.csv"),"source_manifest_sha256":sha256(MANIFEST)}
    atomic_json(OUT/"summary.json",summary)
    report=f"""# Within-market large-trade threshold diagnostics\n\nNo regression is estimated by this script.\n\n- Markets: {summary['markets']} ({summary['match_markets']} match; {summary['outright_markets']} outright)\n- Eligible trades: {summary['eligible_trades']:,}\n- Minimum market trades: {summary['minimum_market_trades']:,}\n- Markets below 1,000 trades: {summary['markets_under_1000_trades']}\n- Markets with fewer than five P99 trades: {summary['markets_under_5_p99_trades']}\n- Markets with no P99 trades: {summary['markets_with_no_p99_trades']}\n- P99 classified trades: {summary['p99_trades']:,} ({summary['p99_actual_trade_share']:.3%})\n- P99 classified value share: {summary['p99_value_share']:.3%}\n- Markets with P99 threshold ties: {summary['p99_tie_markets']}\n"""
    (OUT/"threshold_report.md").write_text(report)
    print(json.dumps(summary,indent=2),flush=True); return 0


if __name__=="__main__": raise SystemExit(main())
