#!/usr/bin/env python3
"""Build the method-neutral v2 all-horizon market-minute panel."""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import json
import math
import shutil
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from build_market_minute_panel import (
    ROOT, SOURCE, ANALYSIS_ROOT, atomic_json, fmt, load_manifests,
    load_markets, load_primary_whales, load_yes_prices, parse_utc, sha256,
    signed_log, trade_direction, utc_iso,
)


BASE_FIELDS = [
    "analysis_record_id", "analysis_version", "market_id", "condition_id",
    "event_id", "fifa_match_id", "market_type", "market_subtype",
    "market_role", "question", "country", "stage", "home_team", "away_team",
    "minute_start_timestamp", "minute_start_utc", "minute_end_timestamp",
    "minute_end_utc", "date_utc", "hour_of_day_utc", "calendar_hour_utc",
    "calendar_15min_block_utc", "phase_at_minute_start",
    "phase_at_minute_end", "actual_kickoff_utc", "resolved_on_timestamp",
    "minutes_from_actual_kickoff", "minutes_to_resolution",
    "minute_contains_kickoff", "minute_contains_resolution",
    "contains_post_kickoff_trade", "contains_post_resolution_trade",
    "pre_kickoff_trade_count", "post_kickoff_trade_count",
    "pre_resolution_trade_count", "post_resolution_trade_count",
    "yes_outcome_won", "trade_count", "distinct_wallet_count",
    "gross_volume_usdc", "net_signed_flow_usdc", "whale_trade_count",
    "whale_distinct_wallet_count", "whale_gross_volume_usdc",
    "whale_net_signed_flow_usdc", "nonwhale_trade_count",
    "nonwhale_distinct_wallet_count", "nonwhale_gross_volume_usdc",
    "nonwhale_net_signed_flow_usdc", "whale_flow_share",
    "signed_log_whale_net_flow", "signed_log_nonwhale_net_flow",
    "lagged_30m_gross_volume_usdc", "lagged_30m_price_change",
    "baseline_price_timestamp", "baseline_price_utc",
    "baseline_price_age_seconds", "yes_price_t", "baseline_available",
]


def fields(config: dict) -> list[str]:
    result = list(BASE_FIELDS)
    tolerances = [int(x) for x in config["price_alignment"]["convenience_tolerance_flags_minutes"]]
    for horizon in [int(x) for x in config["horizon_minutes"]]:
        result += [
            f"target_timestamp_{horizon}m", f"target_timestamp_{horizon}m_utc",
            f"price_timestamp_{horizon}m", f"price_timestamp_{horizon}m_utc",
            f"price_lateness_{horizon}m_seconds", f"yes_price_{horizon}m",
            f"delta_yes_price_{horizon}m", f"directional_impact_{horizon}m",
            f"brier_improvement_{horizon}m", f"target_price_available_{horizon}m",
            f"crosses_kickoff_{horizon}m", f"crosses_resolution_{horizon}m",
        ]
        result += [f"baseline_within_{tol}m" for tol in tolerances]
        result += [f"target_{horizon}m_within_{tol}m" for tol in tolerances]
    return list(dict.fromkeys(result))


def phase(market_type: str, timestamp: int, kickoff: int | None, resolution: int | None) -> str:
    if resolution is not None and timestamp >= resolution:
        return "post_resolution"
    if market_type == "outright":
        return "pre_resolution"
    if kickoff is None:
        return "unknown"
    return "pre_kickoff" if timestamp < kickoff else "post_kickoff_pre_resolution"


def prior_unbounded(timestamps: list[int], prices: list[float], target: int):
    pos = bisect.bisect_right(timestamps, target) - 1
    return (None, None) if pos < 0 else (timestamps[pos], prices[pos])


def future_unbounded(timestamps: list[int], prices: list[float], target: int):
    pos = bisect.bisect_left(timestamps, target)
    return (None, None) if pos >= len(timestamps) else (timestamps[pos], prices[pos])


def aggregate(path: Path, whales: set[str], kickoff: int | None, resolution: int | None):
    groups = {}
    source_rows = 0
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            source_rows += 1
            ts = int(row["timestamp"]); minute = ts - ts % 60
            if minute not in groups:
                groups[minute] = {
                    "context": row, "trade_count": 0, "wallets": set(), "gross": 0.0, "net": 0.0,
                    "whale_trade_count": 0, "whale_wallets": set(), "whale_gross": 0.0, "whale_net": 0.0,
                    "nonwhale_trade_count": 0, "nonwhale_wallets": set(), "nonwhale_gross": 0.0, "nonwhale_net": 0.0,
                    "pre_kickoff_trade_count": 0, "post_kickoff_trade_count": 0,
                    "pre_resolution_trade_count": 0, "post_resolution_trade_count": 0,
                }
            g = groups[minute]; wallet = row["wallet_address"].lower(); value = float(row["trade_value_usdc"])
            signed = trade_direction(row["side"], row["outcome"]) * value
            g["trade_count"] += 1; g["wallets"].add(wallet); g["gross"] += value; g["net"] += signed
            if kickoff is not None:
                g["pre_kickoff_trade_count" if ts < kickoff else "post_kickoff_trade_count"] += 1
            if resolution is None or ts < resolution:
                g["pre_resolution_trade_count"] += 1
            else:
                g["post_resolution_trade_count"] += 1
            prefix = "whale" if wallet in whales else "nonwhale"
            g[f"{prefix}_trade_count"] += 1; g[f"{prefix}_wallets"].add(wallet)
            g[f"{prefix}_gross"] += value; g[f"{prefix}_net"] += signed
    return groups, source_rows


def make_rows(groups, market, timestamps, prices, config):
    horizons = [int(x) for x in config["horizon_minutes"]]
    tolerances = [int(x) for x in config["price_alignment"]["convenience_tolerance_flags_minutes"]]
    kickoff = None; resolution = parse_utc(market["resolved_on_timestamp"])
    rolling = deque(); rolling_sum = 0.0; result = []
    for minute_start, g in sorted(groups.items()):
        context = g["context"]
        if market["market_type"] == "match" and kickoff is None:
            kickoff = parse_utc(context["actual_kickoff_utc"])
        minute_end = minute_start + 60
        while rolling and rolling[0][0] < minute_start - 1800:
            _, old = rolling.popleft(); rolling_sum -= old
        p_ts, p0 = prior_unbounded(timestamps, prices, minute_end)
        lag_ts, lag_p = prior_unbounded(timestamps, prices, minute_end - 1800)
        dt = datetime.fromtimestamp(minute_start, timezone.utc)
        yes_won = int(context["yes_outcome_won"]) if context["yes_outcome_won"] in ("0", "1") else None
        row = {
            "analysis_record_id": f"all:{market['market_id']}:{minute_start}", "analysis_version": config["version"],
            "market_id": market["market_id"], "condition_id": market["condition_id"],
            "event_id": context["event_id"], "fifa_match_id": context["fifa_match_id"],
            "market_type": market["market_type"], "market_subtype": market["market_subtype"],
            "market_role": context["market_role"], "question": market["question"], "country": market["country"],
            "stage": context["stage"], "home_team": context["home_team"], "away_team": context["away_team"],
            "minute_start_timestamp": minute_start, "minute_start_utc": utc_iso(minute_start),
            "minute_end_timestamp": minute_end, "minute_end_utc": utc_iso(minute_end), "date_utc": dt.strftime("%Y-%m-%d"),
            "hour_of_day_utc": dt.strftime("%H"), "calendar_hour_utc": dt.strftime("%Y-%m-%dT%H:00:00Z"),
            "calendar_15min_block_utc": dt.replace(minute=(dt.minute // 15) * 15).strftime("%Y-%m-%dT%H:%M:00Z"),
            "phase_at_minute_start": phase(market["market_type"], minute_start, kickoff, resolution),
            "phase_at_minute_end": phase(market["market_type"], minute_end, kickoff, resolution),
            "actual_kickoff_utc": context["actual_kickoff_utc"], "resolved_on_timestamp": market["resolved_on_timestamp"],
            "minutes_from_actual_kickoff": fmt(None if kickoff is None else (minute_end-kickoff)/60),
            "minutes_to_resolution": fmt(None if resolution is None else (resolution-minute_end)/60),
            "minute_contains_kickoff": int(kickoff is not None and minute_start <= kickoff < minute_end),
            "minute_contains_resolution": int(resolution is not None and minute_start <= resolution < minute_end),
            "contains_post_kickoff_trade": int(g["post_kickoff_trade_count"] > 0),
            "contains_post_resolution_trade": int(g["post_resolution_trade_count"] > 0),
            "pre_kickoff_trade_count": g["pre_kickoff_trade_count"], "post_kickoff_trade_count": g["post_kickoff_trade_count"],
            "pre_resolution_trade_count": g["pre_resolution_trade_count"], "post_resolution_trade_count": g["post_resolution_trade_count"],
            "yes_outcome_won": "" if yes_won is None else yes_won, "trade_count": g["trade_count"],
            "distinct_wallet_count": len(g["wallets"]), "gross_volume_usdc": fmt(g["gross"]),
            "net_signed_flow_usdc": fmt(g["net"]), "whale_trade_count": g["whale_trade_count"],
            "whale_distinct_wallet_count": len(g["whale_wallets"]), "whale_gross_volume_usdc": fmt(g["whale_gross"]),
            "whale_net_signed_flow_usdc": fmt(g["whale_net"]), "nonwhale_trade_count": g["nonwhale_trade_count"],
            "nonwhale_distinct_wallet_count": len(g["nonwhale_wallets"]), "nonwhale_gross_volume_usdc": fmt(g["nonwhale_gross"]),
            "nonwhale_net_signed_flow_usdc": fmt(g["nonwhale_net"]),
            "whale_flow_share": fmt(g["whale_gross"]/g["gross"] if g["gross"] else 0),
            "signed_log_whale_net_flow": fmt(signed_log(g["whale_net"])),
            "signed_log_nonwhale_net_flow": fmt(signed_log(g["nonwhale_net"])),
            "lagged_30m_gross_volume_usdc": fmt(rolling_sum),
            "lagged_30m_price_change": fmt(None if p0 is None or lag_p is None else p0-lag_p),
            "baseline_price_timestamp": "" if p_ts is None else p_ts, "baseline_price_utc": utc_iso(p_ts),
            "baseline_price_age_seconds": "" if p_ts is None else minute_end-p_ts, "yes_price_t": fmt(p0),
            "baseline_available": int(p0 is not None),
        }
        for tol in tolerances:
            row[f"baseline_within_{tol}m"] = int(p_ts is not None and minute_end-p_ts <= tol*60)
        flow_sign = 0 if g["net"] == 0 else (1 if g["net"] > 0 else -1)
        for h in horizons:
            target = minute_end + h*60; f_ts, future = future_unbounded(timestamps, prices, target)
            delta = None if p0 is None or future is None else future-p0
            row.update({
                f"target_timestamp_{h}m": target, f"target_timestamp_{h}m_utc": utc_iso(target),
                f"price_timestamp_{h}m": "" if f_ts is None else f_ts, f"price_timestamp_{h}m_utc": utc_iso(f_ts),
                f"price_lateness_{h}m_seconds": "" if f_ts is None else f_ts-target,
                f"yes_price_{h}m": fmt(future), f"delta_yes_price_{h}m": fmt(delta),
                f"directional_impact_{h}m": fmt(None if delta is None else flow_sign*delta),
                f"brier_improvement_{h}m": fmt(None if delta is None or yes_won is None else (yes_won-p0)**2-(yes_won-future)**2),
                f"target_price_available_{h}m": int(future is not None),
                f"crosses_kickoff_{h}m": int(kickoff is not None and minute_end <= kickoff < target),
                f"crosses_resolution_{h}m": int(resolution is not None and minute_end <= resolution < target),
            })
            for tol in tolerances:
                row[f"target_{h}m_within_{tol}m"] = int(f_ts is not None and f_ts-target <= tol*60)
        result.append(row); rolling.append((minute_start,g["gross"])); rolling_sum += g["gross"]
    return result


def load_progress(out: Path):
    path=out/"progress.json"
    return json.loads(path.read_text()) if path.exists() else {"version":2,"created_at":datetime.now(timezone.utc).isoformat(),"completed_markets":{},"failed":[]}


def save_progress(out: Path, progress: dict):
    progress["updated_at"]=datetime.now(timezone.utc).isoformat(); atomic_json(out/"progress.json",progress)


def write_gzip(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True,exist_ok=True); temp=path.with_suffix(path.suffix+".tmp")
    with gzip.open(temp,"wt",encoding="utf-8",newline="",compresslevel=6) as f:
        w=csv.DictWriter(f,fieldnames=fieldnames); w.writeheader(); w.writerows(rows)
    temp.replace(path)


def setup(out: Path):
    src=ANALYSIS_ROOT/"v1"/"whale_wallet_definitions.csv.gz"; dst=out/"whale_wallet_definitions.csv.gz"
    if not dst.exists(): shutil.copy2(src,dst)
    srcmeta=ANALYSIS_ROOT/"v1"/"whale_wallet_definitions.json"; dstmeta=out/"whale_wallet_definitions.json"
    if not dstmeta.exists(): shutil.copy2(srcmeta,dstmeta)


def process(out: Path, config: dict, market_ids: set[str]|None=None, max_markets:int|None=None):
    setup(out); whales=load_primary_whales(out); markets=load_markets(); trade_manifest,price_manifest=load_manifests(); progress=load_progress(out)
    selected=[x for x in trade_manifest if market_ids is None or x["market_id"] in market_ids]
    if max_markets is not None: selected=selected[:max_markets]
    all_fields=fields(config)
    for i,item in enumerate(selected,1):
        mid=item["market_id"]; target=out/"checkpoints"/f"{mid}.csv.gz"
        if mid in progress["completed_markets"] and target.exists(): print(f"[{i}/{len(selected)}] {mid} checkpointed",flush=True); continue
        try:
            market=markets[mid]; resolution=parse_utc(market["resolved_on_timestamp"]); kickoff=None
            if market["market_type"]=="match":
                with gzip.open(ROOT/item["path"],"rt",encoding="utf-8",newline="") as f:
                    first=next(csv.DictReader(f)); kickoff=parse_utc(first["actual_kickoff_utc"])
            groups,source_rows=aggregate(ROOT/item["path"],whales,kickoff,resolution)
            pitem=price_manifest[mid]; timestamps,prices,no_rows=load_yes_prices(ROOT/pitem["path"])
            rows=make_rows(groups,market,timestamps,prices,config); write_gzip(target,rows,all_fields)
            progress["completed_markets"][mid]={"market_type":market["market_type"],"source_trade_rows":source_rows,"market_minutes":len(rows),"yes_price_points":len(timestamps),"no_price_rows_ignored":no_rows,"sha256":sha256(target)}
            save_progress(out,progress); print(f"[{i}/{len(selected)}] {mid} {market['market_type']} rows={len(rows):,}",flush=True)
        except Exception as exc:
            progress["failed"].append({"market_id":mid,"error":repr(exc),"at":datetime.now(timezone.utc).isoformat()}); save_progress(out,progress); raise


def combine(out: Path, config: dict, require_all=True):
    progress=load_progress(out); markets=load_markets()
    if require_all and len(progress["completed_markets"])!=len(markets): raise RuntimeError(f"{len(progress['completed_markets'])}/{len(markets)} markets completed")
    target=out/"market_minute_all_horizons.csv.gz"; temp=target.with_suffix(target.suffix+".tmp"); fieldnames=fields(config)
    rows=trades=postres=0; phases={}; horizon_available={str(h):0 for h in config["horizon_minutes"]}
    with gzip.open(temp,"wt",encoding="utf-8",newline="",compresslevel=6) as outfh:
        writer=csv.DictWriter(outfh,fieldnames=fieldnames); writer.writeheader()
        for mid in sorted(progress["completed_markets"],key=int):
            with gzip.open(out/"checkpoints"/f"{mid}.csv.gz","rt",encoding="utf-8",newline="") as infh:
                for row in csv.DictReader(infh):
                    writer.writerow(row); rows+=1; trades+=int(row["trade_count"]); postres+=int(row["post_resolution_trade_count"])
                    phases[row["phase_at_minute_start"]]=phases.get(row["phase_at_minute_start"],0)+1
                    for h in config["horizon_minutes"]: horizon_available[str(h)]+=int(row[f"target_price_available_{h}m"])
    temp.replace(target); source_rows=sum(int(x["source_trade_rows"]) for x in progress["completed_markets"].values()); errors=[]
    if trades!=source_rows: errors.append(f"trade reconciliation source={source_rows} panel={trades}")
    manifest={"path":str(target.relative_to(ROOT)),"rows":rows,"trade_rows":trades,"post_resolution_trade_rows":postres,"sha256":sha256(target)}
    with (out/"file_manifest.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(manifest)); w.writeheader(); w.writerow(manifest)
    qc={"generated_at":datetime.now(timezone.utc).isoformat(),"version":"v2","markets_expected":len(markets),"markets_completed":len(progress["completed_markets"]),"source_trade_rows":source_rows,"panel_trade_rows":trades,"market_minute_rows":rows,"post_resolution_trade_rows_flagged":postres,"phase_rows":phases,"horizon_target_price_available":horizon_available,"file":manifest,"errors":errors}
    atomic_json(out/"analysis_ready_qc.json",qc)
    with (out/"sample_flow.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f); w.writerow(["phase_at_minute_start","market_minute_rows"]); w.writerows(sorted(phases.items()))
    if errors: raise RuntimeError(errors)
    config["status"]="built_qc_passed"; config["built_at"]=qc["generated_at"]; atomic_json(out/"config.json",config); print(json.dumps(qc,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--stage",choices=["markets","combine","all"],default="all"); ap.add_argument("--market-id",action="append"); ap.add_argument("--max-markets",type=int); ap.add_argument("--allow-partial-combine",action="store_true"); args=ap.parse_args()
    out=ANALYSIS_ROOT/"v2"; config=json.loads((out/"config.json").read_text())
    if args.stage in ("markets","all"): process(out,config,set(args.market_id) if args.market_id else None,args.max_markets)
    if args.stage in ("combine","all"): combine(out,config,not args.allow_partial_combine)
    return 0


if __name__=="__main__": raise SystemExit(main())
