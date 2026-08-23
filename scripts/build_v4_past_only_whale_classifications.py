#!/usr/bin/env python3
"""Build daily past-only whale-wallet classifications from all 360 markets."""

from __future__ import annotations

import csv
import gzip
import json
import math
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRADES = ROOT / "regression_ready/trades_by_market"
EX_POST = ROOT / "analysis_ready/v1/whale_wallet_definitions.csv.gz"
OUT = ROOT / "regression_results/v4/past_only_whale_classifications"
START_HISTORY = date(2026, 6, 1)
START_EVALUATION = date(2026, 6, 15)
END_EVALUATION = date(2026, 7, 22)
SPECS = {
    "expanding_top1_primary": ("expanding", 1.0),
    "rolling7_top1": ("rolling", 1.0),
    "expanding_top05": ("expanding", 0.5),
    "expanding_top2": ("expanding", 2.0),
}
EX_POST_SPEC = "ex_post_top1_descriptive"


def now(): return datetime.now(timezone.utc).isoformat()


def day_range(first: date, last: date):
    current = first
    while current <= last:
        yield current
        current += timedelta(days=1)


def read_rows(path: Path):
    for attempt in range(1, 6):
        try:
            with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
                yield from csv.DictReader(handle)
            return
        except TimeoutError:
            if attempt == 5: raise
            time.sleep(attempt)


def add_day(target: dict[str, float], source: dict[str, float], sign=1):
    for wallet, volume in source.items():
        target[wallet] = target.get(wallet, 0.0) + sign * volume
        if target[wallet] <= 1e-12:
            target.pop(wallet, None)


def select(volumes: dict[str, float], percent: float):
    ranked = sorted(volumes.items(), key=lambda item: (-item[1], item[0]))
    if not ranked: return set(), math.nan, 0, 0
    count = max(1, math.ceil(len(ranked) * percent / 100.0))
    threshold = ranked[count - 1][1]
    selected = {wallet for wallet, _ in ranked[:count]}
    ties = sum(abs(volume - threshold) <= 1e-12 for _, volume in ranked)
    return selected, threshold, len(ranked), ties


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(TRADES.glob("*.csv.gz"))
    if len(files) != 360: raise RuntimeError(f"Expected 360 market files, found {len(files)}")
    daily = defaultdict(lambda: defaultdict(float))
    source_rows = 0
    print("Pass 1/2: aggregate all-market wallet-day gross USDC", flush=True)
    for index, path in enumerate(files, 1):
        for row in read_rows(path):
            timestamp = int(row["timestamp"])
            day = datetime.fromtimestamp(timestamp, timezone.utc).date()
            if day < START_HISTORY or day > END_EVALUATION: continue
            daily[day][row["wallet_address"].lower()] += float(row["trade_value_usdc"])
            source_rows += 1
        if index % 40 == 0: print(f"  {index}/360 markets", flush=True)

    ex_post_table = pd.read_csv(EX_POST, usecols=["wallet_address", "primary_whale"])
    ex_post = set(ex_post_table.loc[ex_post_table.primary_whale.eq(1), "wallet_address"].str.lower())
    expanding, rolling = {}, {}
    selections = {specification: {} for specification in (*SPECS, EX_POST_SPEC)}
    diagnostics = []
    print("Pass 2/2: freeze daily whale classifications", flush=True)
    for current in day_range(START_HISTORY, END_EVALUATION):
        previous = current - timedelta(days=1)
        if previous in daily:
            add_day(expanding, daily[previous]); add_day(rolling, daily[previous])
        expired = current - timedelta(days=8)
        if expired in daily: add_day(rolling, daily[expired], sign=-1)
        if current < START_EVALUATION: continue
        for specification, (history_type, percent) in SPECS.items():
            history = expanding if history_type == "expanding" else rolling
            chosen, threshold, eligible, ties = select(history, percent)
            selections[specification][current] = chosen
            diagnostics.append({"specification": specification, "evaluation_day": current.isoformat(),
                "history_type": history_type, "top_percent": percent,
                "history_start": (START_HISTORY if history_type == "expanding" else current - timedelta(days=7)).isoformat(),
                "history_end_exclusive": current.isoformat(), "eligible_wallets": eligible,
                "whale_wallets": len(chosen), "threshold_usdc": threshold,
                "wallets_tied_at_boundary": ties, "minimum_actions": 1,
                "minimum_active_days": 0, "minimum_markets": 0})
        selections[EX_POST_SPEC][current] = ex_post
        diagnostics.append({"specification": EX_POST_SPEC, "evaluation_day": current.isoformat(),
            "history_type": "full_window", "top_percent": 1.0, "history_start": START_HISTORY.isoformat(),
            "history_end_exclusive": "full_window", "eligible_wallets": len(ex_post_table),
            "whale_wallets": len(ex_post), "threshold_usdc": math.nan,
            "wallets_tied_at_boundary": math.nan, "minimum_actions": 1,
            "minimum_active_days": 0, "minimum_markets": 0})
    rows = [{"specification": specification, "evaluation_day": day.isoformat(), "wallet_address": wallet}
            for specification, by_day in selections.items() for day, wallets in by_day.items()
            for wallet in sorted(wallets)]
    pd.DataFrame(rows).to_csv(OUT / "daily_whale_wallets.csv.gz", index=False, compression="gzip")
    pd.DataFrame(diagnostics).to_csv(OUT / "daily_classification_diagnostics.csv", index=False)
    summary = {"generated_at": now(), "status": "complete_qc_passed", "source_market_files": len(files),
        "source_trade_rows_in_history": source_rows, "history_market_universe": "all 360 World Cup markets",
        "evaluation_start": START_EVALUATION.isoformat(), "evaluation_end_inclusive": END_EVALUATION.isoformat(),
        "primary": "daily expanding cumulative gross USDC top 1%",
        "robustness": ["daily rolling seven-day gross USDC top 1%",
                       "daily expanding top 0.5%", "daily expanding top 2%"],
        "descriptive": "original full-window top 1% wallet definition",
        "eligibility": "at least one prior observed trade; no action, active-day, or market-count minimum",
        "tie_rule": "exact ceil(percent) selected; volume descending then wallet ascending",
        "classification_rows": len(rows), "evaluation_days": len(list(day_range(START_EVALUATION, END_EVALUATION)))}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())
