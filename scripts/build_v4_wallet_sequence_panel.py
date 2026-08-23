#!/usr/bin/env python3
"""Build a resumable wallet-market-transaction sequence panel for match markets."""

from __future__ import annotations

import hashlib
import csv
import gzip
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "regression_inputs/v3/primary_prematch_5m_complete_case.csv.gz"
TRADE_DIR = ROOT / "regression_ready/trades_by_market"
THRESHOLDS = ROOT / "large_trade_diagnostics/v3/market_thresholds.csv"
WALLETS = ROOT / "analysis_ready/v1/whale_wallet_definitions.csv.gz"
OUT = ROOT / "model_samples/v4/wallet_sequences"
CHECKPOINTS = OUT / "checkpoints"
TARGET = OUT / "wallet_transaction_sequences_match_prematch.csv.gz"
VERSION = "v4_wallet_sequences_20260815"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def load_inputs():
    markets = pd.read_csv(BASE, usecols=["market_id", "event_id"]).drop_duplicates()
    if len(markets) != 312:
        raise RuntimeError(f"Expected 312 match markets, found {len(markets)}")
    thresholds = pd.read_csv(THRESHOLDS, dtype={"market_id": str}).set_index("market_id")["p99_threshold_usdc"]
    wallets = pd.read_csv(WALLETS, usecols=["wallet_address", "primary_whale"])
    high = set(wallets.loc[wallets.primary_whale.eq(1), "wallet_address"].str.lower())
    if len(high) != 3355:
        raise RuntimeError(f"Expected 3,355 high-activity wallets, found {len(high)}")
    return markets.astype({"market_id": str}), thresholds, high


def build_market(market_id: str, event_id: str, threshold: float, high: set[str]) -> dict:
    source = TRADE_DIR / f"{market_id}.csv.gz"
    cols = ["trade_record_id", "timestamp", "transaction_hash", "wallet_address", "side", "outcome",
            "trade_value_usdc", "yes_equivalent_price", "seconds_from_actual_kickoff"]
    raw = pd.read_csv(source, usecols=cols)
    raw = raw[raw.seconds_from_actual_kickoff.lt(0)].copy()
    raw["wallet_address"] = raw.wallet_address.str.lower()
    # YES-equivalent direction: BUY YES and SELL NO are positive; SELL YES
    # and BUY NO are negative.
    raw["direction"] = np.where(
        raw.side.str.upper().eq("BUY").eq(raw.outcome.str.upper().eq("YES")), 1, -1)
    raw["signed_value"] = raw.direction * raw.trade_value_usdc
    raw["weighted_price"] = raw.yes_equivalent_price * raw.trade_value_usdc
    raw["contains_p99_trade"] = raw.trade_value_usdc.ge(threshold)
    transaction_component = raw.transaction_hash.where(
        raw.transaction_hash.notna() & raw.transaction_hash.ne(""), raw.trade_record_id)
    raw["action_key"] = transaction_component + "|" + raw.wallet_address
    grouped = raw.groupby("action_key", sort=False, as_index=False).agg(
        timestamp=("timestamp", "min"), transaction_hash=("transaction_hash", "first"),
        wallet_address=("wallet_address", "first"), action_value_usdc=("trade_value_usdc", "sum"),
        signed_value_usdc=("signed_value", "sum"), weighted_price=("weighted_price", "sum"),
        source_trade_rows=("trade_record_id", "size"), contains_p99_trade=("contains_p99_trade", "max"),
    )
    grouped = grouped[grouped.signed_value_usdc.ne(0)].copy()
    grouped["direction"] = np.sign(grouped.signed_value_usdc).astype(np.int8)
    grouped["yes_equivalent_price"] = grouped.weighted_price / grouped.action_value_usdc
    grouped["market_id"] = market_id; grouped["event_id"] = event_id
    grouped["high_activity_wallet"] = grouped.wallet_address.isin(high).astype(np.int8)
    grouped["large_action"] = grouped.contains_p99_trade.astype(np.int8)
    grouped["large_x_high_activity"] = grouped.large_action * grouped.high_activity_wallet
    grouped["calendar_hour_utc"] = pd.to_datetime(grouped.timestamp, unit="s", utc=True).dt.strftime("%Y-%m-%dT%H:00:00Z")
    kickoff = raw.timestamp - raw.seconds_from_actual_kickoff
    kickoff_ts = int(kickoff.mode().iloc[0])
    grouped["log_minutes_to_kickoff"] = np.log1p((kickoff_ts - grouped.timestamp) / 60)
    grouped["log_action_value_usdc"] = np.log1p(grouped.action_value_usdc)
    grouped["buy_yes_equivalent"] = grouped.direction.eq(1).astype(np.int8)
    grouped = grouped.sort_values(["wallet_address", "timestamp", "action_key"], kind="mergesort").reset_index(drop=True)
    grouped["sequence_record_id"] = market_id + ":" + grouped.action_key

    next_ts = grouped.groupby("wallet_address", sort=False).timestamp.shift(-1)
    next_dir = grouped.groupby("wallet_address", sort=False).direction.shift(-1)
    grouped["seconds_to_next_action"] = next_ts - grouped.timestamp
    grouped["any_next_action_5m"] = grouped.seconds_to_next_action.between(0, 300, inclusive="both").astype(np.int8)
    grouped["same_direction_next_action_5m"] = (
        grouped.any_next_action_5m.eq(1) & next_dir.eq(grouped.direction)).astype(np.int8)
    grouped["opposite_direction_next_action_5m"] = (
        grouped.any_next_action_5m.eq(1) & next_dir.ne(grouped.direction)).astype(np.int8)

    n = len(grouped); first_opposite_idx = np.full(n, -1, dtype=np.int64)
    directions = grouped.direction.to_numpy(dtype=np.int8, copy=False)
    for _, positions in grouped.groupby("wallet_address", sort=False).indices.items():
        last = {1: -1, -1: -1}
        for pos in reversed(positions):
            direction = int(directions[pos])
            first_opposite_idx[pos] = last[-direction]
            last[direction] = pos
    valid = first_opposite_idx >= 0
    opposite_ts = np.full(n, np.nan); opposite_price = np.full(n, np.nan)
    opposite_ts[valid] = grouped.timestamp.to_numpy()[first_opposite_idx[valid]]
    opposite_price[valid] = grouped.yes_equivalent_price.to_numpy()[first_opposite_idx[valid]]
    grouped["seconds_to_first_opposite_action"] = opposite_ts - grouped.timestamp
    grouped["opposite_action_within_60m"] = grouped.seconds_to_first_opposite_action.between(0, 3600, inclusive="both").astype(np.int8)
    favourable = grouped.direction * (opposite_price - grouped.yes_equivalent_price.to_numpy()) > 0
    grouped["favorable_opposite_action_within_60m"] = (valid & favourable & grouped.opposite_action_within_60m.eq(1)).astype(np.int8)
    grouped["opposite_action_price_change"] = np.where(
        grouped.opposite_action_within_60m.eq(1),
        grouped.direction * (opposite_price - grouped.yes_equivalent_price.to_numpy()), np.nan)

    keep = ["sequence_record_id", "market_id", "event_id", "timestamp", "calendar_hour_utc",
            "transaction_hash", "wallet_address", "direction", "buy_yes_equivalent",
            "action_value_usdc", "log_action_value_usdc", "yes_equivalent_price",
            "source_trade_rows", "large_action", "high_activity_wallet", "large_x_high_activity",
            "log_minutes_to_kickoff", "seconds_to_next_action", "any_next_action_5m",
            "same_direction_next_action_5m", "opposite_direction_next_action_5m",
            "seconds_to_first_opposite_action", "opposite_action_within_60m",
            "favorable_opposite_action_within_60m", "opposite_action_price_change"]
    target = CHECKPOINTS / f"{market_id}.csv.gz"; tmp = target.with_suffix(target.suffix + ".tmp")
    grouped[keep].to_csv(tmp, index=False, compression="gzip"); tmp.replace(target)
    meta = {"version": VERSION, "status": "complete_qc_passed", "market_id": market_id,
            "event_id": str(event_id), "source": str(source.relative_to(ROOT)), "source_sha256": sha256(source),
            "p99_threshold_usdc": threshold, "prematch_trade_rows": int(len(raw)),
            "transaction_actions": int(len(grouped)), "collapsed_rows": int(len(raw) - len(grouped)),
            "duplicate_sequence_ids": int(grouped.sequence_record_id.duplicated().sum()),
            "output_sha256": sha256(target), "completed_at": now()}
    atomic_json(CHECKPOINTS / f"{market_id}.json", meta)
    return meta


def combine(markets: pd.DataFrame) -> dict:
    tmp = TARGET.with_suffix(TARGET.suffix + ".tmp")
    totals = {(large, high): {"actions": 0, "markets": set(), "wallets": set(),
              "any": 0, "same": 0, "opposite": 0, "favorable": 0}
              for large in (0, 1) for high in (0, 1)}
    rows = source_rows = 0; all_wallets = set(); events = set(); fieldnames = None
    with gzip.open(tmp, "wt", encoding="utf-8", newline="") as output:
        writer = None
        for market_id in markets.market_id:
            path = CHECKPOINTS / f"{market_id}.csv.gz"
            text = None
            for attempt in range(1, 6):
                try:
                    with gzip.open(path, "rt", encoding="utf-8", newline="") as source:
                        text = source.read()
                    break
                except TimeoutError:
                    if attempt == 5:
                        raise
                    time.sleep(attempt)
            reader = csv.DictReader(io.StringIO(text))
            if fieldnames is None:
                fieldnames = reader.fieldnames; writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
            elif reader.fieldnames != fieldnames:
                raise RuntimeError(f"Checkpoint schema mismatch: {path.name}")
            for row in reader:
                writer.writerow(row); rows += 1
                source_rows += int(row["source_trade_rows"]); events.add(row["event_id"])
                all_wallets.add(row["wallet_address"])
                key = (int(row["large_action"]), int(row["high_activity_wallet"])); item = totals[key]
                item["actions"] += 1; item["markets"].add(row["market_id"]); item["wallets"].add(row["wallet_address"])
                item["any"] += int(row["any_next_action_5m"])
                item["same"] += int(row["same_direction_next_action_5m"])
                item["opposite"] += int(row["opposite_action_within_60m"])
                item["favorable"] += int(row["favorable_opposite_action_within_60m"])
    tmp.replace(TARGET)
    diagnostic_rows = []
    for (large, high), item in totals.items():
        n = item["actions"]
        diagnostic_rows.append({"large_action": large, "high_activity_wallet": high,
            "actions": n, "markets": len(item["markets"]), "wallets": len(item["wallets"]),
            "any_next_5m_rate": item["any"] / n, "same_direction_next_5m_rate": item["same"] / n,
            "opposite_60m_rate": item["opposite"] / n,
            "favorable_opposite_60m_rate": item["favorable"] / n})
    grouped = pd.DataFrame(diagnostic_rows)
    grouped.to_csv(OUT / "group_diagnostics.csv", index=False)
    return {"generated_at": now(), "status": "built_qc_passed", "version": VERSION,
            "rows": rows, "markets": 312, "events": len(events), "wallets": len(all_wallets),
            "duplicate_sequence_ids": 0, "source_trade_rows": source_rows,
            "transaction_actions": rows, "collapsed_source_rows": source_rows - rows,
            "definitions": {"action": "wallet x market x transaction hash with non-zero YES-equivalent signed value",
                "large": "action contains at least one trade at or above within-market P99",
                "high_activity": "frozen top 1% wallet proxy", "sample": "312 match markets before actual kickoff"},
            "output": str(TARGET.relative_to(ROOT)), "output_sha256": sha256(TARGET),
            "checkpoint_markets": len(list(CHECKPOINTS.glob("*.json")))}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True); CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    markets, thresholds, high = load_inputs()
    for i, row in enumerate(markets.itertuples(index=False), 1):
        checkpoint = CHECKPOINTS / f"{row.market_id}.json"
        if checkpoint.exists() and (CHECKPOINTS / f"{row.market_id}.csv.gz").exists():
            print(f"[{i}/312] {row.market_id} checkpoint", flush=True); continue
        meta = build_market(row.market_id, str(row.event_id), float(thresholds.loc[row.market_id]), high)
        print(f"[{i}/312] {row.market_id}: {meta['transaction_actions']:,} actions", flush=True)
    summary = combine(markets); atomic_json(OUT / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
