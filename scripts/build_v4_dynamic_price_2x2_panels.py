#!/usr/bin/env python3
"""Build rolling, expanding, and descriptive ex-post 2x2 price panels."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "regression_inputs/v3/primary_prematch_5m_complete_case.csv.gz"
CLASSES = ROOT / "regression_results/v4/wallet_sequences_dynamic/daily_high_activity_wallets.csv.gz"
CHECKPOINTS = ROOT / "model_samples/v4/wallet_sequences/checkpoints"
OUT = ROOT / "model_samples/v4/dynamic_price_2x2"
START = int(datetime(2026, 6, 15, tzinfo=timezone.utc).timestamp())
END = int(datetime(2026, 7, 23, tzinfo=timezone.utc).timestamp())
SPECS = ("rolling_7d", "expanding", "ex_post_descriptive")
GROUPS = ("large_high", "large_other", "ordinary_high", "ordinary_other")


def now(): return datetime.now(timezone.utc).isoformat()


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024): digest.update(block)
    return digest.hexdigest()


def read_csv_retry(path: Path, **kwargs):
    for attempt in range(1, 6):
        try: return pd.read_csv(path, **kwargs)
        except TimeoutError:
            if attempt == 5: raise
            time.sleep(attempt)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    base_cols = ["model_sample_record_id", "market_id", "event_id", "minute_start_timestamp",
                 "calendar_hour_utc", "delta_yes_price", "yes_price_t", "lagged_30m_price_change",
                 "log_lagged_30m_volume", "log_minutes_to_kickoff"]
    base = read_csv_retry(BASE, usecols=base_cols, dtype={"market_id": str})
    base = base[base.minute_start_timestamp.ge(START) & base.minute_start_timestamp.lt(END)].copy()
    classes = read_csv_retry(CLASSES, dtype={"wallet_address": str})
    high = {(specification, day): set(part.wallet_address)
            for (specification, day), part in classes.groupby(["specification", "evaluation_day"])}
    required = {market: set(part.minute_start_timestamp.astype(int)) for market, part in base.groupby("market_id")}
    aggregates = {specification: [] for specification in SPECS}
    action_cols = ["market_id", "timestamp", "wallet_address", "direction", "action_value_usdc", "large_action"]
    for index, market_id in enumerate(sorted(base.market_id.unique()), 1):
        actions = read_csv_retry(CHECKPOINTS / f"{market_id}.csv.gz", usecols=action_cols,
                                 dtype={"market_id": str})
        actions = actions[actions.timestamp.ge(START) & actions.timestamp.lt(END)].copy()
        actions["minute_start_timestamp"] = (actions.timestamp // 60) * 60
        actions = actions[actions.minute_start_timestamp.isin(required[market_id])]
        actions["evaluation_day"] = pd.to_datetime(actions.timestamp, unit="s", utc=True).dt.strftime("%Y-%m-%d")
        if actions.empty: continue
        for specification in SPECS:
            is_high = np.fromiter((wallet in high.get((specification, day), set())
                                   for wallet, day in zip(actions.wallet_address, actions.evaluation_day)),
                                  dtype=bool, count=len(actions))
            part = actions.copy()
            part["group"] = np.select(
                [part.large_action.eq(1) & is_high, part.large_action.eq(1) & ~is_high,
                 part.large_action.eq(0) & is_high],
                ["large_high", "large_other", "ordinary_high"], default="ordinary_other")
            part["signed_value"] = part.direction * part.action_value_usdc
            aggregate = part.groupby(["market_id", "minute_start_timestamp", "group"], as_index=False).agg(
                trade_count=("timestamp", "size"), gross_volume_usdc=("action_value_usdc", "sum"),
                net_signed_flow_usdc=("signed_value", "sum"))
            aggregates[specification].append(aggregate)
        if index % 40 == 0: print(f"[{index}/{base.market_id.nunique()}] markets", flush=True)

    summaries = []
    for specification in SPECS:
        long = pd.concat(aggregates[specification], ignore_index=True)
        pieces = []
        for measure in ("trade_count", "gross_volume_usdc", "net_signed_flow_usdc"):
            wide = long.pivot_table(index=["market_id", "minute_start_timestamp"], columns="group",
                                    values=measure, aggfunc="sum", fill_value=0)
            wide.columns = [f"{group}_{measure}" for group in wide.columns]
            pieces.append(wide)
        features = pd.concat(pieces, axis=1).reset_index()
        output = base.merge(features, on=["market_id", "minute_start_timestamp"], how="left", validate="one_to_one")
        for group in GROUPS:
            for measure in ("trade_count", "gross_volume_usdc", "net_signed_flow_usdc"):
                column = f"{group}_{measure}"
                if column not in output: output[column] = 0
                output[column] = output[column].fillna(0)
            net = output[f"{group}_net_signed_flow_usdc"]
            output[f"{group}_signed_log_net_flow"] = np.sign(net) * np.log1p(np.abs(net))
            output[f"{group}_log_gross_volume"] = np.log1p(output[f"{group}_gross_volume_usdc"])
        if output.model_sample_record_id.duplicated().any() or output.isna().any().any():
            raise RuntimeError(f"{specification} failed uniqueness/missing QC")
        target = OUT / f"{specification}_price_2x2.csv.gz"
        temporary = target.with_suffix(target.suffix + ".tmp")
        output.to_csv(temporary, index=False, compression="gzip"); temporary.replace(target)
        summaries.append({"specification": specification, "rows": int(len(output)),
            "markets": int(output.market_id.nunique()), "events": int(output.event_id.nunique()),
            "calendar_hours": int(output.calendar_hour_utc.nunique()),
            "price_update_rate": float(output.delta_yes_price.ne(0).mean()),
            "duplicate_record_ids": 0, "missing_values": 0,
            "output": str(target.relative_to(ROOT)), "output_sha256": sha256(target)})
    payload = {"generated_at": now(), "status": "built_qc_passed",
               "evaluation_start": "2026-06-15", "evaluation_end_inclusive": "2026-07-22",
               "base_sha256": sha256(BASE), "classification_sha256": sha256(CLASSES),
               "specifications": summaries}
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())
