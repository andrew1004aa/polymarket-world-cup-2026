#!/usr/bin/env python3
"""Build 7d/14d past-only Large x wallet market-minute price panels."""

from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "regression_inputs/v3/primary_prematch_5m_complete_case.csv.gz"
CLASSIFICATIONS = ROOT / "regression_results/v4/wallet_sequences_past_only/past_only_wallet_classifications.csv.gz"
CHECKPOINTS = ROOT / "model_samples/v4/wallet_sequences/checkpoints"
OUT = ROOT / "model_samples/v4/past_only_price_2x2"
SPECS = {"baseline_7d": 1780876800, "baseline_14d": 1781481600}
GROUPS = ("large_high", "large_other", "ordinary_high", "ordinary_other")


def now(): return datetime.now(timezone.utc).isoformat()


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1024 * 1024): h.update(block)
    return h.hexdigest()


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
    classifications = read_csv_retry(CLASSIFICATIONS,
        usecols=["specification", "wallet_address", "past_only_high_activity"])
    high_sets = {name: set(classifications.loc[
        classifications.specification.eq(name) & classifications.past_only_high_activity.eq(1),
        "wallet_address"]) for name in SPECS}
    samples = {name: base[base.minute_start_timestamp.ge(cutoff)].copy() for name, cutoff in SPECS.items()}
    required = {name: {m: set(x.minute_start_timestamp.astype(int)) for m, x in sample.groupby("market_id")}
                for name, sample in samples.items()}
    aggregates = {name: [] for name in SPECS}
    markets = sorted(base.market_id.unique())
    action_cols = ["market_id", "timestamp", "wallet_address", "direction", "action_value_usdc", "large_action"]
    for i, market_id in enumerate(markets, 1):
        actions = read_csv_retry(CHECKPOINTS / f"{market_id}.csv.gz", usecols=action_cols,
                                 dtype={"market_id": str})
        actions["minute_start_timestamp"] = (actions.timestamp // 60) * 60
        for name, cutoff in SPECS.items():
            minutes = required[name].get(market_id)
            if not minutes: continue
            part = actions[actions.timestamp.ge(cutoff) & actions.minute_start_timestamp.isin(minutes)].copy()
            if part.empty: continue
            high = part.wallet_address.isin(high_sets[name])
            part["group"] = np.select(
                [part.large_action.eq(1) & high, part.large_action.eq(1) & ~high,
                 part.large_action.eq(0) & high],
                ["large_high", "large_other", "ordinary_high"], default="ordinary_other")
            part["signed_value"] = part.direction * part.action_value_usdc
            agg = part.groupby(["market_id", "minute_start_timestamp", "group"], as_index=False).agg(
                trade_count=("timestamp", "size"), gross_volume_usdc=("action_value_usdc", "sum"),
                net_signed_flow_usdc=("signed_value", "sum"))
            aggregates[name].append(agg)
        if i % 50 == 0: print(f"[{i}/312] markets", flush=True)
    summaries = []
    for name, sample in samples.items():
        long = pd.concat(aggregates[name], ignore_index=True)
        pieces = []
        for measure in ("trade_count", "gross_volume_usdc", "net_signed_flow_usdc"):
            wide = long.pivot_table(index=["market_id", "minute_start_timestamp"], columns="group",
                                    values=measure, aggfunc="sum", fill_value=0)
            wide.columns = [f"{group}_{measure}" for group in wide.columns]
            pieces.append(wide)
        features = pd.concat(pieces, axis=1).reset_index()
        output = sample.merge(features, on=["market_id", "minute_start_timestamp"], how="left", validate="one_to_one")
        for group in GROUPS:
            for measure in ("trade_count", "gross_volume_usdc", "net_signed_flow_usdc"):
                col = f"{group}_{measure}"
                if col not in output: output[col] = 0
                output[col] = output[col].fillna(0)
            net = output[f"{group}_net_signed_flow_usdc"]
            output[f"{group}_signed_log_net_flow"] = np.sign(net) * np.log1p(np.abs(net))
            output[f"{group}_log_gross_volume"] = np.log1p(output[f"{group}_gross_volume_usdc"])
        if output.model_sample_record_id.duplicated().any() or output.isna().any().any():
            raise RuntimeError(f"{name} failed uniqueness/missing-value QC")
        target = OUT / f"{name}_price_2x2.csv.gz"; tmp = target.with_suffix(target.suffix + ".tmp")
        output.to_csv(tmp, index=False, compression="gzip"); tmp.replace(target)
        summary = {"specification": name, "cutoff_timestamp": SPECS[name],
            "cutoff_utc": datetime.fromtimestamp(SPECS[name], timezone.utc).isoformat(),
            "rows": int(len(output)), "markets": int(output.market_id.nunique()),
            "events": int(output.event_id.nunique()), "calendar_hours": int(output.calendar_hour_utc.nunique()),
            "price_update_rate": float(output.delta_yes_price.ne(0).mean()),
            "high_activity_wallets": len(high_sets[name]), "duplicate_record_ids": 0,
            "missing_values": 0, "output": str(target.relative_to(ROOT)), "output_sha256": sha256(target)}
        summaries.append(summary)
    payload = {"generated_at": now(), "status": "built_qc_passed", "base_sha256": sha256(BASE),
               "classification_sha256": sha256(CLASSIFICATIONS), "specifications": summaries}
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())
