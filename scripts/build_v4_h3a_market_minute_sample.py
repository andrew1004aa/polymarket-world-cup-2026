#!/usr/bin/env python3
"""Build resumable wallet-concentration features for the frozen H3a minute sample."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "regression_inputs/v3/primary_prematch_5m_complete_case.csv.gz"
MANIFEST = ROOT / "regression_ready/trade_files.csv"
OUT = ROOT / "model_samples/v4/h3a_market_minute"
CHECKPOINTS = OUT / "checkpoints"
TARGET = OUT / "h3a_match_pre_5m.csv.gz"
KEEP = [
    "model_sample_record_id", "market_id", "event_id", "minute_start_timestamp",
    "calendar_hour_utc", "trade_count", "distinct_wallet_count", "gross_volume_usdc",
    "net_signed_flow_usdc", "yes_price_t", "lagged_30m_price_change",
    "log_lagged_30m_volume", "log_minutes_to_kickoff", "delta_yes_price",
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def features(path, required_minutes):
    trades = pd.read_csv(path, usecols=["timestamp", "wallet_address", "trade_value_usdc"])
    trades["minute_start_timestamp"] = (trades.timestamp.astype("int64") // 60) * 60
    trades = trades[trades.minute_start_timestamp.isin(required_minutes)]
    wallet = trades.groupby(["minute_start_timestamp", "wallet_address"], sort=False,
                            observed=True).trade_value_usdc.sum().rename("wallet_volume").reset_index()
    totals = wallet.groupby("minute_start_timestamp", sort=False).wallet_volume.agg(
        wallet_gross_volume_usdc="sum", wallet_volume_squared_sum=lambda x: np.square(x).sum(),
        concentration_distinct_wallets="size")
    top = wallet.sort_values(["minute_start_timestamp", "wallet_volume"], ascending=[True, False])
    top1 = top.groupby("minute_start_timestamp", sort=False).wallet_volume.first().rename("top1")
    top3 = top.groupby("minute_start_timestamp", sort=False).head(3).groupby(
        "minute_start_timestamp", sort=False).wallet_volume.sum().rename("top3")
    result = totals.join([top1, top3]).reset_index()
    result["wallet_top1_volume_share"] = result.top1 / result.wallet_gross_volume_usdc
    result["wallet_top3_volume_share"] = result.top3 / result.wallet_gross_volume_usdc
    result["wallet_volume_hhi"] = (result.wallet_volume_squared_sum /
                                   np.square(result.wallet_gross_volume_usdc))
    return result.drop(columns=["top1", "top3", "wallet_volume_squared_sum"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(BASE, usecols=KEEP)
    manifest = pd.read_csv(MANIFEST)
    path_by_market = {int(row.market_id): ROOT / row.path for row in manifest.itertuples()}
    grouped = list(base.groupby("market_id", sort=True))
    outputs = []
    for order, (market_id, market_rows) in enumerate(grouped, 1):
        market_id = int(market_id)
        target = CHECKPOINTS / f"{market_id}.csv.gz"
        meta_path = CHECKPOINTS / f"{market_id}.json"
        source = path_by_market[market_id]
        source_hash = sha256(source)
        reusable = False
        if target.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            reusable = (meta.get("source_sha256") == source_hash and
                        meta.get("rows") == len(market_rows) and
                        meta.get("output_sha256") == sha256(target))
        if not reusable:
            concentration = features(source, set(market_rows.minute_start_timestamp.astype(int)))
            merged = market_rows.merge(concentration, on="minute_start_timestamp", how="left",
                                       validate="one_to_one")
            if merged.wallet_volume_hhi.isna().any():
                raise RuntimeError(f"Missing concentration features in market {market_id}")
            if not np.allclose(merged.gross_volume_usdc, merged.wallet_gross_volume_usdc,
                               rtol=1e-9, atol=1e-6):
                raise RuntimeError(f"Gross-volume reconciliation failed in market {market_id}")
            if not np.array_equal(merged.distinct_wallet_count.astype(int),
                                  merged.concentration_distinct_wallets.astype(int)):
                raise RuntimeError(f"Wallet-count reconciliation failed in market {market_id}")
            merged["absolute_price_change_5m"] = merged.delta_yes_price.abs()
            merged["log_gross_volume_usdc"] = np.log1p(merged.gross_volume_usdc)
            merged["log_trade_count"] = np.log1p(merged.trade_count)
            merged["log_distinct_wallet_count"] = np.log1p(merged.distinct_wallet_count)
            merged["absolute_flow_imbalance"] = (merged.net_signed_flow_usdc.abs() /
                                                  merged.gross_volume_usdc)
            temporary = target.with_suffix(target.suffix + ".tmp")
            merged.to_csv(temporary, index=False, compression="gzip")
            temporary.replace(target)
            meta = {"market_id": market_id, "rows": len(merged),
                    "source": str(source.relative_to(ROOT)), "source_sha256": source_hash,
                    "output_sha256": sha256(target), "completed_at": datetime.now(timezone.utc).isoformat()}
            meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        outputs.append(target)
        print(f"[{order}/{len(grouped)}] {market_id}: {len(market_rows):,} minutes", flush=True)
    frames = [pd.read_csv(path) for path in outputs]
    combined = pd.concat(frames, ignore_index=True)
    if len(combined) != len(base) or combined.model_sample_record_id.duplicated().any():
        raise RuntimeError("Combined-row QC failed")
    temporary = TARGET.with_suffix(TARGET.suffix + ".tmp")
    combined.to_csv(temporary, index=False, compression="gzip")
    temporary.replace(TARGET)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "status": "built_qc_passed",
        "source": str(BASE.relative_to(ROOT)), "source_sha256": sha256(BASE),
        "markets": int(combined.market_id.nunique()), "events": int(combined.event_id.nunique()),
        "rows": len(combined), "duplicate_record_ids": int(combined.model_sample_record_id.duplicated().sum()),
        "missing_concentration": int(combined.wallet_volume_hhi.isna().sum()),
        "output": str(TARGET.relative_to(ROOT)), "output_sha256": sha256(TARGET),
        "concentration_definition": "wallet gross-USDC volume shares within each market-minute",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
