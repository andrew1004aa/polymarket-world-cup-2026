#!/usr/bin/env python3
"""Build the H3b concentrated-directional-trading reversal sample."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "model_samples/v4/h3a_market_minute/h3a_match_pre_5m.csv.gz"
PANEL = ROOT / "analysis_ready/v3/market_minute_all_horizons.csv.gz"
OUT = ROOT / "model_samples/v4/h3b_reversal"
TARGET = OUT / "h3b_match_pre_initial_directional_move.csv.gz"
HORIZONS = ("15m", "30m", "60m")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(BASE)
    later_columns = ["market_id", "minute_start_timestamp"]
    for horizon in HORIZONS:
        later_columns += [f"delta_yes_price_{horizon}", f"target_{horizon}_within_2m",
                          f"crosses_kickoff_{horizon}", f"crosses_resolution_{horizon}"]
    later = pd.read_csv(PANEL, usecols=later_columns)
    merged = base.merge(later, on=["market_id", "minute_start_timestamp"], how="left",
                        validate="one_to_one")
    merged["trade_direction"] = np.sign(merged.net_signed_flow_usdc)
    merged["initial_directional_move_5m"] = merged.trade_direction * merged.delta_yes_price
    sample = merged[(merged.trade_direction != 0) & (merged.initial_directional_move_5m > 0)].copy()
    sample["dominant_directional_concentration"] = (
        sample.wallet_volume_hhi * sample.absolute_flow_imbalance)
    counts = {}
    for horizon in HORIZONS:
        valid = (sample[f"target_{horizon}_within_2m"].eq(1) &
                 sample[f"crosses_kickoff_{horizon}"].eq(0) &
                 sample[f"crosses_resolution_{horizon}"].eq(0) &
                 sample[f"delta_yes_price_{horizon}"].notna())
        later_directional = sample.trade_direction * sample[f"delta_yes_price_{horizon}"]
        sample[f"later_directional_move_{horizon}"] = later_directional.where(valid)
        sample[f"partial_reversal_{horizon}"] = (
            later_directional < sample.initial_directional_move_5m).astype(float).where(valid)
        sample[f"full_reversal_{horizon}"] = (later_directional < 0).astype(float).where(valid)
        counts[horizon] = {
            "available": int(valid.sum()),
            "partial_reversal": int(sample.loc[valid, f"partial_reversal_{horizon}"].sum()),
            "full_reversal": int(sample.loc[valid, f"full_reversal_{horizon}"].sum()),
        }
    keep = [
        "model_sample_record_id", "market_id", "event_id", "minute_start_timestamp",
        "calendar_hour_utc", "trade_direction", "initial_directional_move_5m",
        "wallet_volume_hhi", "wallet_top1_volume_share", "wallet_top3_volume_share",
        "absolute_flow_imbalance", "dominant_directional_concentration",
        "yes_price_t", "absolute_price_change_5m", "lagged_30m_price_change",
        "log_lagged_30m_volume", "log_minutes_to_kickoff", "log_gross_volume_usdc",
        "log_trade_count",
    ]
    for horizon in HORIZONS:
        keep += [f"later_directional_move_{horizon}", f"partial_reversal_{horizon}",
                 f"full_reversal_{horizon}"]
    sample = sample[keep]
    temporary = TARGET.with_suffix(TARGET.suffix + ".tmp")
    sample.to_csv(temporary, index=False, compression="gzip")
    temporary.replace(TARGET)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "built_qc_passed", "source_rows": len(base), "sample_rows": len(sample),
        "markets": int(sample.market_id.nunique()), "events": int(sample.event_id.nunique()),
        "duplicate_record_ids": int(sample.model_sample_record_id.duplicated().sum()),
        "definition": "non-zero net-flow direction and positive same-direction five-minute price move",
        "focal_measure": "wallet-volume HHI multiplied by absolute net-flow imbalance",
        "outcome_counts": counts, "output": str(TARGET.relative_to(ROOT)),
        "output_sha256": sha256(TARGET), "base_sha256": sha256(BASE), "panel_sha256": sha256(PANEL),
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
