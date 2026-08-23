#!/usr/bin/env python3
"""Build leakage-audited H3c machine-learning samples from the frozen H2c data."""
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "model_samples/v4/h2c/h2c_initial_positive_move.csv.gz"
OUT = ROOT / "model_samples/v4/rq3_ml"

ID_COLUMNS = [
    "initiating_trade_record_id", "market_id", "event_id", "market_type",
    "phase", "initiating_timestamp", "calendar_hour_utc",
]
BASELINE = ["baseline_yes_price", "initial_directional_move_5m"]
INCREMENTAL = [
    "log_initiating_trade_value", "initiating_minute_top1_wallet_share",
    "initiating_minute_top3_wallet_share", "initiating_minute_wallet_hhi",
    "initiating_minute_absolute_flow_imbalance", "follower_imbalance_5m",
    "follower_signed_log_net_5m",
]
TARGET = "full_reversal_30m"
PROHIBITED = [
    "directional_move_15m", "directional_move_30m", "directional_move_60m",
    "partial_reversal_15m", "partial_reversal_30m", "partial_reversal_60m",
    "full_reversal_15m", "full_reversal_60m", "reversal_fraction_15m",
    "reversal_fraction_30m", "reversal_fraction_60m",
    "initiating_wallet_opposite_trade_within_60m",
]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def write_gzip(frame, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", newline="", compresslevel=6) as stream:
        frame.to_csv(stream, index=False)
    temporary.replace(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(SRC)
    data["log_initiating_trade_value"] = np.log1p(data["initiating_trade_value_usdc"])
    columns = ID_COLUMNS + BASELINE + INCREMENTAL + [TARGET]
    samples = {
        "match_pre": data[(data.market_type == "match") & (data.phase == "pre")],
        "match_post": data[(data.market_type == "match") & (data.phase == "post")],
        "outright": data[data.market_type == "outright"],
    }
    outputs, qc = {}, {}
    for name, sample in samples.items():
        before = len(sample)
        sample = sample[columns].dropna(subset=BASELINE + INCREMENTAL + [TARGET]).copy()
        sample[TARGET] = sample[TARGET].astype("int8")
        path = OUT / f"rq3_{name}_full_reversal_30m.csv.gz"
        write_gzip(sample, path)
        group = "event_id" if name.startswith("match") else "market_id"
        qc[name] = {
            "rows_before_complete_case_filter": before,
            "rows": len(sample),
            "rows_excluded_missing": before - len(sample),
            "positive": int(sample[TARGET].sum()),
            "negative": int((sample[TARGET] == 0).sum()),
            "positive_share": float(sample[TARGET].mean()) if len(sample) else None,
            "markets": int(sample.market_id.nunique()),
            "groups": int(sample[group].nunique()),
            "duplicate_record_ids": int(sample.initiating_trade_record_id.duplicated().sum()),
            "group_variable": group,
        }
        outputs[name] = {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}

    leakage = sorted(set(PROHIBITED).intersection(columns))
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "built_qc_passed" if not leakage else "failed_leakage_check",
        "source": str(SRC.relative_to(ROOT)),
        "source_sha256": sha256(SRC),
        "prediction_time": "initiating_timestamp + 5 minutes",
        "target": TARGET,
        "baseline_features": BASELINE,
        "incremental_features": INCREMENTAL,
        "prohibited_future_fields": PROHIBITED,
        "prohibited_fields_found_in_output": leakage,
        "samples": qc,
        "outputs": outputs,
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if leakage:
        raise SystemExit("Future-information leakage detected")


if __name__ == "__main__":
    main()
