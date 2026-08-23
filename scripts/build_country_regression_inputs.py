#!/usr/bin/env python3
"""Build frozen complete-case country outright regression inputs."""

from __future__ import annotations

import hashlib
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "model_samples/v1"
DEFAULT_OUT = ROOT / "regression_inputs/country_v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def load(horizon: int, source: Path, version: str) -> pd.DataFrame:
    path = source / f"outright_{horizon}m.csv.gz"
    columns = [
        "model_sample_record_id", "source_analysis_record_id", "market_id",
        "condition_id", "country", "question", "minute_start_timestamp",
        "minute_start_utc", "date_utc", "calendar_hour_utc", "delta_yes_price",
        "net_signed_flow_usdc", "signed_log_whale_net_flow",
        "signed_log_nonwhale_net_flow", "yes_price_t",
        "lagged_30m_price_change", "lagged_30m_gross_volume_usdc",
        "crosses_resolution", "resolved_on_timestamp",
    ]
    if version == "v3":
        columns = [x for x in columns if "whale" not in x]
        columns += [
            "p99_signed_log_large_net_flow", "p99_signed_log_ordinary_net_flow",
            "p95_signed_log_large_net_flow", "p95_signed_log_ordinary_net_flow",
            "sparse_p99_market",
        ]
    data = pd.read_csv(path, usecols=columns)
    data["signed_log_total_net_flow"] = np.sign(data["net_signed_flow_usdc"]) * np.log1p(
        np.abs(data["net_signed_flow_usdc"])
    )
    data["log_lagged_30m_volume"] = np.log1p(data["lagged_30m_gross_volume_usdc"])
    data["horizon_minutes"] = horizon
    return data


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--version",default="country_v1")
    parser.add_argument("--source-dir",default=None); parser.add_argument("--output-dir",default=None)
    args=parser.parse_args(); source=ROOT/args.source_dir if args.source_dir else DEFAULT_SOURCE
    out=ROOT/args.output_dir if args.output_dir else DEFAULT_OUT; out.mkdir(parents=True, exist_ok=True)
    schema_version="v3" if args.version.startswith("v3") else "v1"
    five, thirty = load(5,source,schema_version), load(30,source,schema_version)
    required = [
        "delta_yes_price", "signed_log_total_net_flow",
        "yes_price_t", "lagged_30m_price_change", "log_lagged_30m_volume",
        "market_id", "calendar_hour_utc", "source_analysis_record_id",
    ]
    if schema_version=="v3": required += ["p99_signed_log_large_net_flow","p99_signed_log_ordinary_net_flow",
        "p95_signed_log_large_net_flow","p95_signed_log_ordinary_net_flow","sparse_p99_market"]
    else: required += ["signed_log_whale_net_flow","signed_log_nonwhale_net_flow"]
    if five["model_sample_record_id"].duplicated().any() or thirty["model_sample_record_id"].duplicated().any():
        raise RuntimeError("Duplicate model-sample identifiers")
    if five["crosses_resolution"].any() or thirty["crosses_resolution"].any():
        raise RuntimeError("A source row crosses resolution")

    five_complete = five.dropna(subset=required).copy()
    thirty_complete = thirty.dropna(subset=required).copy()
    common = set(five_complete.source_analysis_record_id) & set(thirty_complete.source_analysis_record_id)
    five_matched = five_complete[five_complete.source_analysis_record_id.isin(common)].copy()
    thirty_matched = thirty_complete[thirty_complete.source_analysis_record_id.isin(common)].copy()
    for frame in [five_complete, five_matched, thirty_matched]:
        frame.sort_values(["market_id", "minute_start_timestamp"], inplace=True)

    outputs = {
        "outright_5m_complete_case.csv.gz": five_complete,
        "outright_5m_matched_5_30.csv.gz": five_matched,
        "outright_30m_matched_5_30.csv.gz": thirty_matched,
    }
    manifest = []
    for name, frame in outputs.items():
        path = out / name
        tmp = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(tmp, index=False, compression="gzip")
        tmp.replace(path)
        manifest.append({
            "path": str(path.relative_to(ROOT)), "rows": len(frame),
            "markets": int(frame.market_id.nunique()),
            "calendar_hours": int(frame.calendar_hour_utc.nunique()),
            "zero_outcome_share": float((frame.delta_yes_price == 0).mean()),
            "sha256": sha256(path), "bytes": path.stat().st_size,
        })
    config = {
        "version": args.version, "status": "built_qc_passed",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source_5m": str((source / "outright_5m.csv.gz").relative_to(ROOT)),
        "source_30m": str((source / "outright_30m.csv.gz").relative_to(ROOT)),
        "source_5m_sha256": sha256(source / "outright_5m.csv.gz"),
        "source_30m_sha256": sha256(source / "outright_30m.csv.gz"),
        "complete_case_rule": required,
        "files": manifest,
    }
    write_json(out / "config.json", config)
    print(json.dumps(config, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
