#!/usr/bin/env python3
"""Exclude the four pre-specified sparse P99 outright markets for robustness."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "regression_inputs/v3/country"
OUT = ROOT / "robustness_inputs/v3/country_exclude_sparse"
FILES = [
    "outright_5m_complete_case.csv.gz",
    "outright_5m_matched_5_30.csv.gz",
    "outright_30m_matched_5_30.csv.gz",
]
EXPECTED_SPARSE_IDS = {558962, 558977, 558979, 558982}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024): h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, obj: object) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2) + "\n")
    temp.replace(path)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    source_config = json.loads((SOURCE / "config.json").read_text())
    expected = {Path(row["path"]).name: row for row in source_config["files"]}
    outputs = []
    for filename in FILES:
        source = SOURCE / filename
        if sha256(source) != expected[filename]["sha256"]:
            raise RuntimeError(f"Source checksum mismatch: {filename}")
        data = pd.read_csv(source)
        sparse_ids = set(data.loc[data.sparse_p99_market == 1, "market_id"].astype(int).unique())
        if sparse_ids != EXPECTED_SPARSE_IDS:
            raise RuntimeError(f"Unexpected sparse IDs in {filename}: {sparse_ids}")
        filtered = data.loc[data.sparse_p99_market == 0].copy()
        if filtered.market_id.nunique() != 44 or filtered.sparse_p99_market.any():
            raise RuntimeError(f"Sparse exclusion failed: {filename}")
        target = OUT / filename; temp = target.with_suffix(target.suffix + ".tmp")
        filtered.to_csv(temp, index=False, compression="gzip"); temp.replace(target)
        outputs.append({
            "path": str(target.relative_to(ROOT)), "rows": int(len(filtered)),
            "markets": int(filtered.market_id.nunique()),
            "calendar_hours": int(filtered.calendar_hour_utc.nunique()),
            "zero_outcome_share": float((filtered.delta_yes_price == 0).mean()),
            "source_rows": int(len(data)), "rows_removed": int(len(data)-len(filtered)),
            "sha256": sha256(target), "bytes": target.stat().st_size,
        })
    config = {
        "version": "v3_country_exclude_sparse", "status": "built_qc_passed",
        "built_at": datetime.now(timezone.utc).isoformat(), "expected_markets": 44,
        "excluded_market_ids": sorted(EXPECTED_SPARSE_IDS),
        "exclusion_rule": "p99_trade_count < 5", "files": outputs,
        "source_config_sha256": sha256(SOURCE / "config.json"), "errors": [],
    }
    atomic_json(OUT / "config.json", config); print(json.dumps(config, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
