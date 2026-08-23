#!/usr/bin/env python3
"""Build a common prematch sample observed at 1, 5 and 15 minutes."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "robustness_inputs" / "v1"

OUTPUT_FIELDS = [
    "source_analysis_record_id", "market_id", "event_id",
    "calendar_hour_utc", "minute_start_timestamp", "minute_start_utc",
    "net_signed_flow_usdc", "signed_log_total_net_flow",
    "signed_log_whale_net_flow", "signed_log_nonwhale_net_flow",
    "yes_price_t", "lagged_30m_price_change", "log_lagged_30m_volume",
    "log_minutes_to_kickoff", "delta_yes_price_1m",
    "delta_yes_price_5m", "delta_yes_price_15m",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_horizon(path: Path) -> tuple[dict[str, str], int, int]:
    values = {}
    rows = duplicates = 0
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"source_analysis_record_id", "delta_yes_price"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise RuntimeError(f"{path}: missing {missing}")
        for row in reader:
            rows += 1
            key = row["source_analysis_record_id"]
            if key in values:
                duplicates += 1
            values[key] = row["delta_yes_price"]
    return values, rows, duplicates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v1")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    out = ROOT / "robustness_inputs" / args.version
    config_path = out / "horizon_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_fields = config.get("output_fields", OUTPUT_FIELDS)
    output = ROOT / config["output"]
    qc_path = out / "horizon_input_qc.json"
    if output.exists() and qc_path.exists() and not args.force:
        print("Matched horizon input already built; use --force to rebuild.")
        return 0

    one_path = ROOT / config["one_minute_source"]
    five_path = ROOT / config["five_minute_source"]
    fifteen_path = ROOT / config["fifteen_minute_source"]
    print("Loading 1-minute keys", flush=True)
    one, one_rows, one_duplicates = load_horizon(one_path)
    print("Loading 15-minute keys", flush=True)
    fifteen, fifteen_rows, fifteen_duplicates = load_horizon(fifteen_path)

    temporary = output.with_suffix(output.suffix + ".tmp")
    five_rows = matched = 0
    five_duplicates = 0
    seen = set()
    markets = set(); events = set(); hours = set()
    missing_one = missing_fifteen = 0
    with gzip.open(five_path, "rt", encoding="utf-8", newline="") as source, \
            gzip.open(temporary, "wt", encoding="utf-8", newline="", compresslevel=6) as target:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(target, fieldnames=output_fields)
        writer.writeheader()
        for row in reader:
            five_rows += 1
            key = row["source_analysis_record_id"]
            if key in seen:
                five_duplicates += 1
            seen.add(key)
            if key not in one:
                missing_one += 1
                continue
            if key not in fifteen:
                missing_fifteen += 1
                continue
            output_row = {field: row.get(field, "") for field in output_fields}
            output_row.update({
                "delta_yes_price_1m": one[key],
                "delta_yes_price_5m": row["delta_yes_price"],
                "delta_yes_price_15m": fifteen[key],
            })
            if any(output_row[field] == "" for field in output_fields):
                raise RuntimeError(f"Unexpected missing matched field at {key}")
            writer.writerow(output_row)
            matched += 1
            markets.add(row["market_id"]); events.add(row["event_id"])
            hours.add(row["calendar_hour_utc"])
            if five_rows % 250000 == 0:
                print(f"Scanned {five_rows:,} five-minute rows", flush=True)
    temporary.replace(output)

    errors = []
    if one_duplicates or five_duplicates or fifteen_duplicates:
        errors.append("Duplicate join keys detected")
    if len(markets) != 312 or len(events) != 104:
        errors.append("Matched sample lost required market/event coverage")
    qc = {
        "generated_at": now(), "version": args.version,
        "source_rows": {"1m": one_rows, "5m_complete_case": five_rows, "15m": fifteen_rows},
        "duplicate_keys": {"1m": one_duplicates, "5m": five_duplicates, "15m": fifteen_duplicates},
        "five_minute_rows_without_1m": missing_one,
        "five_minute_rows_without_15m": missing_fifteen,
        "matched_rows": matched, "markets": len(markets), "events": len(events),
        "calendar_hours": len(hours), "output": config["output"],
        "output_sha256": sha256(output), "output_bytes": output.stat().st_size,
        "source_sha256": {"1m": sha256(one_path), "5m": sha256(five_path), "15m": sha256(fifteen_path)},
        "errors": errors,
    }
    atomic_json(qc_path, qc)
    config["status"] = "built_qc_passed" if not errors else "built_qc_failed"
    config["built_at"] = qc["generated_at"]
    atomic_json(config_path, config)
    print(json.dumps(qc, indent=2), flush=True)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
