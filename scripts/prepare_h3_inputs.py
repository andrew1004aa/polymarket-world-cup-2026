#!/usr/bin/env python3
"""Construct H3 subsequent-movement and Brier-score outcomes."""

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

SOURCE_FIELDS = [
    "source_analysis_record_id", "market_id", "event_id", "calendar_hour_utc",
    "minute_start_timestamp", "minute_start_utc", "net_signed_flow_usdc",
    "signed_log_total_net_flow", "signed_log_whale_net_flow",
    "signed_log_nonwhale_net_flow", "yes_price_t",
    "lagged_30m_price_change", "log_lagged_30m_volume",
    "log_minutes_to_kickoff", "delta_yes_price_1m",
    "delta_yes_price_5m", "delta_yes_price_15m",
]
DERIVED_FIELDS = [
    "yes_outcome_won", "yes_price_5m", "yes_price_15m",
    "delta_5_to_15", "log_abs_total_net_flow",
    "log_abs_whale_net_flow", "log_abs_nonwhale_net_flow",
    "directional_subsequent_total", "directional_subsequent_whale",
    "brier_improvement_0_5", "brier_improvement_0_15",
    "brier_improvement_5_15",
]
FIELDS = SOURCE_FIELDS + DERIVED_FIELDS


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


def sign(value: float) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v1")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    out = ROOT / args.output_dir if args.output_dir else DEFAULT_OUT
    config_path = out / "h3_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_fields = config.get("source_fields", SOURCE_FIELDS)
    split_signed = config.get("split_signed_fields", [
        "signed_log_whale_net_flow", "signed_log_nonwhale_net_flow"])
    abs_split = config.get("abs_split_fields", [
        "log_abs_whale_net_flow", "log_abs_nonwhale_net_flow"])
    directional_split = config.get("directional_split_field", "directional_subsequent_whale")
    derived_fields = [
        "yes_outcome_won", "yes_price_5m", "yes_price_15m", "delta_5_to_15",
        "log_abs_total_net_flow", *abs_split, "directional_subsequent_total",
        directional_split, "brier_improvement_0_5", "brier_improvement_0_15",
        "brier_improvement_5_15",
    ]
    fields = source_fields + derived_fields
    source = ROOT / config["matched_source"]
    outcomes_path = ROOT / config["market_outcomes_source"]
    output = ROOT / config["output"]
    qc_path = out / "h3_input_qc.json"
    if output.exists() and qc_path.exists() and not args.force:
        print("H3 input already built; use --force to rebuild.")
        return 0
    if sha256(source) != config["matched_source_sha256"]:
        raise SystemExit("Matched source checksum mismatch")

    outcomes = {}
    with outcomes_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["market_type"] == "match":
                if row["yes_outcome_won"] not in {"0", "1"}:
                    raise RuntimeError(f"Invalid outcome for market {row['market_id']}")
                outcomes[row["market_id"]] = int(row["yes_outcome_won"])
    if len(outcomes) != 312:
        raise RuntimeError(f"Expected 312 match outcomes, observed {len(outcomes)}")

    temporary = output.with_suffix(output.suffix + ".tmp")
    rows = invalid_prices = missing_outcomes = 0
    markets = set(); events = set(); hours = set()
    sums = {name: 0.0 for name in ["directional_subsequent_total", directional_split,
                                   "brier_improvement_0_5", "brier_improvement_0_15",
                                   "brier_improvement_5_15"]}
    with gzip.open(source, "rt", encoding="utf-8", newline="") as src, \
            gzip.open(temporary, "wt", encoding="utf-8", newline="", compresslevel=6) as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=fields)
        writer.writeheader()
        for row in reader:
            rows += 1
            market = row["market_id"]
            if market not in outcomes:
                missing_outcomes += 1
                continue
            y = outcomes[market]
            p0 = float(row["yes_price_t"])
            d5 = float(row["delta_yes_price_5m"])
            d15 = float(row["delta_yes_price_15m"])
            p5 = p0 + d5; p15 = p0 + d15; subsequent = d15 - d5
            if not (0 <= p0 <= 1 and 0 <= p5 <= 1 and 0 <= p15 <= 1):
                invalid_prices += 1
                continue
            total_signed = float(row["signed_log_total_net_flow"])
            large_signed = float(row[split_signed[0]])
            ordinary_signed = float(row[split_signed[1]])
            b05 = (y-p0)**2 - (y-p5)**2
            b015 = (y-p0)**2 - (y-p15)**2
            b515 = (y-p5)**2 - (y-p15)**2
            derived = {
                "yes_outcome_won": y, "yes_price_5m": p5,
                "yes_price_15m": p15, "delta_5_to_15": subsequent,
                "log_abs_total_net_flow": abs(total_signed),
                "directional_subsequent_total": sign(total_signed) * subsequent,
                directional_split: sign(large_signed) * subsequent,
                "brier_improvement_0_5": b05,
                "brier_improvement_0_15": b015,
                "brier_improvement_5_15": b515,
            }
            derived.update({abs_split[0]: abs(large_signed), abs_split[1]: abs(ordinary_signed)})
            output_row = {field: row.get(field, "") for field in source_fields}
            output_row.update(derived)
            writer.writerow(output_row)
            markets.add(market); events.add(row["event_id"]); hours.add(row["calendar_hour_utc"])
            for name in sums:
                sums[name] += derived[name]
            if rows % 250000 == 0:
                print(f"Constructed {rows:,} H3 rows", flush=True)
    temporary.replace(output)
    output_rows = rows - missing_outcomes - invalid_prices
    errors = []
    if output_rows != int(config["expected_rows"]): errors.append("Unexpected H3 output row count")
    if missing_outcomes or invalid_prices: errors.append("Missing outcome or invalid derived price")
    if len(markets) != 312 or len(events) != 104: errors.append("H3 coverage mismatch")
    qc = {
        "generated_at": now(), "version": args.version, "source_rows": rows,
        "output_rows": output_rows, "markets": len(markets), "events": len(events),
        "calendar_hours": len(hours), "missing_outcomes": missing_outcomes,
        "invalid_derived_prices": invalid_prices,
        "outcome_counts": {"yes": sum(1 for value in outcomes.values() if value == 1),
                           "no": sum(1 for value in outcomes.values() if value == 0)},
        "outcome_means": {name: sums[name]/output_rows for name in sums},
        "output": config["output"], "output_sha256": sha256(output),
        "output_bytes": output.stat().st_size, "errors": errors,
    }
    atomic_json(qc_path, qc)
    config["status"] = "built_qc_passed" if not errors else "built_qc_failed"
    config["built_at"] = qc["generated_at"]
    atomic_json(config_path, config)
    print(json.dumps(qc, indent=2), flush=True)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
