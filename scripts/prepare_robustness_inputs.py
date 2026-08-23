#!/usr/bin/env python3
"""Build frozen v1 zero-outcome and non-overlap robustness samples."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "robustness_inputs" / "v1"

FIELDS = [
    "model_sample_record_id", "market_id", "event_id", "calendar_hour_utc",
    "minute_start_timestamp", "minute_start_utc", "delta_yes_price",
    "any_price_change", "net_signed_flow_usdc",
    "whale_net_signed_flow_usdc", "nonwhale_net_signed_flow_usdc",
    "signed_log_total_net_flow", "signed_log_whale_net_flow",
    "signed_log_nonwhale_net_flow", "log_abs_total_net_flow",
    "log_abs_whale_net_flow", "log_abs_nonwhale_net_flow", "yes_price_t",
    "lagged_30m_price_change", "log_lagged_30m_volume",
    "log_minutes_to_kickoff",
]


def utc_now() -> str:
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v1")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    out = ROOT / args.output_dir if args.output_dir else ROOT / "robustness_inputs" / args.version
    config_path = out / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    fields = config.get("output_fields", FIELDS)
    split_net_fields = config.get("split_net_fields", [
        "whale_net_signed_flow_usdc", "nonwhale_net_signed_flow_usdc"])
    abs_flow_fields = config.get("abs_flow_fields", [
        "log_abs_whale_net_flow", "log_abs_nonwhale_net_flow"])
    source = ROOT / config["source"]
    if sha256(source) != config["source_sha256"]:
        raise SystemExit("Frozen source checksum mismatch")

    paths = {
        "any_price_change_5m": out / "any_price_change_5m.csv.gz",
        "nonzero_price_change_5m": out / "nonzero_price_change_5m.csv.gz",
        "nonoverlapping_5m": out / "nonoverlapping_5m.csv.gz",
    }
    qc_path = out / "robustness_input_qc.json"
    if not args.force and qc_path.exists() and all(path.exists() for path in paths.values()):
        print("Robustness inputs already built; use --force for a deliberate rebuild.")
        return 0

    temps = {name: path.with_suffix(path.suffix + ".tmp") for name, path in paths.items()}
    handles = {}
    writers = {}
    counts = Counter()
    groups = {name: {"markets": set(), "events": set(), "hours": set()} for name in paths}
    errors = []
    try:
        for name, path in temps.items():
            handle = gzip.open(path, "wt", encoding="utf-8", newline="", compresslevel=6)
            handles[name] = handle
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writers[name] = writer

        with gzip.open(source, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = sorted(set(fields) - {
                "any_price_change", "log_abs_total_net_flow",
                *abs_flow_fields,
            } - set(reader.fieldnames or []))
            if missing:
                raise RuntimeError(f"Missing source fields: {missing}")
            for row in reader:
                counts["source"] += 1
                delta = float(row["delta_yes_price"])
                total = float(row["net_signed_flow_usdc"])
                split_values = [float(row[name]) for name in split_net_fields]
                output = {field: row.get(field, "") for field in fields}
                output.update({
                    "any_price_change": int(delta != 0.0),
                    "log_abs_total_net_flow": math.log1p(abs(total)),
                })
                output.update({name: math.log1p(abs(value)) for name, value in zip(abs_flow_fields, split_values)})
                selected = ["any_price_change_5m"]
                if delta != 0.0:
                    selected.append("nonzero_price_change_5m")
                if int(row["minute_start_timestamp"]) % 300 == 0:
                    selected.append("nonoverlapping_5m")
                for name in selected:
                    writers[name].writerow(output)
                    counts[name] += 1
                    groups[name]["markets"].add(row["market_id"])
                    groups[name]["events"].add(row["event_id"])
                    groups[name]["hours"].add(row["calendar_hour_utc"])
                if counts["source"] % 250000 == 0:
                    print(f"Scanned {counts['source']:,} rows", flush=True)
    except Exception as error:
        errors.append(repr(error))
        raise
    finally:
        for handle in handles.values():
            handle.close()

    if counts["source"] != int(config["expected_source_rows"]):
        raise RuntimeError(
            f"Source rows expected {config['expected_source_rows']}, observed {counts['source']}"
        )
    for name, temporary in temps.items():
        temporary.replace(paths[name])

    qc = {
        "generated_at": utc_now(), "version": args.version,
        "source": config["source"], "source_sha256": config["source_sha256"],
        "source_rows": counts["source"],
        "samples": {}, "errors": errors,
    }
    for name, path in paths.items():
        qc["samples"][name] = {
            "rows": counts[name],
            "markets": len(groups[name]["markets"]),
            "events": len(groups[name]["events"]),
            "calendar_hours": len(groups[name]["hours"]),
            "sha256": sha256(path), "bytes": path.stat().st_size,
            "path": str(path.relative_to(ROOT)),
        }
    if counts["nonzero_price_change_5m"] != int(config.get("expected_nonzero_rows", 27602)):
        qc["errors"].append("Unexpected non-zero price-change count")
    if any(qc["samples"][name]["markets"] != 312 for name in paths):
        qc["errors"].append("One or more samples do not cover all 312 markets")
    if any(qc["samples"][name]["events"] != 104 for name in paths):
        qc["errors"].append("One or more samples do not cover all 104 events")
    atomic_json(qc_path, qc)
    config["status"] = "built_qc_passed" if not qc["errors"] else "built_qc_failed"
    config["built_at"] = qc["generated_at"]
    atomic_json(config_path, config)
    print(json.dumps(qc, indent=2), flush=True)
    return 0 if not qc["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
