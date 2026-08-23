#!/usr/bin/env python3
"""Build auditable, versioned model samples from analysis_ready/v2."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "model_samples"

V1_FIELDS = [
    "model_sample_record_id", "source_analysis_record_id", "model_sample_version",
    "sample_name", "sample_role", "horizon_minutes", "market_id", "condition_id",
    "event_id", "fifa_match_id", "market_type", "market_subtype", "market_role",
    "question", "country", "stage", "home_team", "away_team",
    "minute_start_timestamp", "minute_start_utc", "minute_end_timestamp",
    "minute_end_utc", "date_utc", "hour_of_day_utc", "calendar_hour_utc",
    "calendar_15min_block_utc", "phase_at_minute_start", "phase_at_minute_end",
    "actual_kickoff_utc", "resolved_on_timestamp", "minutes_from_actual_kickoff",
    "minutes_to_resolution", "yes_outcome_won", "trade_count",
    "distinct_wallet_count", "gross_volume_usdc", "net_signed_flow_usdc",
    "whale_trade_count", "whale_distinct_wallet_count", "whale_gross_volume_usdc",
    "whale_net_signed_flow_usdc", "nonwhale_trade_count",
    "nonwhale_distinct_wallet_count", "nonwhale_gross_volume_usdc",
    "nonwhale_net_signed_flow_usdc", "whale_flow_share",
    "signed_log_whale_net_flow", "signed_log_nonwhale_net_flow",
    "lagged_30m_gross_volume_usdc", "lagged_30m_price_change",
    "baseline_price_timestamp", "baseline_price_utc", "baseline_price_age_seconds",
    "yes_price_t", "target_timestamp", "target_timestamp_utc",
    "future_price_timestamp", "future_price_timestamp_utc",
    "future_price_lateness_seconds", "yes_price_future", "delta_yes_price",
    "directional_impact", "brier_improvement", "crosses_kickoff",
    "crosses_resolution", "baseline_tolerance_minutes",
    "target_tolerance_minutes",
]

V3_FLOW_FIELDS = [
    "p99_threshold_usdc", "p95_threshold_usdc", "sparse_p99_market",
    "p99_large_trade_count", "p99_large_distinct_wallet_count",
    "p99_large_gross_volume_usdc", "p99_large_net_signed_flow_usdc",
    "p99_signed_log_large_net_flow", "p99_ordinary_trade_count",
    "p99_ordinary_distinct_wallet_count", "p99_ordinary_gross_volume_usdc",
    "p99_ordinary_net_signed_flow_usdc", "p99_signed_log_ordinary_net_flow",
    "p99_large_gross_volume_share", "p95_large_trade_count",
    "p95_large_distinct_wallet_count", "p95_large_gross_volume_usdc",
    "p95_large_net_signed_flow_usdc", "p95_signed_log_large_net_flow",
    "p95_ordinary_trade_count", "p95_ordinary_distinct_wallet_count",
    "p95_ordinary_gross_volume_usdc", "p95_ordinary_net_signed_flow_usdc",
    "p95_signed_log_ordinary_net_flow", "p95_large_gross_volume_share",
]


def fields_for_version(version: str) -> list[str]:
    if version != "v3":
        return list(V1_FIELDS)
    whale_fields = {field for field in V1_FIELDS if "whale" in field}
    base = [field for field in V1_FIELDS if field not in whale_fields]
    insert_at = base.index("lagged_30m_gross_volume_usdc")
    return base[:insert_at] + V3_FLOW_FIELDS + base[insert_at:]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def transform(row: dict[str, str], name: str, spec: dict, version: str,
              fieldnames: list[str]) -> dict[str, str | int]:
    horizon = int(spec["horizon_minutes"])
    direct = {field: row.get(field, "") for field in fieldnames if field in row}
    direct.update({
        "model_sample_record_id": f"{name}:{row['analysis_record_id']}",
        "source_analysis_record_id": row["analysis_record_id"],
        "model_sample_version": version,
        "sample_name": name,
        "sample_role": spec["role"],
        "horizon_minutes": horizon,
        "target_timestamp": row[f"target_timestamp_{horizon}m"],
        "target_timestamp_utc": row[f"target_timestamp_{horizon}m_utc"],
        "future_price_timestamp": row[f"price_timestamp_{horizon}m"],
        "future_price_timestamp_utc": row[f"price_timestamp_{horizon}m_utc"],
        "future_price_lateness_seconds": row[f"price_lateness_{horizon}m_seconds"],
        "yes_price_future": row[f"yes_price_{horizon}m"],
        "delta_yes_price": row[f"delta_yes_price_{horizon}m"],
        "directional_impact": row[f"directional_impact_{horizon}m"],
        "brier_improvement": row[f"brier_improvement_{horizon}m"],
        "crosses_kickoff": row[f"crosses_kickoff_{horizon}m"],
        "crosses_resolution": row[f"crosses_resolution_{horizon}m"],
        "baseline_tolerance_minutes": int(spec["baseline_tolerance_minutes"]),
        "target_tolerance_minutes": int(spec["target_tolerance_minutes"]),
    })
    return direct


def exclusion_reason(row: dict[str, str], spec: dict) -> str | None:
    horizon = int(spec["horizon_minutes"])
    if row["phase_at_minute_start"] != spec["phase"]:
        return "phase_mismatch"
    if spec["exclude_crosses_kickoff"] and row[f"crosses_kickoff_{horizon}m"] == "1":
        return "crosses_kickoff"
    if spec["exclude_crosses_resolution"] and row[f"crosses_resolution_{horizon}m"] == "1":
        return "crosses_resolution"
    baseline_tol = int(spec["baseline_tolerance_minutes"])
    if row[f"baseline_within_{baseline_tol}m"] != "1":
        return "baseline_outside_tolerance"
    target_tol = int(spec["target_tolerance_minutes"])
    if row[f"target_{horizon}m_within_{target_tol}m"] != "1":
        return "target_outside_tolerance"
    if row[f"delta_yes_price_{horizon}m"] == "":
        return "delta_price_unavailable"
    return None


def build(version: str, force: bool = False) -> None:
    out = MODEL_ROOT / version
    config_path = out / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    fieldnames = fields_for_version(version)
    source = ROOT / config["source_panel"]
    specs = config["samples"]
    final_paths = {name: out / f"{name}.csv.gz" for name in specs}
    if not force and all(path.exists() for path in final_paths.values()) and (out / "model_samples_qc.json").exists():
        print("All model samples already built; use --force to rebuild.", flush=True)
        return

    temp_paths = {name: path.with_suffix(path.suffix + ".tmp") for name, path in final_paths.items()}
    handles = {}; writers = {}; counters = {name: defaultdict(int) for name in specs}
    phase_specs = defaultdict(list)
    for name, spec in specs.items():
        phase_specs[spec["phase"]].append((name, spec))
    try:
        for name, temp in temp_paths.items():
            handle = gzip.open(temp, "wt", encoding="utf-8", newline="", compresslevel=6)
            handles[name] = handle; writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader(); writers[name] = writer

        source_rows = 0
        with gzip.open(source, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                source_rows += 1
                phase = row["phase_at_minute_start"]
                for name, spec in phase_specs.get(phase, []):
                    counters[name]["phase_eligible"] += 1
                    reason = exclusion_reason(row, spec)
                    if reason:
                        counters[name][reason] += 1
                    else:
                        writers[name].writerow(transform(row, name, spec, version, fieldnames))
                        counters[name]["included"] += 1
                if source_rows % 250000 == 0:
                    print(f"Scanned {source_rows:,} source-panel rows", flush=True)
        for handle in handles.values():
            handle.close()
        handles.clear()
        for name in specs:
            temp_paths[name].replace(final_paths[name])
    except Exception:
        for handle in handles.values():
            handle.close()
        for temp in temp_paths.values():
            temp.unlink(missing_ok=True)
        raise

    manifest = []
    flow_rows = []
    errors = []
    for name, spec in specs.items():
        path = final_paths[name]
        count = counters[name]
        accounted = count["included"] + count["crosses_kickoff"] + count["crosses_resolution"] + count["baseline_outside_tolerance"] + count["target_outside_tolerance"] + count["delta_price_unavailable"]
        if accounted != count["phase_eligible"]:
            errors.append(f"{name} phase reconciliation {accounted} != {count['phase_eligible']}")
        manifest.append({
            "sample_name": name, "role": spec["role"], "horizon_minutes": spec["horizon_minutes"],
            "path": str(path.relative_to(ROOT)), "rows": count["included"],
            "sha256": sha256(path), "file_bytes": path.stat().st_size,
        })
        for stage in ["source_panel_rows", "phase_eligible", "crosses_kickoff", "crosses_resolution", "baseline_outside_tolerance", "target_outside_tolerance", "delta_price_unavailable", "included"]:
            value = source_rows if stage == "source_panel_rows" else count[stage]
            flow_rows.append({"sample_name": name, "stage": stage, "rows": value})

    with (out / "file_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0])); writer.writeheader(); writer.writerows(manifest)
    with (out / "sample_flow.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_name", "stage", "rows"]); writer.writeheader(); writer.writerows(flow_rows)

    qc = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "version": version,
        "source_panel": str(source.relative_to(ROOT)), "source_panel_rows": source_rows,
        "source_panel_sha256": sha256(source), "primary_sample": config["primary_sample"],
        "samples": {name: dict(counts) for name, counts in counters.items()},
        "files": manifest, "errors": errors,
    }
    atomic_json(out / "model_samples_qc.json", qc)
    if errors:
        raise RuntimeError(errors)
    config["status"] = "built_qc_passed"; config["built_at"] = qc["generated_at"]
    atomic_json(config_path, config)
    print(json.dumps(qc, indent=2), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    version = args.version or (MODEL_ROOT / "CURRENT_VERSION").read_text(encoding="utf-8").strip()
    build(version, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
