#!/usr/bin/env python3
"""Build reproducible complete-case v3 post-kickoff regression inputs."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "regression_inputs" / "v3" / "post_kickoff"
RESULTS = ROOT / "regression_results" / "v3" / "post_kickoff"
HORIZONS = (1, 5)
REQUIRED = (
    "model_sample_record_id", "market_id", "event_id", "calendar_hour_utc",
    "delta_yes_price", "net_signed_flow_usdc",
    "p99_signed_log_large_net_flow", "p99_signed_log_ordinary_net_flow",
    "yes_price_t", "lagged_30m_price_change", "lagged_30m_gross_volume_usdc",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def signed_log(value: float) -> float:
    return 0.0 if value == 0 else math.copysign(math.log1p(abs(value)), value)


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary = {"generated_at": datetime.now(timezone.utc).isoformat(), "samples": {}, "errors": []}
    for horizon in HORIZONS:
        source = ROOT / "model_samples" / "v3" / f"post_kickoff_pre_resolution_{horizon}m.csv.gz"
        target = OUT / f"post_kickoff_pre_resolution_{horizon}m_complete_case.csv.gz"
        temporary = target.with_suffix(target.suffix + ".tmp")
        source_rows = output_rows = 0
        markets, events, hours, ids = set(), set(), set(), set()
        missing = {field: 0 for field in REQUIRED}
        with gzip.open(source, "rt", encoding="utf-8", newline="") as inp:
            reader = csv.DictReader(inp)
            absent = sorted(set(REQUIRED) - set(reader.fieldnames or []))
            if absent:
                raise RuntimeError(f"Missing source fields for {horizon}m: {absent}")
            fields = list(reader.fieldnames or []) + ["signed_log_total_net_flow", "log_lagged_30m_volume"]
            with gzip.open(temporary, "wt", encoding="utf-8", newline="", compresslevel=6) as out:
                writer = csv.DictWriter(out, fieldnames=fields)
                writer.writeheader()
                for row in reader:
                    source_rows += 1
                    bad = False
                    for field in REQUIRED:
                        if row[field] == "":
                            missing[field] += 1
                            bad = True
                    if bad:
                        continue
                    row["signed_log_total_net_flow"] = format(signed_log(float(row["net_signed_flow_usdc"])), ".12g")
                    row["log_lagged_30m_volume"] = format(math.log1p(max(0.0, float(row["lagged_30m_gross_volume_usdc"]))), ".12g")
                    writer.writerow(row)
                    output_rows += 1
                    ids.add(row["model_sample_record_id"])
                    markets.add(row["market_id"]); events.add(row["event_id"]); hours.add(row["calendar_hour_utc"])
        temporary.replace(target)
        item = {
            "source": str(source.relative_to(ROOT)), "source_sha256": sha256(source),
            "source_rows": source_rows, "complete_case_rows": output_rows,
            "excluded_rows": source_rows - output_rows,
            "missing_by_field_nonexclusive": {k: v for k, v in missing.items() if v},
            "unique_record_ids": len(ids), "markets": len(markets), "events": len(events),
            "calendar_hours": len(hours), "output": str(target.relative_to(ROOT)),
            "output_sha256": sha256(target), "output_bytes": target.stat().st_size,
        }
        if output_rows != len(ids) or len(markets) != 312 or len(events) != 104:
            summary["errors"].append(f"QC failure for {horizon}m")
        summary["samples"][f"{horizon}m"] = item

        result_dir = RESULTS / f"{horizon}m"
        result_dir.mkdir(parents=True, exist_ok=True)
        controls = "yes_price_t + lagged_30m_price_change + log_lagged_30m_volume"
        config = {
            "version": f"v3_postkickoff_{horizon}m", "status": "frozen_pending_estimation",
            "input": item["output"], "input_sha256": item["output_sha256"],
            "expected_rows": output_rows, "expected_markets": 312, "expected_events": 104,
            "expected_calendar_hours": len(hours),
            "required_columns": ["model_sample_record_id", "market_id", "event_id", "calendar_hour_utc",
                                 "delta_yes_price", "signed_log_total_net_flow",
                                 "p99_signed_log_large_net_flow", "p99_signed_log_ordinary_net_flow",
                                 "yes_price_t", "lagged_30m_price_change", "log_lagged_30m_volume"],
            "input_columns": ["model_sample_record_id", "market_id", "event_id", "calendar_hour_utc",
                              "delta_yes_price", "signed_log_total_net_flow",
                              "p99_signed_log_large_net_flow", "p99_signed_log_ordinary_net_flow",
                              "yes_price_t", "lagged_30m_price_change", "log_lagged_30m_volume"],
            "models": {
                "POST_TOTAL": f"delta_yes_price ~ signed_log_total_net_flow + {controls} | market_id + calendar_hour_utc",
                "POST_P99_SPLIT": f"delta_yes_price ~ p99_signed_log_large_net_flow + p99_signed_log_ordinary_net_flow + {controls} | market_id + calendar_hour_utc"
            },
            "equality_model": "POST_P99_SPLIT",
            "equality_terms": ["p99_signed_log_large_net_flow", "p99_signed_log_ordinary_net_flow"],
            "equality_output": "large_ordinary_equality_test.csv",
            "interpretation": "Secondary post-kickoff/pre-resolution association; not a pure in-play or causal model."
        }
        atomic_json(result_dir / "config.json", config)
    atomic_json(OUT / "post_kickoff_input_qc.json", summary)
    print(json.dumps(summary, indent=2))
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
