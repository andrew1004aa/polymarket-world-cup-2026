#!/usr/bin/env python3
"""Prepare common complete-case primary regression input and diagnostics."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = ROOT / "regression_inputs"


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


def signed_log(value: float) -> float:
    return 0.0 if value == 0 else math.copysign(math.log1p(abs(value)), value)


def quantiles(values: list[int]) -> dict:
    ordered = sorted(values)
    if not ordered:
        return {}
    def pick(q: float) -> int:
        return ordered[round((len(ordered) - 1) * q)]
    return {"min": ordered[0], "p25": pick(.25), "median": pick(.5), "p75": pick(.75), "max": ordered[-1]}


def correlation_matrix(names: list[str], n: int, sums: dict, squares: dict, cross: dict) -> dict:
    result = {}
    for left in names:
        result[left] = {}
        for right in names:
            numerator = cross[tuple(sorted((left, right)))] - sums[left] * sums[right] / n
            denom_left = squares[left] - sums[left] ** 2 / n
            denom_right = squares[right] - sums[right] ** 2 / n
            denominator = math.sqrt(max(0.0, denom_left) * max(0.0, denom_right))
            result[left][right] = None if denominator == 0 else numerator / denominator
    return result


def build(version: str, force: bool = False) -> None:
    out = INPUT_ROOT / version
    config_path = out / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = ROOT / config["source"]
    target = ROOT / config["output"]
    qc_path = out / "regression_input_qc.json"
    if target.exists() and qc_path.exists() and not force:
        print("Regression input already exists; use --force to rebuild.")
        return

    required = config["required_source_fields"]
    missing_by_field = Counter(); sequential_exclusions = Counter()
    market_counts = Counter(); event_counts = Counter(); hour_counts = Counter()
    market_ranges = defaultdict(lambda: defaultdict(lambda: [math.inf, -math.inf]))
    split_flow_fields = config.get("split_flow_fields", [
        "signed_log_whale_net_flow", "signed_log_nonwhale_net_flow"])
    diagnostic_names = [
        "delta_yes_price", "signed_log_total_net_flow", *split_flow_fields,
        "yes_price_t", "lagged_30m_price_change", "log_lagged_30m_volume",
        "log_minutes_to_kickoff",
    ]
    sums = Counter(); squares = Counter(); cross = Counter(); zeros = Counter()
    minima = {name: math.inf for name in diagnostic_names}; maxima = {name: -math.inf for name in diagnostic_names}
    source_rows = output_rows = 0
    temp = target.with_suffix(target.suffix + ".tmp")

    with gzip.open(source, "rt", encoding="utf-8", newline="") as inp:
        reader = csv.DictReader(inp)
        output_fields = list(reader.fieldnames) + [
            "signed_log_total_net_flow", "minutes_to_actual_kickoff",
            "log_minutes_to_kickoff", "log_lagged_30m_volume",
        ]
        with gzip.open(temp, "wt", encoding="utf-8", newline="", compresslevel=6) as output:
            writer = csv.DictWriter(output, fieldnames=output_fields); writer.writeheader()
            for row in reader:
                source_rows += 1
                first_missing = None
                for field in required:
                    if row[field] == "":
                        missing_by_field[field] += 1
                        if first_missing is None:
                            first_missing = field
                if first_missing is not None:
                    sequential_exclusions[first_missing] += 1
                    continue

                net = float(row["net_signed_flow_usdc"])
                minutes_to_kickoff = max(0.0, -float(row["minutes_from_actual_kickoff"]))
                derived = {
                    "signed_log_total_net_flow": signed_log(net),
                    "minutes_to_actual_kickoff": minutes_to_kickoff,
                    "log_minutes_to_kickoff": math.log1p(minutes_to_kickoff),
                    # Rolling subtraction upstream can leave negative floating-
                    # point dust. Gross volume is non-negative by definition.
                    "log_lagged_30m_volume": math.log1p(
                        max(0.0, float(row["lagged_30m_gross_volume_usdc"]))
                    ),
                }
                row.update({key: format(value, ".12g") for key, value in derived.items()})
                writer.writerow(row); output_rows += 1

                values = {
                    "delta_yes_price": float(row["delta_yes_price"]),
                    "signed_log_total_net_flow": derived["signed_log_total_net_flow"],
                    "yes_price_t": float(row["yes_price_t"]),
                    "lagged_30m_price_change": float(row["lagged_30m_price_change"]),
                    "log_lagged_30m_volume": derived["log_lagged_30m_volume"],
                    "log_minutes_to_kickoff": derived["log_minutes_to_kickoff"],
                }
                values.update({name: float(row[name]) for name in split_flow_fields})
                for name, value in values.items():
                    sums[name] += value; squares[name] += value * value
                    minima[name] = min(minima[name], value); maxima[name] = max(maxima[name], value)
                    if value == 0: zeros[name] += 1
                for i, left in enumerate(diagnostic_names):
                    for right in diagnostic_names[i:]:
                        cross[tuple(sorted((left, right)))] += values[left] * values[right]

                market = row["market_id"]; event = row["event_id"]; hour = row["calendar_hour_utc"]
                market_counts[market] += 1; event_counts[event] += 1; hour_counts[hour] += 1
                for name in ["delta_yes_price", "signed_log_total_net_flow", *split_flow_fields]:
                    bounds = market_ranges[market][name]; bounds[0] = min(bounds[0], values[name]); bounds[1] = max(bounds[1], values[name])
                if source_rows % 250000 == 0:
                    print(f"Scanned {source_rows:,} primary rows", flush=True)
    temp.replace(target)

    no_within_market_variation = {
        name: 0 for name in ["delta_yes_price", "signed_log_total_net_flow", *split_flow_fields]}
    for variables in market_ranges.values():
        for name, bounds in variables.items():
            if bounds[0] == bounds[1]: no_within_market_variation[name] += 1

    diagnostics = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "version": version,
        "source": str(source.relative_to(ROOT)), "source_rows": source_rows,
        "complete_case_rows": output_rows, "excluded_rows": source_rows - output_rows,
        "missing_by_field_nonexclusive": dict(missing_by_field),
        "sequential_exclusions": dict(sequential_exclusions),
        "groups": {
            "markets": len(market_counts), "events": len(event_counts), "calendar_hours": len(hour_counts),
            "market_size": quantiles(list(market_counts.values())),
            "event_cluster_size": quantiles(list(event_counts.values())),
            "calendar_hour_size": quantiles(list(hour_counts.values())),
            "singleton_markets": sum(v == 1 for v in market_counts.values()),
            "singleton_events": sum(v == 1 for v in event_counts.values()),
            "singleton_calendar_hours": sum(v == 1 for v in hour_counts.values()),
        },
        "variable_diagnostics": {name: {"min": minima[name], "max": maxima[name], "zero_rows": zeros[name]} for name in diagnostic_names},
        "no_within_market_variation": no_within_market_variation,
        "correlations": correlation_matrix(diagnostic_names, output_rows, sums, squares, cross),
        "output": str(target.relative_to(ROOT)), "output_sha256": sha256(target), "output_bytes": target.stat().st_size,
        "errors": [],
    }
    if source_rows != output_rows + sum(sequential_exclusions.values()):
        diagnostics["errors"].append("Complete-case reconciliation failed")
    if len(market_counts) != 312: diagnostics["errors"].append(f"Expected 312 markets, found {len(market_counts)}")
    if len(event_counts) != 104: diagnostics["errors"].append(f"Expected 104 events, found {len(event_counts)}")
    if any(not math.isfinite(v) for name in diagnostic_names for v in (minima[name], maxima[name])):
        diagnostics["errors"].append("Non-finite diagnostic range")
    atomic_json(qc_path, diagnostics)

    with (out / "exclusion_report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["field", "missing_rows_nonexclusive", "sequential_exclusion_rows"])
        for field in required: writer.writerow([field, missing_by_field[field], sequential_exclusions[field]])
    with (out / "group_diagnostics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["group_type", "group_id", "rows"])
        for kind, counts in [("market", market_counts), ("event", event_counts), ("calendar_hour", hour_counts)]:
            for key, value in sorted(counts.items()): writer.writerow([kind, key, value])
    manifest = {
        "path": str(target.relative_to(ROOT)), "rows": output_rows,
        "sha256": diagnostics["output_sha256"], "file_bytes": diagnostics["output_bytes"],
        "source_path": str(source.relative_to(ROOT)), "source_sha256": sha256(source),
    }
    with (out / "file_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest)); writer.writeheader(); writer.writerow(manifest)
    if diagnostics["errors"]: raise RuntimeError(diagnostics["errors"])
    config["status"] = "built_qc_passed"; config["built_at"] = diagnostics["generated_at"]
    atomic_json(config_path, config)
    print(json.dumps(diagnostics, indent=2), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--version", default=None); parser.add_argument("--force", action="store_true"); args = parser.parse_args()
    version = args.version or (INPUT_ROOT / "CURRENT_VERSION").read_text(encoding="utf-8").strip()
    build(version, args.force); return 0


if __name__ == "__main__": raise SystemExit(main())
