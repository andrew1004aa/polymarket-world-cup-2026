#!/usr/bin/env python3
"""Build resumable Large x high-activity-wallet 2x2 market-minute features."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "regression_inputs/v3/primary_prematch_5m_complete_case.csv.gz"
MANIFEST = ROOT / "regression_ready/trade_files.csv"
THRESHOLDS = ROOT / "large_trade_diagnostics/v3/market_thresholds.csv"
WHALES = ROOT / "analysis_ready/v1/whale_wallet_definitions.csv.gz"
OUT = ROOT / "model_samples/v4/large_wallet_2x2"
CHECKPOINTS = OUT / "checkpoints"
TARGET = OUT / "large_wallet_2x2_primary_prematch.csv.gz"
GROUPS = ("large_high_activity", "large_other", "ordinary_high_activity", "ordinary_other")
VERSION = "v4_large_wallet_2x2_20260815"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read_csv(path, compressed=False):
    opener = gzip.open if compressed else open
    with opener(path, "rt", newline="") as stream:
        return list(csv.DictReader(stream))


def signed_direction(row):
    side = row["side"].upper()
    outcome = row["outcome"].upper()
    return 1 if (side == "BUY") == (outcome == "YES") else -1


def signed_log(value):
    return math.copysign(math.log1p(abs(value)), value) if value else 0.0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    whale_hash = sha256(WHALES)
    whales = {
        row["wallet_address"].lower() for row in read_csv(WHALES, True)
        if row["primary_whale"] == "1"
    }
    if len(whales) != 3355:
        raise RuntimeError(f"Expected 3,355 high-activity wallets, found {len(whales)}")
    thresholds = {row["market_id"]: float(row["p99_threshold_usdc"])
                  for row in read_csv(THRESHOLDS)}
    manifest = {row["market_id"]: row for row in read_csv(MANIFEST)}

    required = defaultdict(dict)
    base_rows = 0
    with gzip.open(BASE, "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            market = row["market_id"]
            minute = int(row["minute_start_timestamp"])
            required[market][minute] = {
                "model_sample_record_id": row["model_sample_record_id"],
                "event_id": row["event_id"], "trade_count": int(row["trade_count"]),
                "gross_volume_usdc": float(row["gross_volume_usdc"]),
                "net_signed_flow_usdc": float(row["net_signed_flow_usdc"]),
            }
            base_rows += 1
    if base_rows != 748481 or len(required) != 312:
        raise RuntimeError("Frozen base sample dimensions changed")

    output_fields = ["model_sample_record_id", "market_id", "event_id", "minute_start_timestamp"]
    for group in GROUPS:
        output_fields += [f"{group}_trade_count", f"{group}_distinct_wallet_count",
                          f"{group}_gross_volume_usdc", f"{group}_net_signed_flow_usdc",
                          f"{group}_signed_log_net_flow"]

    checkpoint_paths = []
    for order, market in enumerate(sorted(required, key=int), 1):
        source = ROOT / manifest[market]["path"]
        source_hash = sha256(source)
        target = CHECKPOINTS / f"{market}.csv.gz"
        metadata_path = CHECKPOINTS / f"{market}.json"
        reusable = False
        if target.exists() and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            reusable = (
                metadata.get("version") == VERSION
                and metadata.get("source_sha256") == source_hash
                and metadata.get("whale_definition_sha256") == whale_hash
                and metadata.get("p99_threshold_usdc") == thresholds[market]
                and metadata.get("required_minutes") == len(required[market])
                and metadata.get("output_sha256") == sha256(target)
            )
        if not reusable:
            minutes = required[market]
            aggregate = {}
            for minute in minutes:
                aggregate[minute] = {
                    group: {"count": 0, "wallets": set(), "gross": 0.0, "net": 0.0}
                    for group in GROUPS
                }
            retained_trades = 0
            with gzip.open(source, "rt", newline="") as stream:
                for trade in csv.DictReader(stream):
                    minute = (int(trade["timestamp"]) // 60) * 60
                    if minute not in aggregate:
                        continue
                    value = float(trade["trade_value_usdc"])
                    wallet = trade["wallet_address"].lower()
                    large = value >= thresholds[market]
                    high = wallet in whales
                    group = (
                        "large_high_activity" if large and high else
                        "large_other" if large else
                        "ordinary_high_activity" if high else "ordinary_other"
                    )
                    cell = aggregate[minute][group]
                    cell["count"] += 1
                    cell["wallets"].add(wallet)
                    cell["gross"] += value
                    cell["net"] += signed_direction(trade) * value
                    retained_trades += 1
            temporary = target.with_suffix(target.suffix + ".tmp")
            count_mismatch = gross_mismatch = net_mismatch = 0
            with gzip.open(temporary, "wt", newline="", compresslevel=6) as stream:
                writer = csv.DictWriter(stream, fieldnames=output_fields)
                writer.writeheader()
                for minute in sorted(aggregate):
                    base = minutes[minute]
                    cells = aggregate[minute]
                    count = sum(cell["count"] for cell in cells.values())
                    gross = sum(cell["gross"] for cell in cells.values())
                    net = sum(cell["net"] for cell in cells.values())
                    count_mismatch += count != base["trade_count"]
                    gross_mismatch += not math.isclose(gross, base["gross_volume_usdc"], rel_tol=1e-9, abs_tol=1e-5)
                    net_mismatch += not math.isclose(net, base["net_signed_flow_usdc"], rel_tol=1e-9, abs_tol=1e-5)
                    row = {"model_sample_record_id": base["model_sample_record_id"],
                           "market_id": market, "event_id": base["event_id"],
                           "minute_start_timestamp": minute}
                    for group, cell in cells.items():
                        row.update({
                            f"{group}_trade_count": cell["count"],
                            f"{group}_distinct_wallet_count": len(cell["wallets"]),
                            f"{group}_gross_volume_usdc": format(cell["gross"], ".12g"),
                            f"{group}_net_signed_flow_usdc": format(cell["net"], ".12g"),
                            f"{group}_signed_log_net_flow": format(signed_log(cell["net"]), ".12g"),
                        })
                    writer.writerow(row)
            if count_mismatch or gross_mismatch or net_mismatch:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Reconciliation failed for {market}: count={count_mismatch}, "
                    f"gross={gross_mismatch}, net={net_mismatch}"
                )
            temporary.replace(target)
            metadata = {
                "version": VERSION, "status": "complete_qc_passed", "market_id": market,
                "source": str(source.relative_to(ROOT)), "source_sha256": source_hash,
                "whale_definition_sha256": whale_hash,
                "p99_threshold_usdc": thresholds[market],
                "required_minutes": len(minutes), "retained_trades": retained_trades,
                "count_mismatches": count_mismatch, "gross_mismatches": gross_mismatch,
                "net_mismatches": net_mismatch, "output_sha256": sha256(target),
                "completed_at": now(),
            }
            atomic_json(metadata_path, metadata)
        checkpoint_paths.append(target)
        print(f"[{order}/312] {market}: {len(required[market]):,} minutes", flush=True)

    temporary = TARGET.with_suffix(TARGET.suffix + ".tmp")
    totals = {group: {"trades": 0, "gross": 0.0, "active_minutes": 0, "markets": set()}
              for group in GROUPS}
    combined_rows = 0
    with gzip.open(temporary, "wt", newline="", compresslevel=6) as destination:
        writer = csv.DictWriter(destination, fieldnames=output_fields)
        writer.writeheader()
        for path in checkpoint_paths:
            with gzip.open(path, "rt", newline="") as source:
                for row in csv.DictReader(source):
                    writer.writerow(row)
                    combined_rows += 1
                    for group in GROUPS:
                        count = int(row[f"{group}_trade_count"])
                        totals[group]["trades"] += count
                        totals[group]["gross"] += float(row[f"{group}_gross_volume_usdc"])
                        if count:
                            totals[group]["active_minutes"] += 1
                            totals[group]["markets"].add(row["market_id"])
    if combined_rows != base_rows:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Combined rows {combined_rows} != {base_rows}")
    temporary.replace(TARGET)

    table = []
    for group in GROUPS:
        table.append({
            "group": group, "trades": totals[group]["trades"],
            "gross_volume_usdc": totals[group]["gross"],
            "active_market_minutes": totals[group]["active_minutes"],
            "markets_with_activity": len(totals[group]["markets"]),
        })
    large_trades = totals["large_high_activity"]["trades"] + totals["large_other"]["trades"]
    high_trades = totals["large_high_activity"]["trades"] + totals["ordinary_high_activity"]["trades"]
    summary = {
        "generated_at": now(), "status": "built_qc_passed", "version": VERSION,
        "base": str(BASE.relative_to(ROOT)), "base_sha256": sha256(BASE),
        "whale_definition": "top 1% of wallets by full canonical-window cumulative USDC volume; operational high-activity-wallet proxy",
        "whale_wallets": len(whales), "whale_definition_sha256": whale_hash,
        "large_definition": "trade_value_usdc >= within-market P99 threshold",
        "rows": combined_rows, "markets": len(required), "groups": table,
        "large_trades_from_high_activity_fraction": totals["large_high_activity"]["trades"] / large_trades,
        "high_activity_trades_that_are_large_fraction": totals["large_high_activity"]["trades"] / high_trades,
        "output": str(TARGET.relative_to(ROOT)), "output_sha256": sha256(TARGET),
        "reconciliation": {"count_mismatches": 0, "gross_mismatches": 0, "net_mismatches": 0},
    }
    atomic_json(OUT / "summary.json", summary)
    with (OUT / "group_diagnostics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
