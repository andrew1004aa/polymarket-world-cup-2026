#!/usr/bin/env python3
"""Select one deterministic ordinary-trade control per P99 event in one market."""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "v4_diagnostics/ordinary_match_prototype"
CALIPER_SECONDS = 6 * 3600


def direction(side: str, outcome: str) -> int:
    return 1 if (side.upper() == "BUY") == (outcome.upper() == "YES") else -1


def phase(timestamp: int, kickoff: str) -> str:
    if not kickoff:
        return "outright"
    kickoff_ts = int(datetime.fromisoformat(kickoff.replace("Z", "+00:00")).timestamp())
    return "pre" if timestamp < kickoff_ts else "post"


def price_bin(price: float) -> int:
    return min(9, max(0, int(price * 10)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market-id", default="1897043")
    parser.add_argument("--large-event-buffer-seconds", type=int, default=900)
    args = parser.parse_args(); mid = args.market_id
    threshold = None
    with (ROOT / "large_trade_diagnostics/v3/market_thresholds.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["market_id"] == mid:
                threshold = float(row["p99_threshold_usdc"]); break
    if threshold is None: raise KeyError(mid)

    large = []; candidates = defaultdict(list)
    trade_path = ROOT / f"regression_ready/trades_by_market/{mid}.csv.gz"
    with gzip.open(trade_path, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            item = {
                "id": row["trade_record_id"], "timestamp": int(row["timestamp"]),
                "direction": direction(row["side"], row["outcome"]),
                "phase": phase(int(row["timestamp"]), row["actual_kickoff_utc"]),
                "price_bin": price_bin(float(row["yes_equivalent_price"])),
                "value": float(row["trade_value_usdc"]),
            }
            key = (item["direction"], item["phase"], item["price_bin"])
            (large if item["value"] >= threshold else candidates[key]).append(item)
    large_times = sorted(item["timestamp"] for item in large)
    buffer_seconds = args.large_event_buffer_seconds
    for key, values in list(candidates.items()):
        clean = []
        for item in values:
            position = bisect.bisect_left(large_times, item["timestamp"])
            neighbours = large_times[max(0, position - 1):position + 1]
            if all(abs(item["timestamp"] - value) > buffer_seconds for value in neighbours):
                clean.append(item)
        candidates[key] = sorted(clean, key=lambda x: (x["timestamp"], x["id"]))

    used = set(); matches = []
    # Match the hardest events first: keys with the fewest available controls.
    large.sort(key=lambda x: (len(candidates[(x["direction"], x["phase"], x["price_bin"])]), x["timestamp"], x["id"]))
    for event in large:
        key = (event["direction"], event["phase"], event["price_bin"]); pool = candidates[key]
        times = [x["timestamp"] for x in pool]; centre = bisect.bisect_left(times, event["timestamp"])
        chosen = None
        for radius in range(len(pool)):
            positions = {centre - radius, centre + radius}
            available = [pool[pos] for pos in positions if 0 <= pos < len(pool) and pool[pos]["id"] not in used
                         and pool[pos]["timestamp"] != event["timestamp"]
                         and abs(pool[pos]["timestamp"] - event["timestamp"]) <= CALIPER_SECONDS]
            if available:
                chosen = min(available, key=lambda x: (abs(x["timestamp"] - event["timestamp"]), x["timestamp"], x["id"])); break
        if chosen:
            used.add(chosen["id"])
            matches.append({
                "large_trade_record_id": event["id"], "large_timestamp": event["timestamp"],
                "ordinary_trade_record_id": chosen["id"], "ordinary_timestamp": chosen["timestamp"],
                "absolute_time_distance_seconds": abs(chosen["timestamp"] - event["timestamp"]),
                "direction": event["direction"], "phase": event["phase"], "price_bin": event["price_bin"],
            })

    OUT.mkdir(parents=True, exist_ok=True)
    suffix = f"buffer_{buffer_seconds}s"
    match_path = OUT / f"market_{mid}_{suffix}_matches.csv"
    with match_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(matches[0])); writer.writeheader(); writer.writerows(matches)
    ids_path = OUT / f"market_{mid}_{suffix}_ordinary_ids.txt"
    ids_path.write_text("\n".join(x["ordinary_trade_record_id"] for x in matches) + "\n")
    subprocess.run([
        sys.executable, str(ROOT / "scripts/build_v4_push_follow_reverse_prototype.py"),
        "--market-id", mid, "--initiating-record-ids", str(ids_path),
        "--output-stem", f"market_{mid}_ordinary_control_{suffix}",
    ], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    generated = ROOT / f"v4_diagnostics/push_follow_reverse_prototype/market_{mid}_ordinary_control_{suffix}_events.csv"
    feature_path = OUT / f"market_{mid}_ordinary_control_{suffix}_events.csv"; generated.replace(feature_path)
    distances = sorted(x["absolute_time_distance_seconds"] for x in matches)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "status": "prototype_built_not_frozen",
        "market_id": mid, "p99_events": len(large), "matched_controls": len(matches),
        "match_rate": len(matches) / len(large) if large else None,
        "caliper_seconds": CALIPER_SECONDS, "exact_match_fields": ["market_id", "yes_equivalent_direction", "kickoff_phase", "yes_price_decile"],
        "large_event_exclusion_buffer_seconds": buffer_seconds,
        "without_replacement": True, "same_timestamp_controls_excluded": True,
        "median_absolute_time_distance_seconds": distances[len(distances)//2] if distances else None,
        "maximum_absolute_time_distance_seconds": max(distances) if distances else None,
        "match_table": str(match_path.relative_to(ROOT)), "control_features": str(feature_path.relative_to(ROOT)),
    }
    (OUT / f"market_{mid}_{suffix}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
