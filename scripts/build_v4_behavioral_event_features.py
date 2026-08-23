#!/usr/bin/env python3
"""Run the validated push-follow-reverse builder for all 360 markets.

Every market is checkpointed independently. Re-running skips outputs whose
source trade hash and P99 threshold are unchanged. Canonical inputs and v3
artifacts are never modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "regression_ready/trade_files.csv"
THRESHOLDS = ROOT / "large_trade_diagnostics/v3/market_thresholds.csv"
PROTOTYPE = ROOT / "scripts/build_v4_push_follow_reverse_prototype.py"
PROTOTYPE_OUT = ROOT / "v4_diagnostics/push_follow_reverse_prototype"
OUT = ROOT / "analysis_ready/v4/behavioral_events"
CHECKPOINTS = OUT / "checkpoints"
LOCK = OUT / ".builder.lock"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def thresholds() -> dict[str, str]:
    return {row["market_id"]: row["p99_threshold_usdc"] for row in load_rows(THRESHOLDS)}


def process(entry: dict[str, str], threshold: str, force: bool) -> dict:
    market_id = entry["market_id"]
    source = ROOT / entry["path"]
    source_hash = sha256(source)
    checkpoint = CHECKPOINTS / f"{market_id}.json"
    target = OUT / f"{market_id}.csv"
    if checkpoint.exists() and target.exists() and not force:
        prior = json.loads(checkpoint.read_text())
        if (prior.get("status") == "complete" and prior.get("source_trade_sha256") == source_hash
                and prior.get("p99_threshold_usdc") == threshold
                and prior.get("output_sha256") == sha256(target)):
            return prior

    subprocess.run(
        [sys.executable, str(PROTOTYPE), "--market-id", market_id],
        cwd=ROOT, check=True, stdout=subprocess.DEVNULL,
    )
    generated = PROTOTYPE_OUT / f"market_{market_id}_events.csv"
    generated_summary = PROTOTYPE_OUT / f"market_{market_id}_summary.json"
    temp = target.with_suffix(".csv.tmp")
    generated.replace(temp)
    temp.replace(target)
    prototype_summary = json.loads(generated_summary.read_text())
    meta = {
        "status": "complete", "market_id": market_id,
        "source_trade_path": entry["path"], "source_trade_sha256": source_hash,
        "p99_threshold_usdc": threshold,
        "canonical_trade_rows": int(entry["rows"]),
        "initiating_p99_events": prototype_summary["initiating_p99_events"],
        "events_with_baseline_price": prototype_summary["events_with_baseline_price"],
        "events_with_5m_price": prototype_summary["events_with_5m_price"],
        "output_path": str(target.relative_to(ROOT)), "output_sha256": sha256(target),
        "completed_at": now(),
    }
    atomic_json(checkpoint, meta)
    return meta


def combine(results: list[dict]) -> dict:
    expected_events = sum(int(result["initiating_p99_events"]) for result in results)
    threshold_rows = load_rows(THRESHOLDS)
    declared_threshold_events = sum(int(row["p99_trade_count"]) for row in threshold_rows)
    actual_by_market = {str(row["market_id"]): int(row["initiating_p99_events"]) for row in results}
    threshold_count_discrepancies = [
        {
            "market_id": row["market_id"],
            "threshold_diagnostic_count": int(row["p99_trade_count"]),
            "csv_reloaded_float_classification_count": actual_by_market[row["market_id"]],
            "difference": int(row["p99_trade_count"]) - actual_by_market[row["market_id"]],
        }
        for row in threshold_rows
        if int(row["p99_trade_count"]) != actual_by_market[row["market_id"]]
    ]
    target = OUT / "behavioral_p99_events_all.csv"
    temporary = target.with_suffix(".csv.tmp")
    rows = 0; fieldnames = None
    with temporary.open("w", encoding="utf-8", newline="") as destination:
        writer = None
        for result in sorted(results, key=lambda x: int(x["market_id"])):
            path = ROOT / result["output_path"]
            with path.open(encoding="utf-8", newline="") as source:
                reader = csv.DictReader(source)
                if writer is None:
                    fieldnames = reader.fieldnames
                    writer = csv.DictWriter(destination, fieldnames=fieldnames)
                    writer.writeheader()
                elif reader.fieldnames != fieldnames:
                    raise RuntimeError(f"Header mismatch for market {result['market_id']}")
                for row in reader:
                    writer.writerow(row); rows += 1
    if rows != expected_events:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Combined rows {rows} != checkpoint events {expected_events}")
    temporary.replace(target)
    summary = {
        "generated_at": now(), "status": "built_qc_passed",
        "markets_expected": 360, "markets_complete": len(results),
        "canonical_trade_rows": sum(int(x["canonical_trade_rows"]) for x in results),
        "p99_initiating_events": rows,
        "threshold_diagnostic_p99_events": declared_threshold_events,
        "threshold_count_discrepancies": threshold_count_discrepancies,
        "events_with_baseline_price": sum(int(x["events_with_baseline_price"]) for x in results),
        "events_with_5m_price": sum(int(x["events_with_5m_price"]) for x in results),
        "output_path": str(target.relative_to(ROOT)), "output_sha256": sha256(target),
        "notes": [
            "Only within-market P99 initiating trades are included in this event table.",
            "Follower measures exclude the initiating wallet and trades at the identical timestamp.",
            "This table is not yet a matched large-versus-ordinary causal sample.",
            "Two threshold-tie rows in market 1897089 differ because the NumPy quantile was serialized to CSV and reloaded as a float; v4 preserves the CSV-reloaded classification actually used by frozen v3.",
        ],
    }
    atomic_json(OUT / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-markets", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    try:
        LOCK.mkdir()
    except FileExistsError:
        raise RuntimeError(
            f"Another builder appears active ({LOCK}). Stop it first, or remove the lock only after verifying no builder is running."
        )
    (LOCK / "owner.json").write_text(json.dumps({"pid": os.getpid(), "started_at": now()}) + "\n")
    try:
        entries = load_rows(MANIFEST); threshold_map = thresholds()
        if args.max_markets is not None:
            entries = entries[:args.max_markets]
        results = []
        for index, entry in enumerate(entries, 1):
            market_id = entry["market_id"]
            if args.build_only:
                checkpoint = CHECKPOINTS / f"{market_id}.json"
                if not checkpoint.exists():
                    raise RuntimeError(f"Missing checkpoint for market {market_id}")
                result = json.loads(checkpoint.read_text())
            else:
                result = process(entry, threshold_map[market_id], args.force)
            results.append(result)
            print(f"[{index}/{len(entries)}] {market_id}: {result['initiating_p99_events']:,} events", flush=True)
        if len(entries) == 360:
            print(json.dumps(combine(results), indent=2), flush=True)
        else:
            print(json.dumps({"status": "partial", "markets": len(results),
                              "events": sum(int(x["initiating_p99_events"]) for x in results)}, indent=2))
        return 0
    finally:
        (LOCK / "owner.json").unlink(missing_ok=True)
        LOCK.rmdir()


if __name__ == "__main__":
    raise SystemExit(main())
