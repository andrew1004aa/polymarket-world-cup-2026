#!/usr/bin/env python3
"""Audit whether canonical trades support the proposed v4 behavioural design.

The audit is streaming and resumable. It never modifies canonical trade files.
Each market produces an independent JSON checkpoint before the combined report
is rebuilt.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "regression_ready/trade_files.csv"
OUT = ROOT / "v4_diagnostics/behavioral_feasibility"
CHECKPOINTS = OUT / "checkpoints"
REQUIRED = {
    "trade_record_id", "timestamp", "transaction_hash", "wallet_address",
    "side", "outcome", "condition_id", "market_id", "market_type",
    "executed_token_price", "size_shares", "trade_value_usdc",
    "yes_equivalent_price",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def direction(side: str, outcome: str) -> int | None:
    pair = (side.strip().upper(), outcome.strip().upper())
    if pair not in {("BUY", "YES"), ("BUY", "NO"), ("SELL", "YES"), ("SELL", "NO")}:
        return None
    return 1 if (pair[0] == "BUY") == (pair[1] == "YES") else -1


def load_manifest() -> list[dict[str, str]]:
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def audit_market(entry: dict[str, str], force: bool) -> dict:
    market_id = entry["market_id"]
    source = ROOT / entry["path"]
    checkpoint = CHECKPOINTS / f"{market_id}.json"
    source_hash = sha256(source)
    if checkpoint.exists() and not force:
        prior = json.loads(checkpoint.read_text())
        if prior.get("status") == "complete" and prior.get("source_sha256") == source_hash:
            return prior

    missing_values = Counter()
    pairs = Counter()
    wallets: dict[str, dict[str, object]] = defaultdict(
        lambda: {"directions": set(), "rows": 0, "value": 0.0}
    )
    hashes = set()
    rows = invalid_direction = duplicate_record_ids = missing_match_event_ids = 0
    record_ids = set()
    timestamp_min = timestamp_max = None

    with gzip.open(source, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing_columns = sorted(REQUIRED - fields)
        if missing_columns:
            raise RuntimeError(f"{market_id}: missing columns {missing_columns}")
        for row in reader:
            rows += 1
            for field in REQUIRED:
                if not row[field].strip():
                    missing_values[field] += 1
            if row["market_type"].strip().lower() == "match" and not row.get("event_id", "").strip():
                missing_match_event_ids += 1
            record_id = row["trade_record_id"]
            if record_id in record_ids:
                duplicate_record_ids += 1
            record_ids.add(record_id)
            hashes.add(row["transaction_hash"].strip().lower())
            pair = (row["side"].strip().upper(), row["outcome"].strip().upper())
            pairs["/".join(pair)] += 1
            signed_direction = direction(*pair)
            if signed_direction is None:
                invalid_direction += 1
            wallet = row["wallet_address"].strip().lower()
            if wallet:
                wallet_state = wallets[wallet]
                wallet_state["rows"] = int(wallet_state["rows"]) + 1
                wallet_state["value"] = float(wallet_state["value"]) + float(row["trade_value_usdc"] or 0)
                if signed_direction is not None:
                    cast_directions = wallet_state["directions"]
                    assert isinstance(cast_directions, set)
                    cast_directions.add(signed_direction)
            timestamp = int(row["timestamp"])
            timestamp_min = timestamp if timestamp_min is None else min(timestamp_min, timestamp)
            timestamp_max = timestamp if timestamp_max is None else max(timestamp_max, timestamp)

    two_sided_wallets = sum(1 for value in wallets.values() if len(value["directions"]) == 2)
    repeated_wallets = sum(1 for value in wallets.values() if int(value["rows"]) >= 2)
    result = {
        "status": "complete",
        "market_id": market_id,
        "source_path": entry["path"],
        "source_sha256": source_hash,
        "completed_at": now(),
        "rows": rows,
        "unique_transaction_hashes": len(hashes),
        "unique_wallets": len(wallets),
        "repeated_wallets": repeated_wallets,
        "two_sided_wallets": two_sided_wallets,
        "timestamp_min": timestamp_min,
        "timestamp_max": timestamp_max,
        "side_outcome_counts": dict(sorted(pairs.items())),
        "missing_required_values": dict(sorted(missing_values.items())),
        "invalid_direction_rows": invalid_direction,
        "duplicate_trade_record_ids": duplicate_record_ids,
        "missing_match_event_ids": missing_match_event_ids,
        "supports_wallet_concentration": bool(rows and wallets),
        "supports_within_window_following": bool(timestamp_min is not None and wallets),
        "supports_observed_round_trip_proxy": bool(two_sided_wallets),
    }
    atomic_json(checkpoint, result)
    return result


def combine(results: list[dict]) -> dict:
    totals = Counter()
    pairs = Counter()
    missing = Counter()
    failed_markets = []
    for result in results:
        if result.get("status") != "complete":
            failed_markets.append(result.get("market_id"))
            continue
        for field in ("rows", "unique_transaction_hashes", "unique_wallets", "repeated_wallets",
                      "two_sided_wallets", "invalid_direction_rows", "duplicate_trade_record_ids",
                      "missing_match_event_ids"):
            totals[field] += int(result.get(field, 0))
        pairs.update(result["side_outcome_counts"])
        missing.update(result["missing_required_values"])
    summary = {
        "generated_at": now(),
        "status": "passed" if not failed_markets and not totals["invalid_direction_rows"]
                  and not totals["missing_match_event_ids"] else "review_required",
        "markets_expected": len(results),
        "markets_complete": len(results) - len(failed_markets),
        "failed_markets": failed_markets,
        "market_summed_counts": dict(totals),
        "side_outcome_counts": dict(sorted(pairs.items())),
        "missing_required_values": dict(sorted(missing.items())),
        "interpretation_limits": [
            "wallet_address is the single address exposed for each Data API trade row; counterparty roles are unavailable",
            "two-sided activity is an observed-window round-trip proxy, not proof of market making",
            "multiple addresses may share an owner and one address does not establish legal identity",
            "order submission, cancellation, spread, depth, inventory, spoofing and layering cannot be observed",
        ],
    }
    atomic_json(OUT / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-markets", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    entries = load_manifest()
    if args.max_markets is not None:
        entries = entries[:args.max_markets]
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    results = []
    for index, entry in enumerate(entries, 1):
        try:
            result = audit_market(entry, args.force)
            print(f"[{index}/{len(entries)}] market {entry['market_id']}: {result['rows']:,} rows", flush=True)
        except Exception as exc:  # keep audit resumable and report every failure
            result = {"status": "failed", "market_id": entry["market_id"], "error": repr(exc)}
            atomic_json(CHECKPOINTS / f"{entry['market_id']}.failed.json", result)
            print(f"[{index}/{len(entries)}] market {entry['market_id']}: FAILED {exc!r}", flush=True)
        results.append(result)
    summary = combine(results)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
