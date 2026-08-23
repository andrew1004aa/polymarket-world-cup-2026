#!/usr/bin/env python3
"""Create resumable compact per-market CSV.GZ files from raw API checkpoints."""

from __future__ import annotations

import argparse
import csv
import fcntl
import gzip
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORK_DIR = ROOT / "intermediate" / "market_partitions"
RAW_DIR = ROOT / "raw" / "trades" / "data_api"
WHITELIST = WORK_DIR / "market_whitelist.csv"
CSV_DIR = WORK_DIR / "csv"
LOG_DIR = WORK_DIR / "logs"
PROGRESS_PATH = WORK_DIR / "progress.json"
MANIFEST_PATH = WORK_DIR / "market_manifest.csv"
QC_PATH = WORK_DIR / "quality_control_report.md"
LOCK_PATH = WORK_DIR / "partition.lock"

START = 1780272000  # 2026-06-01T00:00:00Z
END = 1785542400  # 2026-08-01T00:00:00Z, end exclusive

FIELDS = (
    "timestamp",
    "transactionHash",
    "proxyWallet",
    "side",
    "outcome",
    "asset",
    "conditionId",
    "market_id",
    "market_type",
    "title",
    "price",
    "size",
)

MANIFEST_FIELDS = (
    "market_id",
    "condition_id",
    "market_type",
    "title",
    "row_count",
    "first_timestamp",
    "last_timestamp",
    "exact_duplicates_removed",
    "missing_timestamp",
    "missing_condition_id",
    "missing_asset",
    "missing_wallet",
    "missing_transaction_hash",
    "source_checkpoint_count",
    "csv_file",
    "sha256",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json_atomic(path: Path, data: Any) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    temporary.replace(path)


def load_progress() -> dict[str, Any]:
    if PROGRESS_PATH.exists():
        return json.loads(PROGRESS_PATH.read_text())
    return {
        "schema_version": 1,
        "start": START,
        "end_exclusive": END,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "completed_markets": {},
        "failed_markets": {},
    }


def load_whitelist() -> list[dict[str, str]]:
    with WHITELIST.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 360:
        raise ValueError(f"Expected 360 whitelist rows, got {len(rows)}")
    return rows


def read_checkpoint(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        body = json.load(handle)
    request = body.get("request") or {}
    response = body.get("response") or []
    if not isinstance(response, list):
        raise ValueError(f"Response is not a list: {path}")
    return request, response


def index_checkpoints(
    wanted: set[str],
) -> tuple[dict[str, list[Path]], list[dict[str, str]]]:
    index: dict[str, list[Path]] = defaultdict(list)
    failures: list[dict[str, str]] = []
    by_prefix: dict[str, str] = {}
    for condition_id in wanted:
        prefix = condition_id[:18]
        if prefix in by_prefix:
            raise ValueError(
                f"Condition ID prefix collision: {condition_id} and "
                f"{by_prefix[prefix]}"
            )
        by_prefix[prefix] = condition_id

    paths = sorted(RAW_DIR.rglob("*.json.gz"))
    total = len(paths)
    for number, path in enumerate(paths, 1):
        if number % 500 == 0 or number == total:
            print(f"Indexing checkpoints: {number:,}/{total:,}", flush=True)
        condition_id = by_prefix.get(path.parent.name.lower())
        if condition_id:
            index[condition_id].append(path)
    return index, failures


def exact_row_key(row: dict[str, Any]) -> bytes:
    canonical = json.dumps(
        row, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).digest()


def partition_market(
    market: dict[str, str], checkpoints: list[Path]
) -> dict[str, Any]:
    market_id = market["market_id"]
    condition_id = market["condition_id"].lower()
    output_path = CSV_DIR / f"{market_id}.csv.gz"
    temporary = output_path.with_suffix(f".csv.gz.tmp.{os.getpid()}")

    seen: set[bytes] = set()
    stats: dict[str, Any] = {
        "market_id": market_id,
        "condition_id": condition_id,
        "market_type": market["market_type"],
        "title": market["question"],
        "row_count": 0,
        "first_timestamp": None,
        "last_timestamp": None,
        "exact_duplicates_removed": 0,
        "missing_timestamp": 0,
        "missing_condition_id": 0,
        "missing_asset": 0,
        "missing_wallet": 0,
        "missing_transaction_hash": 0,
        "source_checkpoint_count": len(checkpoints),
        "csv_file": str(output_path.relative_to(WORK_DIR)),
    }

    with gzip.open(temporary, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()

        for checkpoint in sorted(checkpoints):
            _, rows = read_checkpoint(checkpoint)
            for raw in rows:
                timestamp_value = raw.get("timestamp")
                try:
                    timestamp = int(timestamp_value)
                except (TypeError, ValueError):
                    stats["missing_timestamp"] += 1
                    continue
                if not START <= timestamp < END:
                    continue
                if str(raw.get("conditionId") or "").lower() != condition_id:
                    continue

                row_key = exact_row_key(raw)
                if row_key in seen:
                    stats["exact_duplicates_removed"] += 1
                    continue
                seen.add(row_key)

                if not raw.get("conditionId"):
                    stats["missing_condition_id"] += 1
                if not raw.get("asset"):
                    stats["missing_asset"] += 1
                if not raw.get("proxyWallet"):
                    stats["missing_wallet"] += 1
                if not raw.get("transactionHash"):
                    stats["missing_transaction_hash"] += 1

                output = {field: raw.get(field, "") for field in FIELDS}
                output["market_id"] = market_id
                output["market_type"] = market["market_type"]
                writer.writerow(output)

                stats["row_count"] += 1
                if (
                    stats["first_timestamp"] is None
                    or timestamp < stats["first_timestamp"]
                ):
                    stats["first_timestamp"] = timestamp
                if (
                    stats["last_timestamp"] is None
                    or timestamp > stats["last_timestamp"]
                ):
                    stats["last_timestamp"] = timestamp

    temporary.replace(output_path)
    digest = hashlib.sha256()
    with output_path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    stats["sha256"] = digest.hexdigest()
    return stats


def write_manifest(completed: dict[str, Any]) -> None:
    rows = sorted(
        completed.values(), key=lambda row: (row["market_type"], int(row["market_id"]))
    )
    temporary = MANIFEST_PATH.with_suffix(f".csv.tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in MANIFEST_FIELDS}
            for row in rows
        )
    temporary.replace(MANIFEST_PATH)


def write_qc(progress: dict[str, Any], total_markets: int) -> None:
    rows = list(progress["completed_markets"].values())
    totals = {
        field: sum(int(row.get(field) or 0) for row in rows)
        for field in (
            "row_count",
            "exact_duplicates_removed",
            "missing_timestamp",
            "missing_condition_id",
            "missing_asset",
            "missing_wallet",
            "missing_transaction_hash",
        )
    }
    text = f"""# Market Partition Quality-Control Report

- Generated: {utc_now()}
- Exact interval: 2026-06-01T00:00:00Z to 2026-08-01T00:00:00Z (end exclusive)
- Whitelisted markets: {total_markets}
- Completed market files: {len(rows)}
- Included trade rows: {totals['row_count']}
- Exact duplicate rows removed: {totals['exact_duplicates_removed']}
- Missing timestamps: {totals['missing_timestamp']}
- Missing condition IDs: {totals['missing_condition_id']}
- Missing token IDs: {totals['missing_asset']}
- Missing wallets: {totals['missing_wallet']}
- Missing transaction hashes: {totals['missing_transaction_hash']}
- Failed markets: {len(progress['failed_markets'])}

The dataset was partitioned without statistical analysis or market interpretation.
"""
    QC_PATH.write_text(text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retry-failed", action="store_true", help="Retry previously failed markets"
    )
    args = parser.parse_args()

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    lock_handle = LOCK_PATH.open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise SystemExit(
            "Another partition_trades_by_market.py process is already running. "
            "Do not start a second copy."
        ) from error
    lock_handle.seek(0)
    lock_handle.truncate()
    lock_handle.write(f"pid={os.getpid()} started={utc_now()}\n")
    lock_handle.flush()

    whitelist = load_whitelist()
    wanted = {row["condition_id"].lower() for row in whitelist}
    progress = load_progress()

    if progress.get("start") != START or progress.get("end_exclusive") != END:
        raise ValueError("Existing progress.json uses a different date interval")

    print(f"Whitelisted markets: {len(whitelist)}", flush=True)
    checkpoint_index, index_failures = index_checkpoints(wanted)
    if index_failures:
        save_json_atomic(LOG_DIR / "checkpoint_read_failures.json", index_failures)
        raise RuntimeError(
            f"{len(index_failures)} raw checkpoints could not be read; see logs"
        )

    missing_sources = sorted(wanted - checkpoint_index.keys())
    if missing_sources:
        save_json_atomic(LOG_DIR / "missing_checkpoint_markets.json", missing_sources)
        raise RuntimeError(
            f"{len(missing_sources)} whitelisted markets have no raw checkpoints"
        )

    for number, market in enumerate(whitelist, 1):
        market_id = market["market_id"]
        output_path = CSV_DIR / f"{market_id}.csv.gz"
        completed = progress["completed_markets"].get(market_id)
        if completed and output_path.exists():
            print(
                f"[{number}/{len(whitelist)}] skip completed {market_id}",
                flush=True,
            )
            continue
        if market_id in progress["failed_markets"] and not args.retry_failed:
            print(
                f"[{number}/{len(whitelist)}] skip failed {market_id}; "
                f"use --retry-failed",
                flush=True,
            )
            continue

        print(
            f"[{number}/{len(whitelist)}] {market_id} {market['question']}",
            flush=True,
        )
        try:
            stats = partition_market(
                market, checkpoint_index[market["condition_id"].lower()]
            )
            progress["completed_markets"][market_id] = stats
            progress["failed_markets"].pop(market_id, None)
            print(
                f"  wrote {stats['row_count']:,} rows; "
                f"duplicates={stats['exact_duplicates_removed']:,}",
                flush=True,
            )
        except Exception as error:
            progress["failed_markets"][market_id] = {
                "error": repr(error),
                "updated_at": utc_now(),
            }
            print(f"  FAILED: {error!r}", flush=True)

        progress["updated_at"] = utc_now()
        save_json_atomic(PROGRESS_PATH, progress)
        write_manifest(progress["completed_markets"])
        write_qc(progress, len(whitelist))

    write_manifest(progress["completed_markets"])
    write_qc(progress, len(whitelist))
    print(
        json.dumps(
            {
                "completed_markets": len(progress["completed_markets"]),
                "failed_markets": len(progress["failed_markets"]),
                "trade_rows": sum(
                    row["row_count"]
                    for row in progress["completed_markets"].values()
                ),
            },
            indent=2,
        ),
        flush=True,
    )
    return 1 if progress["failed_markets"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
