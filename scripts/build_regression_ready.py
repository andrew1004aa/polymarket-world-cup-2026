#!/usr/bin/env python3
"""Build a resumable, analysis-neutral regression-ready data layer.

This script performs deterministic joins and arithmetic only. It does not
classify wallets, estimate price impact, construct returns, or run statistics.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PARTITIONS = ROOT / "intermediate" / "market_partitions" / "csv"
SOURCE_MANIFEST = ROOT / "intermediate" / "market_partitions" / "market_manifest.csv"
SOURCE_MARKETS = ROOT / "intermediate" / "market_partitions" / "markets_final.csv"
SOURCE_RAW_MARKETS = ROOT / "raw" / "markets" / "markets.csv"
SOURCE_EVENTS = ROOT / "raw" / "events" / "events.csv"
SOURCE_MAPPING = ROOT / "raw" / "events" / "event_market_mapping.csv"
SOURCE_WALLETS = ROOT / "intermediate" / "wallets" / "wallets.csv"
SOURCE_WALLET_MARKET = ROOT / "intermediate" / "wallets" / "wallet_market.csv"
SOURCE_PRICES = ROOT / "raw" / "prices" / "prices.csv"

OUT = ROOT / "regression_ready"
TRADE_DIR = OUT / "trades_by_market"
PRICE_DIR = OUT / "prices_by_market"
TABLE_DIR = OUT / "tables"
PROGRESS_PATH = OUT / "progress.json"
TRADE_MANIFEST_PATH = OUT / "trade_files.csv"
PRICE_MANIFEST_PATH = OUT / "price_files.csv"
QC_PATH = OUT / "regression_ready_qc.json"

START_TS = 1780272000
END_TS = 1785542400

TRADE_FIELDS = [
    "trade_record_id", "source_row_number", "timestamp", "trade_time_utc",
    "transaction_hash", "wallet_address", "side", "outcome", "token_id",
    "condition_id", "market_id", "market_type", "market_subtype", "question",
    "country", "event_id", "fifa_match_id", "market_role", "stage",
    "home_team", "away_team", "actual_kickoff_utc", "scheduled_kickoff_utc",
    "seconds_from_actual_kickoff", "executed_token_price", "size_shares",
    "trade_value_usdc", "yes_equivalent_price", "resolution_status",
    "resolved_outcome", "yes_outcome_won", "resolved_on_timestamp",
]

PRICE_FIELDS = [
    "price_record_id", "market_id", "condition_id", "market_type", "question",
    "outcome", "token_id", "timestamp", "timestamp_utc", "price",
    "yes_equivalent_price", "fidelity_minutes",
]


def utc_iso(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc(value: str) -> int | None:
    value = value.strip()
    if not value:
        return None
    if value.endswith(" UTC"):
        parsed = datetime.strptime(value, "%Y-%m-%d %H:%M:%S.%f UTC").replace(tzinfo=timezone.utc)
    else:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return int(parsed.timestamp())


def decimal_text(value: Decimal) -> str:
    return format(value, "f")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_progress() -> dict:
    if PROGRESS_PATH.exists():
        return json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    return {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_trade_markets": {},
        "completed_price_markets": {},
        "completed_static_tables": {},
        "failed": [],
    }


def save_progress(progress: dict) -> None:
    progress["updated_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(PROGRESS_PATH, progress)


def load_context():
    markets = read_csv(SOURCE_MARKETS)
    if len(markets) != 360:
        raise ValueError(f"Expected 360 final markets, found {len(markets)}")
    by_market = {row["market_id"]: row for row in markets}
    if len(by_market) != 360:
        raise ValueError("Duplicate market_id in final markets")

    mapping = read_csv(SOURCE_MAPPING)
    by_mapping = {row["market_id"]: row for row in mapping}
    if len(mapping) != 312 or len(by_mapping) != 312:
        raise ValueError("Event mapping must contain 312 unique match markets")

    events = read_csv(SOURCE_EVENTS)
    by_event = {row["event_id"]: row for row in events}
    if len(events) != 104 or len(by_event) != 104:
        raise ValueError("Event table must contain 104 unique events")

    raw = {row["market_id"]: row for row in read_csv(SOURCE_RAW_MARKETS)}
    manifest = read_csv(SOURCE_MANIFEST)
    if len(manifest) != 360:
        raise ValueError("Trade manifest must contain 360 markets")
    return markets, by_market, by_mapping, by_event, raw, manifest


def build_trade_files(progress: dict, by_market, by_mapping, by_event, manifest) -> None:
    TRADE_DIR.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(manifest, 1):
        market_id = item["market_id"]
        source = SOURCE_PARTITIONS / f"{market_id}.csv.gz"
        target = TRADE_DIR / f"{market_id}.csv.gz"
        prior = progress["completed_trade_markets"].get(market_id)
        if prior and target.exists() and prior.get("source_sha256") == item["sha256"]:
            print(f"[trades {index}/360] {market_id} checkpointed", flush=True)
            continue

        market = by_market[market_id]
        maprow = by_mapping.get(market_id)
        event = by_event.get(maprow["event_id"]) if maprow else None
        kickoff_ts = parse_utc(event["actual_kickoff_utc"]) if event else None
        temp = target.with_suffix(target.suffix + ".tmp")
        rows = 0
        min_ts = None
        max_ts = None
        value_sum = Decimal(0)
        errors = []
        try:
            with gzip.open(source, "rt", newline="", encoding="utf-8-sig") as inp, gzip.open(
                temp, "wt", newline="", encoding="utf-8", compresslevel=6
            ) as out:
                reader = csv.DictReader(inp)
                writer = csv.DictWriter(out, fieldnames=TRADE_FIELDS)
                writer.writeheader()
                for source_row_number, row in enumerate(reader, 1):
                    ts = int(row["timestamp"])
                    price = Decimal(row["price"])
                    size = Decimal(row["size"])
                    if not START_TS <= ts < END_TS:
                        raise ValueError(f"timestamp outside study interval at row {source_row_number}")
                    if not Decimal(0) <= price <= Decimal(1) or size <= 0:
                        raise ValueError(f"invalid price/size at row {source_row_number}")
                    outcome = row["outcome"].strip().upper()
                    yes_price = price if outcome == "YES" else Decimal(1) - price
                    value = price * size
                    value_sum += value
                    min_ts = ts if min_ts is None else min(min_ts, ts)
                    max_ts = ts if max_ts is None else max(max_ts, ts)
                    rows += 1
                    writer.writerow({
                        "trade_record_id": f"{market_id}:{source_row_number}",
                        "source_row_number": source_row_number,
                        "timestamp": ts,
                        "trade_time_utc": utc_iso(ts),
                        "transaction_hash": row["transactionHash"].lower(),
                        "wallet_address": row["proxyWallet"].lower(),
                        "side": row["side"].upper(),
                        "outcome": outcome,
                        "token_id": row["asset"],
                        "condition_id": row["conditionId"].lower(),
                        "market_id": market_id,
                        "market_type": market["market_type"],
                        "market_subtype": market["market_subtype"],
                        "question": market["question"],
                        "country": market["country"],
                        "event_id": maprow["event_id"] if maprow else "",
                        "fifa_match_id": maprow["fifa_match_id"] if maprow else "",
                        "market_role": maprow["market_role"] if maprow else "",
                        "stage": event["stage"] if event else "",
                        "home_team": event["home_team"] if event else "",
                        "away_team": event["away_team"] if event else "",
                        "actual_kickoff_utc": event["actual_kickoff_utc"] if event else "",
                        "scheduled_kickoff_utc": event["scheduled_kickoff_utc"] if event else "",
                        "seconds_from_actual_kickoff": ts - kickoff_ts if kickoff_ts is not None else "",
                        "executed_token_price": decimal_text(price),
                        "size_shares": decimal_text(size),
                        "trade_value_usdc": decimal_text(value),
                        "yes_equivalent_price": decimal_text(yes_price),
                        "resolution_status": market["resolution_status"],
                        "resolved_outcome": market["resolved_outcome"],
                        "yes_outcome_won": "1" if market["resolved_outcome"] == "Yes" else "0",
                        "resolved_on_timestamp": market["resolved_on_timestamp"],
                    })
            if rows != int(item["row_count"]):
                raise ValueError(f"row count {rows} != manifest {item['row_count']}")
            temp.replace(target)
            progress["completed_trade_markets"][market_id] = {
                "path": str(target.relative_to(ROOT)),
                "rows": rows,
                "first_timestamp": min_ts,
                "last_timestamp": max_ts,
                "trade_value_usdc": decimal_text(value_sum),
                "source_sha256": item["sha256"],
                "sha256": sha256(target),
            }
            save_progress(progress)
            print(f"[trades {index}/360] {market_id} {rows:,} rows", flush=True)
        except Exception as exc:
            temp.unlink(missing_ok=True)
            errors.append(repr(exc))
            progress["failed"].append({"stage": "trades", "market_id": market_id, "error": repr(exc)})
            save_progress(progress)
            raise


def build_price_files(progress: dict, by_market) -> None:
    PRICE_DIR.mkdir(parents=True, exist_ok=True)
    completed = progress["completed_price_markets"]
    current_market = None
    handle = None
    writer = None
    temp = None
    rows = 0
    value_min = None
    value_max = None
    seen_closed = set()
    skipping_completed = False

    def close_current():
        nonlocal handle, writer, temp, rows, value_min, value_max, current_market, skipping_completed
        if current_market is None:
            return
        target = PRICE_DIR / f"{current_market}.csv.gz"
        if skipping_completed:
            expected = completed[current_market]
            if rows != int(expected["rows"]):
                raise ValueError(f"checkpointed price row count changed for {current_market}")
            print(f"[prices] {current_market} checkpointed", flush=True)
        else:
            handle.close()
            temp.replace(target)
            completed[current_market] = {
                "path": str(target.relative_to(ROOT)),
                "rows": rows,
                "first_timestamp": value_min,
                "last_timestamp": value_max,
                "sha256": sha256(target),
            }
            save_progress(progress)
            print(f"[prices] {current_market} {rows:,} rows", flush=True)
        seen_closed.add(current_market)
        handle = writer = temp = None
        rows = 0
        value_min = value_max = None
        skipping_completed = False

    with SOURCE_PRICES.open(newline="", encoding="utf-8-sig") as inp:
        reader = csv.DictReader(inp)
        for row_number, row in enumerate(reader, 1):
            market_id = row["market_id"]
            if market_id not in by_market:
                raise ValueError(f"Unknown final market in prices.csv: {market_id}")
            if market_id != current_market:
                close_current()
                if market_id in seen_closed:
                    raise ValueError(f"prices.csv is not grouped by market_id; {market_id} reappeared")
                current_market = market_id
                target = PRICE_DIR / f"{market_id}.csv.gz"
                skipping_completed = market_id in completed and target.exists()
                if not skipping_completed:
                    temp = target.with_suffix(target.suffix + ".tmp")
                    handle = gzip.open(temp, "wt", newline="", encoding="utf-8", compresslevel=6)
                    writer = csv.DictWriter(handle, fieldnames=PRICE_FIELDS)
                    writer.writeheader()
            ts = int(row["timestamp"])
            price = Decimal(row["price"])
            outcome = row["outcome"].upper()
            if not START_TS <= ts < END_TS or not Decimal(0) <= price <= Decimal(1):
                raise ValueError(f"Invalid price row {row_number}")
            yes_price = price if outcome == "YES" else Decimal(1) - price
            rows += 1
            value_min = ts if value_min is None else min(value_min, ts)
            value_max = ts if value_max is None else max(value_max, ts)
            if not skipping_completed:
                writer.writerow({
                    "price_record_id": f"{row['token_id']}:{ts}",
                    "market_id": market_id,
                    "condition_id": row["condition_id"].lower(),
                    "market_type": row["market_type"],
                    "question": row["question"],
                    "outcome": outcome,
                    "token_id": row["token_id"],
                    "timestamp": ts,
                    "timestamp_utc": row["timestamp_utc"],
                    "price": decimal_text(price),
                    "yes_equivalent_price": decimal_text(yes_price),
                    "fidelity_minutes": row["fidelity_minutes"],
                })
        close_current()
    if len(completed) != 360:
        raise ValueError(f"Expected 360 completed price markets, got {len(completed)}")


def gzip_copy(source: Path, target: Path) -> dict:
    temp = target.with_suffix(target.suffix + ".tmp")
    rows = 0
    with source.open("rb") as inp, gzip.open(temp, "wb", compresslevel=6) as out:
        header = True
        for line in inp:
            out.write(line)
            if header:
                header = False
            else:
                rows += 1
    temp.replace(target)
    return {"path": str(target.relative_to(ROOT)), "rows": rows, "sha256": sha256(target)}


def copy_csv(source: Path, target: Path) -> dict:
    temp = target.with_suffix(target.suffix + ".tmp")
    shutil.copyfile(source, temp)
    temp.replace(target)
    with target.open("rb") as handle:
        rows = max(0, sum(1 for _ in handle) - 1)
    return {"path": str(target.relative_to(ROOT)), "rows": rows, "sha256": sha256(target)}


def build_static_tables(progress: dict) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    specs = [
        ("markets", SOURCE_MARKETS, TABLE_DIR / "markets.csv", copy_csv),
        ("events", SOURCE_EVENTS, TABLE_DIR / "events.csv", copy_csv),
        ("event_market_mapping", SOURCE_MAPPING, TABLE_DIR / "event_market_mapping.csv", copy_csv),
        ("wallets", SOURCE_WALLETS, TABLE_DIR / "wallets.csv.gz", gzip_copy),
        ("wallet_market", SOURCE_WALLET_MARKET, TABLE_DIR / "wallet_market.csv.gz", gzip_copy),
    ]
    for name, source, target, operation in specs:
        prior = progress["completed_static_tables"].get(name)
        if prior and target.exists():
            print(f"[static] {name} checkpointed", flush=True)
            continue
        progress["completed_static_tables"][name] = operation(source, target)
        save_progress(progress)
        print(f"[static] {name}", flush=True)


def write_manifests(progress: dict) -> None:
    for path, key in [
        (TRADE_MANIFEST_PATH, "completed_trade_markets"),
        (PRICE_MANIFEST_PATH, "completed_price_markets"),
    ]:
        rows = progress[key]
        fields = ["market_id", "path", "rows", "first_timestamp", "last_timestamp", "sha256"]
        if key == "completed_trade_markets":
            fields.insert(5, "trade_value_usdc")
            fields.insert(6, "source_sha256")
        temp = path.with_suffix(path.suffix + ".tmp")
        with temp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for market_id in sorted(rows, key=lambda x: int(x)):
                writer.writerow({"market_id": market_id, **rows[market_id]})
        temp.replace(path)


def build_qc(progress: dict, manifest) -> dict:
    trade = progress["completed_trade_markets"]
    prices = progress["completed_price_markets"]
    static = progress["completed_static_tables"]
    expected_trade_rows = sum(int(row["row_count"]) for row in manifest)
    trade_rows = sum(int(row["rows"]) for row in trade.values())
    price_rows = sum(int(row["rows"]) for row in prices.values())
    errors = []
    checksum_errors = []
    if len(trade) != 360: errors.append(f"trade market files: {len(trade)} != 360")
    if trade_rows != expected_trade_rows: errors.append(f"trade rows: {trade_rows} != {expected_trade_rows}")
    if len(prices) != 360: errors.append(f"price market files: {len(prices)} != 360")
    if price_rows != 17980605: errors.append(f"price rows: {price_rows} != 17980605")
    for name, expected in {"markets":360,"events":104,"event_market_mapping":312,"wallets":335414,"wallet_market":2497766}.items():
        if name not in static: errors.append(f"missing static table: {name}")
        elif int(static[name]["rows"]) != expected: errors.append(f"{name} rows: {static[name]['rows']} != {expected}")
    for stage_name, records in (("trade", trade), ("price", prices), ("static", static)):
        for record_id, record in records.items():
            target = ROOT / record["path"]
            if not target.exists():
                checksum_errors.append(f"{stage_name}:{record_id}: missing {record['path']}")
            elif sha256(target) != record["sha256"]:
                checksum_errors.append(f"{stage_name}:{record_id}: SHA-256 mismatch")
    errors.extend(checksum_errors)
    qc = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "study_interval": {"start_utc_inclusive":"2026-06-01T00:00:00Z","end_utc_exclusive":"2026-08-01T00:00:00Z"},
        "trade_market_files": len(trade),
        "trade_rows": trade_rows,
        "expected_trade_rows": expected_trade_rows,
        "trade_value_usdc": decimal_text(sum(Decimal(row["trade_value_usdc"]) for row in trade.values())),
        "price_market_files": len(prices),
        "price_rows": price_rows,
        "verified_checksums": len(trade) + len(prices) + len(static) - len(checksum_errors),
        "checksum_errors": checksum_errors,
        "static_tables": static,
        "failed_records_in_progress": progress["failed"],
        "errors": errors,
        "notes": [
            "All output variables are direct source fields, deterministic joins, timestamp conversions, or simple arithmetic.",
            "No whale threshold, price impact, return, volatility, regression, hypothesis test, or inference is included.",
            "Historical order-book fields are absent and were not imputed.",
        ],
    }
    atomic_json(QC_PATH, qc)
    return qc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "trades", "prices", "static", "qc"], default="all")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    markets, by_market, by_mapping, by_event, raw, manifest = load_context()
    progress = load_progress()
    if args.stage in ("all", "trades"):
        build_trade_files(progress, by_market, by_mapping, by_event, manifest)
    if args.stage in ("all", "prices"):
        build_price_files(progress, by_market)
    if args.stage in ("all", "static"):
        build_static_tables(progress)
    write_manifests(progress)
    if args.stage in ("all", "qc"):
        qc = build_qc(progress, manifest)
        print(json.dumps(qc, indent=2), flush=True)
        return 1 if qc["errors"] else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
