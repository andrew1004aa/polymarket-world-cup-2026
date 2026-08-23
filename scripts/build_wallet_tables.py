#!/usr/bin/env python3
"""Build resumable wallet and wallet-market tables from partitioned trades."""

from __future__ import annotations

import argparse
import csv
import fcntl
import gzip
import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARTITION_DIR = ROOT / "intermediate" / "market_partitions"
SOURCE_DIR = PARTITION_DIR / "csv"
MANIFEST_PATH = PARTITION_DIR / "market_manifest.csv"
OUTPUT_DIR = ROOT / "intermediate" / "wallets"
DB_PATH = OUTPUT_DIR / "wallet_aggregation.sqlite3"
WALLETS_PATH = OUTPUT_DIR / "wallets.csv"
WALLET_MARKET_PATH = OUTPUT_DIR / "wallet_market.csv"
QC_PATH = OUTPUT_DIR / "wallet_tables_qc.json"
LOCK_PATH = OUTPUT_DIR / "build_wallet_tables.lock"


@dataclass
class Aggregate:
    trade_count: int = 0
    buy_count: int = 0
    sell_count: int = 0
    total_shares: Decimal = Decimal(0)
    total_volume_usdc: Decimal = Decimal(0)
    first_trade: int | None = None
    last_trade: int | None = None

    def add(self, side: str, size: Decimal, price: Decimal, timestamp: int) -> None:
        self.trade_count += 1
        if side.upper() == "BUY":
            self.buy_count += 1
        elif side.upper() == "SELL":
            self.sell_count += 1
        self.total_shares += size
        self.total_volume_usdc += size * price
        if self.first_trade is None or timestamp < self.first_trade:
            self.first_trade = timestamp
        if self.last_trade is None or timestamp > self.last_trade:
            self.last_trade = timestamp


def utc_iso(timestamp: int | None) -> str:
    if timestamp is None:
        return ""
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def format_number(value: float) -> str:
    return format(value, ".15g")


def connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS wallets (
            wallet_address TEXT PRIMARY KEY,
            trade_count INTEGER NOT NULL,
            markets_traded INTEGER NOT NULL,
            total_shares REAL NOT NULL,
            total_volume_usdc REAL NOT NULL,
            first_trade INTEGER NOT NULL,
            last_trade INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS wallet_market (
            wallet_address TEXT NOT NULL,
            market_id TEXT NOT NULL,
            condition_id TEXT NOT NULL,
            market_type TEXT NOT NULL,
            trade_count INTEGER NOT NULL,
            buy_count INTEGER NOT NULL,
            sell_count INTEGER NOT NULL,
            total_shares REAL NOT NULL,
            total_volume_usdc REAL NOT NULL,
            first_trade INTEGER NOT NULL,
            last_trade INTEGER NOT NULL,
            PRIMARY KEY (wallet_address, market_id)
        );

        CREATE INDEX IF NOT EXISTS idx_wallet_market_market
        ON wallet_market(market_id);

        CREATE TABLE IF NOT EXISTS processed_markets (
            market_id TEXT PRIMARY KEY,
            source_sha256 TEXT NOT NULL,
            source_row_count INTEGER NOT NULL,
            wallet_count INTEGER NOT NULL,
            processed_at TEXT NOT NULL
        );
        """
    )
    return connection


def load_manifest() -> list[dict[str, str]]:
    with MANIFEST_PATH.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 360:
        raise ValueError(f"Expected 360 manifest rows, got {len(rows)}")
    return rows


def aggregate_market(path: Path, expected_condition: str) -> dict[str, Aggregate]:
    aggregates: dict[str, Aggregate] = defaultdict(Aggregate)
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), 2):
            wallet = row["proxyWallet"].strip().lower()
            if not wallet:
                raise ValueError(f"{path.name}:{row_number}: blank wallet")
            if row["conditionId"].strip().lower() != expected_condition:
                raise ValueError(f"{path.name}:{row_number}: condition ID mismatch")
            try:
                size = Decimal(row["size"])
                price = Decimal(row["price"])
                timestamp = int(row["timestamp"])
            except (InvalidOperation, ValueError) as error:
                raise ValueError(
                    f"{path.name}:{row_number}: invalid numeric value"
                ) from error
            aggregates[wallet].add(row["side"], size, price, timestamp)
    return aggregates


def store_market(
    connection: sqlite3.Connection,
    manifest: dict[str, str],
    aggregates: dict[str, Aggregate],
) -> None:
    market_id = manifest["market_id"]
    condition_id = manifest["condition_id"].lower()
    market_type = manifest["market_type"]

    wallet_rows = []
    wallet_market_rows = []
    for wallet, aggregate in aggregates.items():
        assert aggregate.first_trade is not None
        assert aggregate.last_trade is not None
        wallet_rows.append(
            (
                wallet,
                aggregate.trade_count,
                1,
                float(aggregate.total_shares),
                float(aggregate.total_volume_usdc),
                aggregate.first_trade,
                aggregate.last_trade,
            )
        )
        wallet_market_rows.append(
            (
                wallet,
                market_id,
                condition_id,
                market_type,
                aggregate.trade_count,
                aggregate.buy_count,
                aggregate.sell_count,
                float(aggregate.total_shares),
                float(aggregate.total_volume_usdc),
                aggregate.first_trade,
                aggregate.last_trade,
            )
        )

    with connection:
        connection.executemany(
            """
            INSERT INTO wallets VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(wallet_address) DO UPDATE SET
                trade_count = trade_count + excluded.trade_count,
                markets_traded = markets_traded + 1,
                total_shares = total_shares + excluded.total_shares,
                total_volume_usdc = total_volume_usdc + excluded.total_volume_usdc,
                first_trade = MIN(first_trade, excluded.first_trade),
                last_trade = MAX(last_trade, excluded.last_trade)
            """,
            wallet_rows,
        )
        connection.executemany(
            "INSERT INTO wallet_market VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            wallet_market_rows,
        )
        connection.execute(
            "INSERT INTO processed_markets VALUES (?, ?, ?, ?, ?)",
            (
                market_id,
                manifest["sha256"],
                int(manifest["row_count"]),
                len(aggregates),
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def export_wallets(connection: sqlite3.Connection) -> int:
    fields = (
        "wallet_address",
        "number_of_trades",
        "markets_traded",
        "total_shares",
        "total_volume_usdc",
        "first_trade_timestamp",
        "first_trade_utc",
        "last_trade_timestamp",
        "last_trade_utc",
    )
    temporary = WALLETS_PATH.with_suffix(".csv.tmp")
    count = 0
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        cursor = connection.execute(
            """
            SELECT wallet_address, trade_count, markets_traded, total_shares,
                   total_volume_usdc, first_trade, last_trade
            FROM wallets ORDER BY wallet_address
            """
        )
        for wallet, trades, markets, shares, volume, first, last in cursor:
            writer.writerow(
                (
                    wallet,
                    trades,
                    markets,
                    format_number(shares),
                    format_number(volume),
                    first,
                    utc_iso(first),
                    last,
                    utc_iso(last),
                )
            )
            count += 1
    temporary.replace(WALLETS_PATH)
    return count


def export_wallet_market(connection: sqlite3.Connection) -> int:
    fields = (
        "wallet_address",
        "market_id",
        "condition_id",
        "market_type",
        "trade_count",
        "buy_count",
        "sell_count",
        "total_shares",
        "total_volume_usdc",
        "first_trade_timestamp",
        "first_trade_utc",
        "last_trade_timestamp",
        "last_trade_utc",
    )
    temporary = WALLET_MARKET_PATH.with_suffix(".csv.tmp")
    count = 0
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        cursor = connection.execute(
            """
            SELECT wallet_address, market_id, condition_id, market_type,
                   trade_count, buy_count, sell_count, total_shares,
                   total_volume_usdc, first_trade, last_trade
            FROM wallet_market ORDER BY wallet_address, market_id
            """
        )
        for row in cursor:
            (
                wallet, market_id, condition_id, market_type, trades,
                buys, sells, shares, volume, first, last,
            ) = row
            writer.writerow(
                (
                    wallet,
                    market_id,
                    condition_id,
                    market_type,
                    trades,
                    buys,
                    sells,
                    format_number(shares),
                    format_number(volume),
                    first,
                    utc_iso(first),
                    last,
                    utc_iso(last),
                )
            )
            count += 1
    temporary.replace(WALLET_MARKET_PATH)
    return count


def build_qc(
    connection: sqlite3.Connection,
    manifest: list[dict[str, str]],
    wallet_rows: int,
    wallet_market_rows: int,
) -> dict[str, object]:
    expected_trades = sum(int(row["row_count"]) for row in manifest)
    completed = connection.execute(
        "SELECT COUNT(*) FROM processed_markets"
    ).fetchone()[0]
    wallet_totals = connection.execute(
        """
        SELECT COALESCE(SUM(trade_count), 0), COALESCE(SUM(markets_traded), 0),
               COALESCE(SUM(total_shares), 0),
               COALESCE(SUM(total_volume_usdc), 0),
               MIN(first_trade), MAX(last_trade)
        FROM wallets
        """
    ).fetchone()
    market_totals = connection.execute(
        """
        SELECT COALESCE(SUM(trade_count), 0),
               COALESCE(SUM(buy_count), 0), COALESCE(SUM(sell_count), 0),
               COALESCE(SUM(total_shares), 0),
               COALESCE(SUM(total_volume_usdc), 0),
               MIN(first_trade), MAX(last_trade)
        FROM wallet_market
        """
    ).fetchone()

    errors: list[str] = []
    if completed != 360:
        errors.append(f"Expected 360 processed markets, got {completed}")
    if wallet_totals[0] != expected_trades:
        errors.append("Wallet trade total does not reconcile to manifest")
    if market_totals[0] != expected_trades:
        errors.append("Wallet-market trade total does not reconcile to manifest")
    if wallet_totals[1] != wallet_market_rows:
        errors.append("Sum(markets_traded) does not equal wallet-market rows")
    if market_totals[1] + market_totals[2] != expected_trades:
        errors.append("BUY plus SELL count does not equal total trades")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "processed_markets": completed,
        "wallet_rows": wallet_rows,
        "wallet_market_rows": wallet_market_rows,
        "expected_trade_rows": expected_trades,
        "wallet_trade_rows": wallet_totals[0],
        "wallet_market_trade_rows": market_totals[0],
        "buy_rows": market_totals[1],
        "sell_rows": market_totals[2],
        "total_shares_wallets": wallet_totals[2],
        "total_shares_wallet_market": market_totals[3],
        "total_volume_usdc_wallets": wallet_totals[3],
        "total_volume_usdc_wallet_market": market_totals[4],
        "first_trade_timestamp": wallet_totals[4],
        "first_trade_utc": utc_iso(wallet_totals[4]),
        "last_trade_timestamp": wallet_totals[5],
        "last_trade_utc": utc_iso(wallet_totals[5]),
        "errors": errors,
        "aggregation_notes": [
            "One input row is one Polymarket Data API trade record.",
            "total_volume_usdc is the sum of size multiplied by price.",
            "Wallet addresses are normalized to lowercase.",
            "No whale, institution, or ownership classification was applied.",
            "SQLite REAL values are used for aggregated shares and USDC volume.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Skip aggregation and rebuild CSV outputs from the checkpoint database",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lock_handle = LOCK_PATH.open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise SystemExit("Another wallet build process is already running") from error

    manifest = load_manifest()
    connection = connect()
    try:
        if not args.export_only:
            processed = {
                row[0]: row[1]
                for row in connection.execute(
                    "SELECT market_id, source_sha256 FROM processed_markets"
                )
            }
            for number, market in enumerate(manifest, 1):
                market_id = market["market_id"]
                if market_id in processed:
                    if processed[market_id] != market["sha256"]:
                        raise RuntimeError(
                            f"Source checksum changed for completed market {market_id}"
                        )
                    print(f"[{number}/360] skip completed {market_id}", flush=True)
                    continue
                source = SOURCE_DIR / f"{market_id}.csv.gz"
                print(f"[{number}/360] aggregate {market_id}", flush=True)
                aggregates = aggregate_market(source, market["condition_id"].lower())
                source_rows = sum(item.trade_count for item in aggregates.values())
                if source_rows != int(market["row_count"]):
                    raise RuntimeError(
                        f"{market_id}: aggregated {source_rows} rows, expected "
                        f"{market['row_count']}"
                    )
                store_market(connection, market, aggregates)
                print(
                    f"  {source_rows:,} trades; {len(aggregates):,} wallets",
                    flush=True,
                )

        completed = connection.execute(
            "SELECT COUNT(*) FROM processed_markets"
        ).fetchone()[0]
        if completed != 360:
            print(f"Aggregation incomplete: {completed}/360 markets", flush=True)
            return 2

        print("Exporting wallets.csv...", flush=True)
        wallet_rows = export_wallets(connection)
        print("Exporting wallet_market.csv...", flush=True)
        wallet_market_rows = export_wallet_market(connection)
        qc = build_qc(connection, manifest, wallet_rows, wallet_market_rows)
        temporary = QC_PATH.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(qc, indent=2, ensure_ascii=False))
        temporary.replace(QC_PATH)
        print(json.dumps(qc, indent=2, ensure_ascii=False), flush=True)
        return 1 if qc["errors"] else 0
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
