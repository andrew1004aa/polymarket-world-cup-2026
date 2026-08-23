#!/usr/bin/env python3
"""Merge the final Dune market results into a 360-row market table."""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORK_DIR = ROOT / "intermediate" / "market_partitions"
DUNE_DIR = WORK_DIR / "dune"
OUTPUT = WORK_DIR / "markets_final.csv"
QC_OUTPUT = WORK_DIR / "markets_final_qc.json"

SOURCES = (
    ("match", "8139394", DUNE_DIR / "dune_match_markets.csv"),
    ("outright", "8140259", DUNE_DIR / "dune_outright_markets.csv"),
)

FIELDS = (
    "market_id",
    "condition_id",
    "market_type",
    "market_subtype",
    "country",
    "question",
    "slug",
    "event_slug",
    "market_start_time",
    "market_end_time",
    "country_market_versions",
    "resolution_status",
    "resolution_result",
    "is_resolved",
    "resolved_outcome",
    "country_won_world_cup",
    "winning_outcome_index",
    "maximum_settlement_value",
    "total_settlement_value",
    "resolved_on_timestamp",
    "yes_outcome_won",
    "outcome_index_0_won",
    "outcome_token_count",
    "settled_token_count",
    "trade_count",
    "transaction_count",
    "volume_usdc",
    "volume_shares",
    "all_match_markets_volume_usdc",
    "total_traded_usdc",
    "sample_total_volume_usdc",
    "first_trade_time",
    "last_trade_time",
    "dune_query_id",
)


def canonical_bool(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    return None


def atomic_csv(path: Path, rows: list[dict[str, str]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> int:
    metadata = json.loads((ROOT / "market_metadata.json").read_text())
    metadata_by_condition = {
        str(item["market"]["conditionId"]).strip().lower(): item
        for item in metadata
    }

    manifest_by_market: dict[str, dict[str, str]] = {}
    with (WORK_DIR / "market_manifest.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        for row in csv.DictReader(handle):
            manifest_by_market[row["market_id"]] = row

    final_rows: list[dict[str, str]] = []
    source_columns: dict[str, list[str]] = {}

    for broad_type, query_id, path in SOURCES:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            source_columns[path.name] = list(reader.fieldnames or [])
            dune_rows = list(reader)

        for source in dune_rows:
            condition_id = source["condition_id"].strip().lower()
            item = metadata_by_condition.get(condition_id)
            if item is None:
                raise ValueError(f"Missing local metadata: {condition_id}")

            market = item["market"]
            event = item["event"]
            market_id = str(market["id"])
            subtype = source.get("market_type", "").strip()
            if broad_type == "outright" and not subtype:
                subtype = "Outright winner"

            row = {field: "" for field in FIELDS}
            for field in FIELDS:
                if field in source:
                    row[field] = source[field].strip()
            row.update(
                {
                    "market_id": market_id,
                    "condition_id": condition_id,
                    "market_type": broad_type,
                    "market_subtype": subtype,
                    "question": source.get("question", "").strip()
                    or str(market.get("question") or ""),
                    "slug": str(market.get("slug") or ""),
                    "event_slug": str(event.get("slug") or ""),
                    "dune_query_id": query_id,
                }
            )
            row["sample_total_volume_usdc"] = (
                source.get("all_match_markets_volume_usdc", "").strip()
                or source.get("total_traded_usdc", "").strip()
            )
            final_rows.append(row)

    final_rows.sort(key=lambda row: (row["market_type"], int(row["market_id"])))

    errors: list[str] = []
    warnings: list[str] = []
    condition_ids = [row["condition_id"] for row in final_rows]
    market_ids = [row["market_id"] for row in final_rows]
    counts = Counter(row["market_type"] for row in final_rows)

    if len(final_rows) != 360:
        errors.append(f"Expected 360 rows, got {len(final_rows)}")
    if counts != {"match": 312, "outright": 48}:
        errors.append(f"Unexpected market type counts: {dict(counts)}")
    if len(set(condition_ids)) != len(condition_ids):
        errors.append("Duplicate condition_id values")
    if len(set(market_ids)) != len(market_ids):
        errors.append("Duplicate market_id values")

    post_resolution_trades: list[str] = []
    trade_count_mismatches: list[dict[str, str]] = []
    for row in final_rows:
        market_id = row["market_id"]
        if row["resolution_status"].lower() != "resolved":
            errors.append(f"{market_id}: resolution_status is not resolved")
        if canonical_bool(row["is_resolved"]) is not True:
            errors.append(f"{market_id}: is_resolved is not true")
        if row["outcome_token_count"] != "2":
            errors.append(f"{market_id}: outcome_token_count != 2")
        if row["settled_token_count"] != "2":
            errors.append(f"{market_id}: settled_token_count != 2")
        try:
            if float(row["maximum_settlement_value"]) != 1.0:
                errors.append(f"{market_id}: maximum_settlement_value != 1")
            if float(row["total_settlement_value"]) != 1.0:
                errors.append(f"{market_id}: total_settlement_value != 1")
        except ValueError:
            errors.append(f"{market_id}: invalid settlement value")

        yes_won = row["resolved_outcome"].strip().lower() == "yes"
        expected_index = "0" if yes_won else "1"
        if row["winning_outcome_index"] != expected_index:
            errors.append(f"{market_id}: winning outcome index disagreement")
        if canonical_bool(row["resolution_result"]) is not yes_won:
            errors.append(f"{market_id}: resolution_result disagreement")
        if row["market_type"] == "match":
            if canonical_bool(row["yes_outcome_won"]) is not yes_won:
                errors.append(f"{market_id}: yes_outcome_won disagreement")
            if canonical_bool(row["outcome_index_0_won"]) is not yes_won:
                errors.append(f"{market_id}: outcome_index_0_won disagreement")
        else:
            if canonical_bool(row["country_won_world_cup"]) is not yes_won:
                errors.append(f"{market_id}: country_won_world_cup disagreement")

        if (
            row["last_trade_time"]
            and row["resolved_on_timestamp"]
            and row["last_trade_time"] > row["resolved_on_timestamp"]
        ):
            post_resolution_trades.append(market_id)

        manifest = manifest_by_market.get(market_id)
        if manifest is None:
            errors.append(f"{market_id}: absent from market_manifest.csv")
        elif int(row["trade_count"]) != int(manifest["row_count"]):
            trade_count_mismatches.append(
                {
                    "market_id": market_id,
                    "dune_trade_count": row["trade_count"],
                    "data_api_row_count": manifest["row_count"],
                }
            )

    outright_winners = [
        row for row in final_rows
        if row["market_type"] == "outright"
        and canonical_bool(row["country_won_world_cup"]) is True
    ]
    if len(outright_winners) != 1:
        errors.append(
            f"Expected one outright winner, got {len(outright_winners)}"
        )
    if trade_count_mismatches:
        warnings.append(
            f"{len(trade_count_mismatches)} markets have Dune/Data API count differences"
        )
    if post_resolution_trades:
        warnings.append(
            f"{len(post_resolution_trades)} markets have last_trade_time after "
            "resolved_on_timestamp"
        )

    atomic_csv(OUTPUT, final_rows)
    qc = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output": str(OUTPUT),
        "rows": len(final_rows),
        "market_type_counts": dict(counts),
        "unique_condition_ids": len(set(condition_ids)),
        "unique_market_ids": len(set(market_ids)),
        "outright_winner_count": len(outright_winners),
        "outright_winner": outright_winners[0]["country"]
        if len(outright_winners) == 1
        else None,
        "post_resolution_trade_markets": post_resolution_trades,
        "trade_count_mismatches": trade_count_mismatches,
        "errors": errors,
        "warnings": warnings,
        "source_columns": source_columns,
    }
    temporary_qc = QC_OUTPUT.with_suffix(".json.tmp")
    temporary_qc.write_text(json.dumps(qc, indent=2, ensure_ascii=False))
    temporary_qc.replace(QC_OUTPUT)

    print(json.dumps(qc, indent=2, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
