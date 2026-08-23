#!/usr/bin/env python3
"""Build the fixed Dune-defined World Cup market whitelist."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DUNE_DIR = ROOT / "intermediate" / "market_partitions" / "dune"
OUTPUT = ROOT / "intermediate" / "market_partitions" / "market_whitelist.csv"

DUNE_FILES = (
    ("match", DUNE_DIR / "dune_match_markets.csv"),
    ("outright", DUNE_DIR / "dune_outright_markets.csv"),
)

FIELDS = (
    "market_id",
    "condition_id",
    "market_type",
    "question",
    "slug",
    "event_slug",
    "selection_source",
    "dune_query_id",
)


def main() -> int:
    metadata = json.loads((ROOT / "market_metadata.json").read_text())
    metadata_by_condition = {
        str(item["market"]["conditionId"]).lower(): item for item in metadata
    }

    output_rows: list[dict[str, str]] = []
    seen: set[str] = set()

    for market_type, path in DUNE_FILES:
        query_id = "8139394" if market_type == "match" else "8140259"
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for dune_row in csv.DictReader(handle):
                condition_id = dune_row["condition_id"].strip().lower()
                if not condition_id:
                    raise ValueError(f"Blank condition_id in {path}")
                if condition_id in seen:
                    raise ValueError(f"Duplicate condition_id: {condition_id}")
                seen.add(condition_id)

                item = metadata_by_condition.get(condition_id)
                if item is None:
                    raise ValueError(
                        f"Dune condition_id absent from market_metadata.json: "
                        f"{condition_id}"
                    )

                market = item["market"]
                event = item["event"]
                if item["market_type"] != market_type:
                    raise ValueError(
                        f"Market type disagreement for {condition_id}: "
                        f"Dune={market_type}, local={item['market_type']}"
                    )

                output_rows.append(
                    {
                        "market_id": str(market["id"]),
                        "condition_id": condition_id,
                        "market_type": market_type,
                        "question": str(market.get("question") or ""),
                        "slug": str(market.get("slug") or ""),
                        "event_slug": str(event.get("slug") or ""),
                        "selection_source": "Dune latest query result",
                        "dune_query_id": query_id,
                    }
                )

    type_counts = {
        kind: sum(row["market_type"] == kind for row in output_rows)
        for kind in ("match", "outright")
    }
    if type_counts != {"match": 312, "outright": 48}:
        raise ValueError(f"Unexpected whitelist counts: {type_counts}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    temporary.replace(OUTPUT)

    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "markets": len(output_rows),
                **type_counts,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
