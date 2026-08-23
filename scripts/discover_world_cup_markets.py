#!/usr/bin/env python3
"""Discover 2026 FIFA World Cup match-winner and outright-winner markets."""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
GAMMA = "https://gamma-api.polymarket.com"
DEFAULT_TAGS = ["2026-fifa-world-cup"]
DEFAULT_SERIES_IDS = ["11433"]  # sport=fifwc in Gamma /sports metadata
FORBIDDEN = re.compile(
    r"\b(player|goal|goals|corner|corners|card|cards|handicap|spread|"
    r"over[ /-]?under|qualif|group|top scorer|golden boot|clean sheet)\b", re.I
)
WORLD_CUP = re.compile(r"(2026.*(?:fifa )?world cup|(?:fifa )?world cup.*2026)", re.I)
OUTRIGHT = re.compile(r"\bwin (?:the )?2026 (?:fifa )?world cup\b", re.I)
NON_COUNTRY_OUTRIGHT = re.compile(
    r"^(?:Team [A-Z]+|Other|Africa \(CAF\)|Asia \(AFC\)|Europe \(UEFA\)|"
    r"North America \(CONCACAF\)|South America \(CONMEBOL\)|Oceania \(OCF\))$",
    re.I,
)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def get(session: requests.Session, path: str, params: dict[str, Any], retries: int) -> Any:
    error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = session.get(GAMMA + path, params=params, timeout=60)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            error = exc
            if attempt < retries:
                time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"Gamma request failed: {error}")


def fetch_tag(
    session: requests.Session, tag_slug: str, raw_dir: Path, retries: int
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    cursor: str | None = None
    page = 0
    while True:
        params: dict[str, Any] = {
            "limit": 500, "tag_slug": tag_slug, "related_tags": "false",
        }
        if cursor:
            params["after_cursor"] = cursor
        body = get(session, "/events/keyset", params, retries)
        page += 1
        atomic_json(raw_dir / tag_slug / f"page_{page:04d}.json", body)
        events.extend(body.get("events", []))
        cursor = body.get("next_cursor")
        if not cursor:
            break
    return events


def fetch_series(
    session: requests.Session, series_id: str, raw_dir: Path, retries: int
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    cursor: str | None = None
    page = 0
    while True:
        params: dict[str, Any] = {"limit": 500, "series_id": series_id}
        if cursor:
            params["after_cursor"] = cursor
        body = get(session, "/events/keyset", params, retries)
        page += 1
        atomic_json(raw_dir / f"series_{series_id}" / f"page_{page:04d}.json", body)
        events.extend(body.get("events", []))
        cursor = body.get("next_cursor")
        if not cursor:
            break
    return events


def classify(event: dict[str, Any], market: dict[str, Any]) -> tuple[str | None, str]:
    event_text = " ".join(str(event.get(k) or "") for k in ("title", "slug", "description"))
    market_text = " ".join(str(market.get(k) or "") for k in ("question", "slug"))
    combined = event_text + " " + market_text
    tag_slugs = {
        str(tag.get("slug") or "").lower() for tag in event.get("tags", [])
    }
    series_slug = str(event.get("seriesSlug") or "").lower()
    is_fifwc_series = series_slug in {"soccer-fifwc", "fifwc"}
    if (
        "2026-fifa-world-cup" not in tag_slugs
        and not is_fifwc_series
        and not WORLD_CUP.search(combined)
    ):
        return None, "not explicitly 2026 FIFA World Cup"
    if FORBIDDEN.search(market_text):
        return None, "excluded market type/keyword"
    if OUTRIGHT.search(market_text):
        group_title = str(market.get("groupItemTitle") or "").strip()
        if NON_COUNTRY_OUTRIGHT.fullmatch(group_title):
            return None, "outright is placeholder, other, or continent (not a country)"
        return "outright", ""
    sports_type = str(market.get("sportsMarketType") or "").lower()
    if is_fifwc_series and sports_type == "moneyline":
        return "match", ""
    return None, f"not outright or match moneyline (sportsMarketType={sports_type!r})"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag-slug", action="append", dest="tags")
    parser.add_argument("--series-id", action="append", dest="series_ids")
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args()
    tags = args.tags or DEFAULT_TAGS
    series_ids = args.series_ids or DEFAULT_SERIES_IDS

    raw_dir = ROOT / "raw/markets/gamma_discovery"
    session = requests.Session()
    by_event: dict[str, dict[str, Any]] = {}
    tag_counts: dict[str, int] = {}
    failures: list[dict[str, str]] = []
    for tag in tags:
        try:
            events = fetch_tag(session, tag, raw_dir, args.retries)
            tag_counts[tag] = len(events)
            for event in events:
                by_event[str(event.get("id") or event.get("slug"))] = event
        except Exception as exc:
            failures.append({"tag_slug": tag, "error": repr(exc)})
    series_counts: dict[str, int] = {}
    for series_id in series_ids:
        try:
            events = fetch_series(session, series_id, raw_dir, args.retries)
            series_counts[series_id] = len(events)
            for event in events:
                by_event[str(event.get("id") or event.get("slug"))] = event
        except Exception as exc:
            failures.append({"series_id": series_id, "error": repr(exc)})

    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen_markets: set[str] = set()
    for event in by_event.values():
        for market in event.get("markets", []):
            market_id = str(market.get("id") or market.get("conditionId") or "")
            if not market_id or market_id in seen_markets:
                continue
            seen_markets.add(market_id)
            market_type, reason = classify(event, market)
            record = {"market_type": market_type, "market": market, "event": event}
            if market_type:
                included.append(record)
            else:
                excluded.append({
                    "market_id": market_id,
                    "question": market.get("question"),
                    "event_id": event.get("id"),
                    "event_title": event.get("title"),
                    "reason": reason,
                })

    atomic_json(ROOT / "market_metadata.json", included)
    atomic_json(ROOT / "raw/markets/excluded_markets.json", excluded)
    atomic_json(ROOT / "raw/markets/discovery_summary.json", {
        "tag_counts": tag_counts,
        "series_counts": series_counts,
        "unique_events": len(by_event),
        "included_markets": len(included),
        "match_markets": sum(x["market_type"] == "match" for x in included),
        "outright_markets": sum(x["market_type"] == "outright" for x in included),
        "excluded_markets": len(excluded),
        "failed_requests": failures,
    })

    csv_path = ROOT / "raw/markets/markets.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "market_id", "condition_id", "market_type", "event_id", "question",
        "slug", "sports_market_type", "game_id", "start_date", "end_date",
        "active", "closed", "clob_token_ids",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in included:
            market, event = item["market"], item["event"]
            writer.writerow({
                "market_id": market.get("id"),
                "condition_id": market.get("conditionId"),
                "market_type": item["market_type"],
                "event_id": event.get("id"),
                "question": market.get("question"),
                "slug": market.get("slug"),
                "sports_market_type": market.get("sportsMarketType"),
                "game_id": market.get("gameId") or event.get("gameId"),
                "start_date": market.get("startDate"),
                "end_date": market.get("endDate"),
                "active": market.get("active"),
                "closed": market.get("closed"),
                "clob_token_ids": json.dumps(market.get("clobTokenIds")),
            })

    print(json.dumps({
        "included": len(included),
        "match": sum(x["market_type"] == "match" for x in included),
        "outright": sum(x["market_type"] == "outright" for x in included),
        "excluded": len(excluded),
        "failures": failures,
    }, indent=2))
    if not included:
        raise RuntimeError(
            "No target markets found. Inspect raw/markets/discovery_summary.json "
            "and supply the exact Gamma tag with --tag-slug."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
