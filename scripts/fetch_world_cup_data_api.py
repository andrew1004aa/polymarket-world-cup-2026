#!/usr/bin/env python3
"""Resumable World Cup-period trade collection from Polymarket Data API."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
API = "https://data-api.polymarket.com/trades"
METADATA = ROOT / "market_metadata.json"
RAW_DIR = ROOT / "raw/trades/data_api"
PROGRESS = ROOT / "data_api_progress.json"
API_LOG = ROOT / "logs/data_api_log.jsonl"
OUTPUT = ROOT / "trades.csv"
REPORT = ROOT / "docs/data_api_collection_report.md"
FIELDS = [
    "proxyWallet", "side", "asset", "conditionId", "size", "price",
    "timestamp", "title", "slug", "icon", "eventSlug", "outcome",
    "outcomeIndex", "name", "pseudonym", "bio", "profileImage",
    "profileImageOptimized", "transactionHash", "market_id",
    "market_type", "window_start", "window_end_exclusive", "raw_json",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def epoch(value: str) -> int:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return int(dt.timestamp())


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_gzip(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, path)


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def load_markets(path: Path) -> list[dict[str, Any]]:
    output = []
    for record in read_json(path, []):
        market = record.get("market", record)
        condition_id = market.get("conditionId")
        if condition_id:
            output.append({
                "condition_id": condition_id,
                "market_id": market.get("id", ""),
                "market_type": record.get("market_type", ""),
                "question": market.get("question", ""),
            })
    if not output:
        raise RuntimeError(f"No condition IDs found in {path}")
    return output


class Client:
    def __init__(self, retries: int, timeout: int, pause: float):
        self.retries = retries
        self.timeout = timeout
        self.pause = pause
        self.session = requests.Session()

    def page(
        self, condition_id: str, start: int, end: int, limit: int, offset: int
    ) -> list[dict[str, Any]]:
        params = {
            "market": condition_id, "start": start, "end": end,
            "limit": limit, "offset": offset, "takerOnly": "true",
        }
        error: Exception | None = None
        for attempt in range(self.retries + 1):
            started = now()
            try:
                response = self.session.get(API, params=params, timeout=self.timeout)
                response.raise_for_status()
                body = response.json()
                if not isinstance(body, list):
                    raise ValueError("Expected a JSON array")
                append_jsonl(API_LOG, {
                    "time": started, "params": params, "attempt": attempt,
                    "status": response.status_code, "rows": len(body),
                })
                time.sleep(self.pause)
                return body
            except (requests.RequestException, ValueError) as exc:
                error = exc
                append_jsonl(API_LOG, {
                    "time": started, "params": params, "attempt": attempt,
                    "error": repr(exc),
                })
                if attempt < self.retries:
                    time.sleep(min(30, 2 ** attempt))
        raise RuntimeError(f"Data API request failed: {error}")


def collect_window(
    client: Client,
    market: dict[str, Any],
    start: int,
    end: int,
    limit: int,
    max_offset: int,
    min_window: int,
    depth: int = 0,
) -> tuple[list[Path], list[dict[str, Any]]]:
    cid = market["condition_id"]
    pages: list[list[dict[str, Any]]] = []
    saturated = False
    for offset in range(0, max_offset + 1, limit):
        page = client.page(cid, start, end, limit, offset)
        pages.append(page)
        if len(page) < limit:
            break
        if offset + limit > max_offset:
            saturated = True

    if saturated:
        if end - start <= min_window:
            return [], [{
                "condition_id": cid, "start": start, "end": end,
                "reason": "offset ceiling reached at minimum time window",
            }]
        middle = start + (end - start) // 2
        left_paths, left_sat = collect_window(
            client, market, start, middle, limit, max_offset, min_window, depth + 1
        )
        right_paths, right_sat = collect_window(
            client, market, middle, end, limit, max_offset, min_window, depth + 1
        )
        return left_paths + right_paths, left_sat + right_sat

    rows = [row for page in pages for row in page]
    # Keep the exact API rows in the checkpoint; exact interval filtering occurs later.
    name = f"{start}_{end}_{hashlib.sha256(cid.encode()).hexdigest()[:12]}.json.gz"
    path = RAW_DIR / cid[:18] / name
    write_gzip(path, {
        "request": {
            "market": cid, "start": start, "end": end,
            "limit": limit, "max_offset": max_offset,
        },
        "response": rows,
    })
    return [path], []


def iter_raw(read_retries: int = 5):
    paths = sorted(RAW_DIR.glob("*/*.json.gz"))
    total = len(paths)
    for index, path in enumerate(paths, 1):
        if index == 1 or index % 10 == 0 or index == total:
            print(
                f"\rReading raw checkpoints: {index:,}/{total:,} "
                f"({index / total:.1%}) — {path.name[:42]}",
                end="",
                flush=True,
            )
        body = None
        last_error: Exception | None = None
        for attempt in range(read_retries + 1):
            try:
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    body = json.load(f)
                break
            except (
                TimeoutError, OSError, EOFError, gzip.BadGzipFile,
                json.JSONDecodeError,
            ) as exc:
                last_error = exc
                if attempt >= read_retries:
                    raise RuntimeError(
                        f"Unable to read raw checkpoint after retries: {path}"
                    ) from exc
                wait = min(30, 2 ** attempt)
                print(
                    f"Local checkpoint read failed; retrying in {wait}s "
                    f"({attempt + 1}/{read_retries}): {path}",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(wait)
        if body is None:
            raise RuntimeError(
                f"Unable to read raw checkpoint: {path}: {last_error}"
            )
        yield path, body["request"], body["response"]
    if total:
        print(flush=True)


def build_csv(markets: list[dict[str, Any]], exact_start: int, exact_end: int) -> dict[str, int]:
    market_map = {m["condition_id"].lower(): m for m in markets}
    seen: set[str] = set()
    included = duplicates = missing_timestamp = outside = 0
    tmp = OUTPUT.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for _, request, rows in iter_raw():
            for row in rows:
                try:
                    timestamp = int(row.get("timestamp"))
                except (TypeError, ValueError):
                    missing_timestamp += 1
                    continue
                if not exact_start <= timestamp < exact_end:
                    outside += 1
                    continue
                raw = json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
                key = hashlib.sha256(raw.encode()).hexdigest()
                if key in seen:
                    duplicates += 1
                    continue
                seen.add(key)
                cid = str(row.get("conditionId") or request["market"]).lower()
                market = market_map.get(cid, {})
                output = {
                    **row,
                    "market_id": market.get("market_id", ""),
                    "market_type": market.get("market_type", ""),
                    "window_start": request["start"],
                    "window_end_exclusive": request["end"],
                    "raw_json": raw,
                }
                writer.writerow(output)
                included += 1
                if included % 100_000 == 0:
                    print(
                        f"\nCSV rows written: {included:,}; "
                        f"duplicates removed: {duplicates:,}",
                        flush=True,
                    )
    os.replace(tmp, OUTPUT)
    return {
        "included_rows": included, "duplicate_exact_rows": duplicates,
        "missing_timestamps": missing_timestamp, "outside_window": outside,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=METADATA)
    parser.add_argument("--start", default="2026-06-11T00:00:00Z")
    parser.add_argument("--end", default="2026-07-20T00:00:00Z")
    parser.add_argument("--window-seconds", type=int, default=86_400)
    parser.add_argument("--min-window-seconds", type=int, default=1)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--max-offset", type=int, default=10_000)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--pause", type=float, default=0.06)
    parser.add_argument("--max-markets", type=int)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument(
        "--refresh-from",
        help=(
            "Before downloading, remove completed-window checkpoints whose "
            "start is at or after this UTC timestamp"
        ),
    )
    args = parser.parse_args()

    start, end = epoch(args.start), epoch(args.end)
    current_time = int(time.time())
    if end > current_time:
        raise RuntimeError(
            "--end is still in the future. Run the final collection only after "
            f"{args.end}; otherwise future empty windows could be checkpointed."
        )
    markets = load_markets(args.metadata)
    if args.max_markets:
        markets = markets[:args.max_markets]
    progress = read_json(PROGRESS, {
        "schema_version": 1, "created_at": now(), "completed_windows": {},
        "failed_requests": [], "saturated_windows": [],
    })
    if args.refresh_from:
        refresh_from = epoch(args.refresh_from)
        removed = 0
        for cid, windows in progress["completed_windows"].items():
            retained = []
            for window_key in windows:
                window_start = int(window_key.split(":", 1)[0])
                if window_start >= refresh_from:
                    removed += 1
                else:
                    retained.append(window_key)
            progress["completed_windows"][cid] = retained
        progress["updated_at"] = now()
        atomic_json(PROGRESS, progress)
        print(
            f"Invalidated {removed} completed windows starting at or after "
            f"{args.refresh_from}",
            flush=True,
        )
    client = Client(args.retries, args.timeout, args.pause)

    if not args.build_only:
        for number, market in enumerate(markets, 1):
            cid = market["condition_id"]
            completed = set(progress["completed_windows"].get(cid, []))
            cursor = start
            while cursor < end:
                window_end = min(end, cursor + args.window_seconds)
                window_key = f"{cursor}:{window_end}"
                if window_key not in completed:
                    try:
                        paths, saturated = collect_window(
                            client, market, cursor, window_end, args.limit,
                            args.max_offset, args.min_window_seconds,
                        )
                        if saturated:
                            progress["saturated_windows"].extend(saturated)
                        else:
                            progress["completed_windows"].setdefault(cid, []).append(
                                window_key
                            )
                        progress["updated_at"] = now()
                        atomic_json(PROGRESS, progress)
                        print(
                            f"[{number}/{len(markets)}] {market['question'][:55]} "
                            f"{datetime.fromtimestamp(cursor, timezone.utc).date()} "
                            f"checkpoints={len(paths)} saturated={len(saturated)}",
                            flush=True,
                        )
                    except Exception as exc:
                        progress["failed_requests"].append({
                            "condition_id": cid, "start": cursor, "end": window_end,
                            "error": repr(exc), "time": now(),
                        })
                        atomic_json(PROGRESS, progress)
                        raise
                cursor = window_end

    summary = build_csv(markets, start, end)
    summary["markets"] = len(markets)
    summary["saturated_windows"] = len(progress["saturated_windows"])
    summary["failed_requests"] = len(progress["failed_requests"])
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        "# Polymarket Data API Collection Report\n\n"
        f"- Generated: {now()}\n"
        f"- Exact interval: {args.start} to {args.end} (end exclusive)\n"
        f"- Target markets: {summary['markets']:,}\n"
        f"- Included rows: {summary['included_rows']:,}\n"
        f"- Exact duplicate rows removed: {summary['duplicate_exact_rows']:,}\n"
        f"- Missing timestamps: {summary['missing_timestamps']:,}\n"
        f"- Rows outside exact interval: {summary['outside_window']:,}\n"
        f"- Saturated minimum windows: {summary['saturated_windows']:,}\n"
        f"- Failed requests recorded: {summary['failed_requests']:,}\n\n"
        "This is a collection report only. No whale classification or statistical "
        "analysis was performed.\n",
        encoding="utf-8",
    )
    progress["output"] = {**summary, "csv": str(OUTPUT.relative_to(ROOT))}
    progress["updated_at"] = now()
    atomic_json(PROGRESS, progress)
    print(json.dumps(summary, indent=2))
    print(f"Saved: {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
