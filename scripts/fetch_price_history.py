#!/usr/bin/env python3
"""Download resumable Polymarket CLOB price history for the 360-market sample.

The script preserves every successful API response as a gzip-compressed JSON
checkpoint, then creates a long-format CSV without resampling or filling gaps.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MARKETS = ROOT / "raw/markets/markets.csv"
DEFAULT_WHITELIST = ROOT / "intermediate/market_partitions/market_whitelist.csv"
RAW_DIR = ROOT / "raw/prices/checkpoints"
PROGRESS_PATH = ROOT / "raw/prices/progress.json"
API_LOG_PATH = ROOT / "logs/price_history_api_log.jsonl"
OUTPUT_PATH = ROOT / "raw/prices/prices.csv"
QC_PATH = ROOT / "raw/prices/price_history_qc.json"
API_URL = "https://clob.polymarket.com/prices-history"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_time(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Timestamps must include a timezone, preferably Z")
    return int(parsed.timestamp())


def iso_time(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def atomic_gzip_bytes(path: Path, payload: bytes) -> None:
    """Save the response body byte-for-byte inside gzip compression."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with gzip.open(temporary, "wb") as output:
        output.write(payload)
    os.replace(temporary, path)


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def decode_token_ids(value: str) -> list[str]:
    """Decode Gamma CSV's occasionally double-encoded clob_token_ids field."""
    current: Any = value.strip()
    for _ in range(4):
        if isinstance(current, list):
            break
        if not isinstance(current, str):
            break
        try:
            current = json.loads(current)
        except json.JSONDecodeError:
            break
    if isinstance(current, list):
        tokens = [str(item).strip() for item in current]
    else:
        # Token IDs are base-10 uint256 values. This is a safe fallback for
        # legacy CSV quoting such as "\"[\\\"123...\\\", ...]\"".
        tokens = re.findall(r"\d{20,}", value)
    if len(tokens) != 2 or any(not token.isdigit() for token in tokens):
        raise ValueError(f"Expected exactly two decimal CLOB token IDs, got: {value!r}")
    return tokens


def load_whitelist(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8", newline="") as source:
        market_ids = {
            str(row.get("market_id", "")).strip()
            for row in csv.DictReader(source)
            if str(row.get("market_id", "")).strip()
        }
    if len(market_ids) != 360:
        raise RuntimeError(f"Expected 360 market IDs in whitelist; found {len(market_ids)}")
    return market_ids


def load_tokens(path: Path, whitelist_path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen_markets: set[str] = set()
    seen_tokens: set[str] = set()
    whitelist = load_whitelist(whitelist_path)
    with path.open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            market_id = str(row.get("market_id", "")).strip()
            if market_id not in whitelist:
                continue
            condition_id = str(row.get("condition_id", "")).strip().lower()
            if not market_id or not condition_id:
                raise ValueError(f"Missing market_id or condition_id in row: {row}")
            if market_id in seen_markets:
                raise ValueError(f"Duplicate market_id in markets file: {market_id}")
            seen_markets.add(market_id)
            token_ids = decode_token_ids(str(row.get("clob_token_ids", "")))
            # Gamma returns clobTokenIds in the same order as outcomes [Yes, No].
            for outcome, token_id in zip(("YES", "NO"), token_ids):
                if token_id in seen_tokens:
                    raise ValueError(f"Duplicate token ID in markets file: {token_id}")
                seen_tokens.add(token_id)
                records.append({
                    "market_id": market_id,
                    "condition_id": condition_id,
                    "market_type": str(row.get("market_type", "")).strip(),
                    "question": str(row.get("question", "")).strip(),
                    "outcome": outcome,
                    "token_id": token_id,
                })
    if len(seen_markets) != 360 or len(records) != 720:
        raise RuntimeError(
            f"Expected 360 markets and 720 tokens; found {len(seen_markets)} and {len(records)}"
        )
    missing = whitelist - seen_markets
    if missing:
        raise RuntimeError(f"Whitelisted markets missing from markets CSV: {sorted(missing)}")
    return records


def time_windows(start: int, end: int, window_seconds: int):
    cursor = start
    while cursor < end:
        boundary = min(end, cursor + window_seconds)
        yield cursor, boundary
        cursor = boundary


def checkpoint_path(record: dict[str, str], window_start: int, window_end: int) -> Path:
    name = (
        f'{record["outcome"].lower()}_{record["token_id"]}_'
        f"{window_start}_{window_end}.json.gz"
    )
    return RAW_DIR / record["market_id"] / name


def valid_checkpoint(path: Path) -> tuple[bool, int]:
    try:
        with gzip.open(path, "rb") as source:
            body = json.loads(source.read())
        history = body.get("history")
        if not isinstance(history, list):
            return False, 0
        for point in history:
            if not isinstance(point, dict) or "t" not in point or "p" not in point:
                return False, 0
        return True, len(history)
    except (FileNotFoundError, OSError, EOFError, json.JSONDecodeError, AttributeError):
        return False, 0


def initial_progress(start: int, end: int, fidelity: int, window_seconds: int) -> dict[str, Any]:
    return {
        "version": 1,
        "api": API_URL,
        "start_timestamp": start,
        "start_utc": iso_time(start),
        "end_timestamp_exclusive": end,
        "end_utc_exclusive": iso_time(end),
        "fidelity_minutes": fidelity,
        "window_seconds": window_seconds,
        "completed_price_downloads": {},
        "failed_requests": {},
        "retry_counts": {},
        "updated_at": utc_now(),
    }


def load_progress(start: int, end: int, fidelity: int, window_seconds: int) -> dict[str, Any]:
    progress = read_json(
        PROGRESS_PATH, initial_progress(start, end, fidelity, window_seconds)
    )
    # Migrate the initial single-window test progress, which contains no
    # successful checkpoints, to the bounded-window format.
    if "window_seconds" not in progress and not progress.get("completed_price_downloads"):
        progress["window_seconds"] = window_seconds
    expected = (start, end, fidelity, window_seconds)
    actual = (
        progress.get("start_timestamp"),
        progress.get("end_timestamp_exclusive"),
        progress.get("fidelity_minutes"),
        progress.get("window_seconds"),
    )
    if actual != expected:
        raise RuntimeError(
            "Existing progress.json uses a different time window or fidelity. "
            f"Existing={actual}; requested={expected}."
        )
    progress.setdefault("completed_price_downloads", {})
    progress.setdefault("failed_requests", {})
    progress.setdefault("retry_counts", {})
    return progress


def save_progress(progress: dict[str, Any]) -> None:
    progress["updated_at"] = utc_now()
    atomic_json(PROGRESS_PATH, progress)


def download(
    records: list[dict[str, str]],
    start: int,
    end: int,
    fidelity: int,
    retries: int,
    timeout: int,
    pause: float,
    limit: int | None,
    window_seconds: int,
) -> None:
    progress = load_progress(start, end, fidelity, window_seconds)
    session = requests.Session()
    selected = records if limit is None else records[:limit]
    total = len(records)

    for position, record in enumerate(selected, 1):
        token_id = record["token_id"]
        token_key = f'{record["market_id"]}:{record["outcome"]}:{token_id}'
        token_points = 0
        token_complete = True
        windows = list(time_windows(start, end, window_seconds))
        for window_number, (window_start, window_end) in enumerate(windows, 1):
            key = f"{token_key}:{window_start}:{window_end}"
            path = checkpoint_path(record, window_start, window_end)
            valid, points = valid_checkpoint(path)
            if valid:
                token_points += points
                progress["completed_price_downloads"][key] = {
                    "path": str(path.relative_to(ROOT)), "points": points
                }
                progress["failed_requests"].pop(key, None)
                save_progress(progress)
                print(
                    f"[{position}/{len(selected)} | {total} total] window "
                    f"{window_number}/{len(windows)} already complete ({points:,} points)"
                )
                continue

            params = {
                "market": token_id,
                "startTs": window_start,
                "endTs": window_end,
                "fidelity": fidelity,
            }
            last_error: Exception | None = None
            window_complete = False
            for attempt in range(retries + 1):
                request_time = utc_now()
                response = None
                try:
                    response = session.get(API_URL, params=params, timeout=timeout)
                    response.raise_for_status()
                    body = response.json()
                    history = body.get("history") if isinstance(body, dict) else None
                    if not isinstance(history, list):
                        raise ValueError("Response does not contain a history array")
                    for point in history:
                        if not isinstance(point, dict) or "t" not in point or "p" not in point:
                            raise ValueError("Malformed price-history point")
                    atomic_gzip_bytes(path, response.content)
                    progress["completed_price_downloads"][key] = {
                        "path": str(path.relative_to(ROOT)),
                        "points": len(history),
                        "completed_at": utc_now(),
                    }
                    progress["failed_requests"].pop(key, None)
                    append_jsonl(API_LOG_PATH, {
                        "time": request_time,
                        "market_id": record["market_id"],
                        "outcome": record["outcome"],
                        "token_id": token_id,
                        "params": params,
                        "attempt": attempt,
                        "status": response.status_code,
                        "points": len(history),
                        "checkpoint": str(path.relative_to(ROOT)),
                    })
                    save_progress(progress)
                    token_points += len(history)
                    window_complete = True
                    print(
                        f"[{position}/{len(selected)} | {total} total] window "
                        f"{window_number}/{len(windows)} downloaded ({len(history):,} points)"
                    )
                    time.sleep(pause)
                    break
                except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
                    last_error = exc
                    progress["retry_counts"][key] = int(progress["retry_counts"].get(key, 0)) + 1
                    error_body = ""
                    if response is not None:
                        error_body = response.text[:500]
                    progress["failed_requests"][key] = {
                        "last_error": repr(exc),
                        "response_body": error_body,
                        "last_attempt_at": utc_now(),
                    }
                    append_jsonl(API_LOG_PATH, {
                        "time": request_time,
                        "market_id": record["market_id"],
                        "outcome": record["outcome"],
                        "token_id": token_id,
                        "params": params,
                        "attempt": attempt,
                        "error": repr(exc),
                        "response_body": error_body,
                    })
                    save_progress(progress)
                    if attempt < retries:
                        retry_after = 0
                        if response is not None:
                            try:
                                retry_after = int(response.headers.get("Retry-After", "0"))
                            except ValueError:
                                retry_after = 0
                        wait = max(retry_after, min(60, 2 ** attempt))
                        print(f"  retrying in {wait}s after {exc!r}", file=sys.stderr, flush=True)
                        time.sleep(wait)
            if not window_complete:
                token_complete = False
                print(
                    f"[{position}/{len(selected)}] FAILED window {window_number}/{len(windows)} "
                    f"for {token_key}: {last_error!r}", file=sys.stderr
                )
        if token_complete:
            progress["failed_requests"].pop(token_key, None)  # remove legacy test failure
            progress["completed_price_downloads"][token_key] = {
                "windows": len(windows), "points": token_points, "completed_at": utc_now()
            }
            save_progress(progress)


def build_csv(
    records: list[dict[str, str]], start: int, end: int, fidelity: int,
    window_seconds: int,
) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_PATH.with_name(f"{OUTPUT_PATH.name}.{os.getpid()}.tmp")
    fields = [
        "market_id", "condition_id", "market_type", "question", "outcome",
        "token_id", "timestamp", "timestamp_utc", "price",
        "requested_start_timestamp", "requested_end_timestamp_exclusive",
        "fidelity_minutes", "request_window_start", "request_window_end",
        "raw_point_json",
    ]
    rows = missing = empty = outside = malformed = 0
    token_counts: dict[str, int] = {}
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for index, record in enumerate(records, 1):
            count = 0
            for window_start, window_end in time_windows(start, end, window_seconds):
                path = checkpoint_path(record, window_start, window_end)
                valid, _ = valid_checkpoint(path)
                if not valid:
                    missing += 1
                    continue
                with gzip.open(path, "rt", encoding="utf-8") as source:
                    history = json.load(source)["history"]
                if not history:
                    empty += 1
                for point in history:
                    try:
                        timestamp = int(point["t"])
                        price = point["p"]
                        float(price)
                    except (KeyError, TypeError, ValueError):
                        malformed += 1
                        continue
                    if not start <= timestamp < end:
                        outside += 1
                        continue
                    writer.writerow({
                        **record,
                        "timestamp": timestamp,
                        "timestamp_utc": iso_time(timestamp),
                        "price": price,
                        "requested_start_timestamp": start,
                        "requested_end_timestamp_exclusive": end,
                        "fidelity_minutes": fidelity,
                        "request_window_start": window_start,
                        "request_window_end": window_end,
                        "raw_point_json": json.dumps(point, ensure_ascii=False, separators=(",", ":")),
                    })
                    rows += 1
                    count += 1
            token_counts[record["token_id"]] = count
            if index % 20 == 0 or index == len(records):
                print(f"Building prices.csv: {index}/{len(records)} tokens, {rows:,} rows", flush=True)
    os.replace(temporary, OUTPUT_PATH)
    qc = {
        "generated_at": utc_now(),
        "markets_expected": 360,
        "tokens_expected": 720,
        "request_windows_expected": 720 * len(list(time_windows(start, end, window_seconds))),
        "request_windows_missing_checkpoints": missing,
        "request_windows_with_empty_history": empty,
        "price_rows": rows,
        "malformed_points_excluded": malformed,
        "points_outside_exact_window_excluded": outside,
        "start_timestamp": start,
        "start_utc": iso_time(start),
        "end_timestamp_exclusive": end,
        "end_utc_exclusive": iso_time(end),
        "fidelity_minutes": fidelity,
        "window_seconds": window_seconds,
        "notes": [
            "Raw API responses are preserved unchanged inside gzip checkpoints.",
            "prices.csv is long-format and contains no resampling, interpolation, or gap filling.",
            "YES and NO token histories were downloaded separately.",
        ],
    }
    atomic_json(QC_PATH, qc)
    print(json.dumps(qc, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markets", type=Path, default=DEFAULT_MARKETS)
    parser.add_argument("--whitelist", type=Path, default=DEFAULT_WHITELIST)
    parser.add_argument("--start", default="2026-06-01T00:00:00Z")
    parser.add_argument("--end", default="2026-08-01T00:00:00Z")
    parser.add_argument("--fidelity", type=int, default=1, help="Minutes; 1 is the highest API resolution")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--pause", type=float, default=0.15)
    parser.add_argument("--window-days", type=int, default=14, help="API request window; 14 days is verified to work")
    parser.add_argument("--limit", type=int, help="Download only the first N tokens (testing only)")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    if args.build_only and args.download_only:
        parser.error("--build-only and --download-only are mutually exclusive")
    if args.fidelity < 1:
        parser.error("--fidelity must be at least 1 minute")
    if args.window_days < 1 or args.window_days > 14:
        parser.error("--window-days must be between 1 and 14")
    start, end = parse_time(args.start), parse_time(args.end)
    if start >= end:
        parser.error("--start must be before --end")

    records = load_tokens(args.markets, args.whitelist)
    window_seconds = args.window_days * 86400
    print(f"Loaded {len(records) // 2} markets and {len(records)} outcome tokens")
    print(f"Exact window: {iso_time(start)} <= timestamp < {iso_time(end)}")
    if not args.build_only:
        download(
            records, start, end, args.fidelity, args.retries, args.timeout,
            args.pause, args.limit, window_seconds,
        )
    if not args.download_only and args.limit is None:
        build_csv(records, start, end, args.fidelity, window_seconds)
    elif args.limit is not None and not args.download_only:
        print("Skipping prices.csv build because --limit was used; rerun without --limit to finish.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
