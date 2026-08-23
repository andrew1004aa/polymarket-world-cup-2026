#!/usr/bin/env python3
"""Build resumable H2a other-wallet pre/post flow features for P99 events."""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVENT_DIR = ROOT / "analysis_ready/v4/behavioral_events"
TRADE_MANIFEST = ROOT / "regression_ready/trade_files.csv"
OUT = ROOT / "analysis_ready/v4/h2a_prepost_flow"
CHECKPOINTS = OUT / "checkpoints"
LOCK = OUT / ".builder.lock"
WINDOWS = (60, 300, 900)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024): digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n"); temp.replace(path)


def load_csv(path: Path) -> list[dict[str, str]]:
    opener = __import__('gzip').open if path.suffix == '.gz' else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def direction(row: dict[str, str]) -> int:
    return 1 if (row["side"].upper() == "BUY") == (row["outcome"].upper() == "YES") else -1


def fmt(value: float | None) -> str:
    return "" if value is None else format(value, ".12g")


def window_features(trades: list[dict], times: list[int], start: int, end: int,
                    initiating_wallet: str, initiating_direction: int) -> dict[str, float | int | str]:
    left = bisect.bisect_right(times, start)
    right = bisect.bisect_left(times, end)
    same_value = opposite_value = 0.0
    same_count = opposite_count = 0
    same_wallets, opposite_wallets = set(), set()
    for trade in trades[left:right]:
        if trade["wallet"] == initiating_wallet: continue
        if trade["direction"] == initiating_direction:
            same_value += trade["value"]; same_count += 1; same_wallets.add(trade["wallet"])
        else:
            opposite_value += trade["value"]; opposite_count += 1; opposite_wallets.add(trade["wallet"])
    total = same_value + opposite_value
    return {
        "same_value": same_value, "opposite_value": opposite_value,
        "directional_net": same_value - opposite_value,
        "directional_imbalance": (same_value - opposite_value) / total if total else None,
        "same_count": same_count, "opposite_count": opposite_count,
        "same_wallet_count": len(same_wallets), "opposite_wallet_count": len(opposite_wallets),
    }


def process(entry: dict[str, str], force: bool) -> dict:
    mid = entry["market_id"]
    source = ROOT / entry["path"]
    events_path = EVENT_DIR / f"{mid}.csv"
    target = OUT / f"{mid}.csv"
    checkpoint = CHECKPOINTS / f"{mid}.json"
    source_hash, events_hash = sha256(source), sha256(events_path)
    if checkpoint.exists() and target.exists() and not force:
        prior = json.loads(checkpoint.read_text())
        if prior.get("source_trade_sha256") == source_hash and prior.get("source_events_sha256") == events_hash \
                and prior.get("output_sha256") == sha256(target): return prior

    raw_trades = load_csv(source)
    trades = sorted(({
        "timestamp": int(row["timestamp"]), "wallet": row["wallet_address"].lower(),
        "direction": direction(row), "value": float(row["trade_value_usdc"]),
    } for row in raw_trades), key=lambda x: x["timestamp"])
    times = [row["timestamp"] for row in trades]
    events = load_csv(events_path)
    output = []
    for event in events:
        t = int(event["initiating_timestamp"]); wallet = event["initiating_wallet"].lower(); d = int(event["initiating_direction"])
        row = dict(event)
        for seconds in WINDOWS:
            tag = f"{seconds // 60}m"
            # Open intervals exclude every trade at the initiating timestamp.
            pre = window_features(trades, times, t - seconds, t, wallet, d)
            post = window_features(trades, times, t, t + seconds, wallet, d)
            for period, values in (("pre", pre), ("post", post)):
                for name, value in values.items(): row[f"other_wallet_{period}_{name}_{tag}"] = fmt(value) if isinstance(value, float) else value
            row[f"other_wallet_post_minus_pre_directional_net_{tag}"] = fmt(post["directional_net"] - pre["directional_net"])
            pre_log = math.copysign(math.log1p(abs(pre["directional_net"])), pre["directional_net"]) if pre["directional_net"] else 0.0
            post_log = math.copysign(math.log1p(abs(post["directional_net"])), post["directional_net"]) if post["directional_net"] else 0.0
            row[f"other_wallet_post_minus_pre_signed_log_directional_net_{tag}"] = fmt(post_log - pre_log)
            row[f"other_wallet_any_post_trade_{tag}"] = int(post["same_count"] + post["opposite_count"] > 0)
        output.append(row)
    temp = target.with_suffix(".csv.tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0])); writer.writeheader(); writer.writerows(output)
    temp.replace(target)
    meta = {"status": "complete", "market_id": mid, "event_rows": len(output),
            "source_trade_sha256": source_hash, "source_events_sha256": events_hash,
            "output_path": str(target.relative_to(ROOT)), "output_sha256": sha256(target), "completed_at": now()}
    atomic_json(checkpoint, meta); return meta


def combine(results: list[dict]) -> dict:
    expected = sum(int(row["event_rows"]) for row in results)
    target = OUT / "h2a_p99_prepost_events_all.csv"; temp = target.with_suffix(".csv.tmp")
    rows = 0; fieldnames = None
    with temp.open("w", encoding="utf-8", newline="") as dst:
        writer = None
        for result in sorted(results, key=lambda x: int(x["market_id"])):
            with (ROOT / result["output_path"]).open(encoding="utf-8", newline="") as src:
                reader = csv.DictReader(src)
                if writer is None:
                    fieldnames = reader.fieldnames; writer = csv.DictWriter(dst, fieldnames=fieldnames); writer.writeheader()
                elif reader.fieldnames != fieldnames: raise RuntimeError(f"Header mismatch {result['market_id']}")
                for row in reader: writer.writerow(row); rows += 1
    if rows != expected: temp.unlink(missing_ok=True); raise RuntimeError(f"Rows {rows} != {expected}")
    temp.replace(target)
    diagnostics = {tag: {"any_post_trade": 0, "nonmissing_pre_imbalance": 0, "nonmissing_post_imbalance": 0}
                   for tag in ("1m", "5m", "15m")}
    with target.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            for tag in diagnostics:
                diagnostics[tag]["any_post_trade"] += int(row[f"other_wallet_any_post_trade_{tag}"])
                diagnostics[tag]["nonmissing_pre_imbalance"] += int(bool(row[f"other_wallet_pre_directional_imbalance_{tag}"]))
                diagnostics[tag]["nonmissing_post_imbalance"] += int(bool(row[f"other_wallet_post_directional_imbalance_{tag}"]))
    summary = {"generated_at": now(), "status": "built_qc_passed", "markets": len(results), "event_rows": rows,
               "windows_seconds": list(WINDOWS), "same_timestamp_trades_excluded": True,
               "initiating_wallet_excluded": True, "coverage": diagnostics,
               "output_path": str(target.relative_to(ROOT)), "output_sha256": sha256(target)}
    atomic_json(OUT / "summary.json", summary); return summary


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--max-markets", type=int); parser.add_argument("--force", action="store_true")
    args = parser.parse_args(); OUT.mkdir(parents=True, exist_ok=True); CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    try: LOCK.mkdir()
    except FileExistsError: raise RuntimeError(f"Another H2a builder appears active: {LOCK}")
    (LOCK / "owner.json").write_text(json.dumps({"pid": os.getpid(), "started_at": now()}) + "\n")
    try:
        entries = load_csv(TRADE_MANIFEST)
        if args.max_markets is not None: entries = entries[:args.max_markets]
        results = []
        for index, entry in enumerate(entries, 1):
            result = process(entry, args.force); results.append(result)
            print(f"[{index}/{len(entries)}] {entry['market_id']}: {result['event_rows']:,} events", flush=True)
        if len(entries) == 360: print(json.dumps(combine(results), indent=2), flush=True)
        else: print(json.dumps({"status": "partial", "markets": len(results), "events": sum(x["event_rows"] for x in results)}, indent=2))
        return 0
    finally:
        (LOCK / "owner.json").unlink(missing_ok=True); LOCK.rmdir()


if __name__ == "__main__": raise SystemExit(main())
