#!/usr/bin/env python3
"""Build one-market v4 push-follow-reverse event prototype.

This intentionally small prototype freezes event semantics before a resumable
360-market feature builder is implemented. Canonical inputs are read-only.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "v4_diagnostics/push_follow_reverse_prototype"
FOLLOW_WINDOWS = (60, 300, 900)
PRICE_HORIZONS = (300, 900, 1800, 3600)
MAX_BASELINE_AGE = 120
MAX_TARGET_LATENESS = 60


def direction(side: str, outcome: str) -> int:
    side, outcome = side.upper(), outcome.upper()
    if side not in {"BUY", "SELL"} or outcome not in {"YES", "NO"}:
        raise ValueError(f"Unexpected side/outcome {side}/{outcome}")
    return 1 if (side == "BUY") == (outcome == "YES") else -1


def load_threshold(market_id: str) -> float:
    path = ROOT / "large_trade_diagnostics/v3/market_thresholds.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["market_id"] == market_id:
                return float(row["p99_threshold_usdc"])
    raise KeyError(market_id)


def load_trades(market_id: str) -> list[dict]:
    path = ROOT / f"regression_ready/trades_by_market/{market_id}.csv.gz"
    trades = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            trades.append({
                "trade_record_id": row["trade_record_id"],
                "timestamp": int(row["timestamp"]),
                "trade_time_utc": row["trade_time_utc"],
                "transaction_hash": row["transaction_hash"],
                "wallet": row["wallet_address"].lower(),
                "direction": direction(row["side"], row["outcome"]),
                "value": float(row["trade_value_usdc"]),
                "price": float(row["yes_equivalent_price"]),
                "question": row["question"],
                "event_id": row["event_id"],
                "kickoff": row["actual_kickoff_utc"],
            })
    return sorted(trades, key=lambda x: (x["timestamp"], x["trade_record_id"]))


def load_prices(market_id: str) -> tuple[list[int], list[float]]:
    # The YES token is used to avoid treating complementary YES/NO histories as
    # independent observations. Points are not resampled or interpolated.
    path = ROOT / f"regression_ready/prices_by_market/{market_id}.csv.gz"
    points = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["outcome"].upper() == "YES":
                points[int(row["timestamp"])] = float(row["yes_equivalent_price"])
    timestamps = sorted(points)
    return timestamps, [points[t] for t in timestamps]


def baseline_price(timestamps: list[int], prices: list[float], event_time: int):
    index = bisect.bisect_right(timestamps, event_time) - 1
    if index < 0 or event_time - timestamps[index] > MAX_BASELINE_AGE:
        return (None, None, None)
    return prices[index], timestamps[index], event_time - timestamps[index]


def target_price(timestamps: list[int], prices: list[float], target: int):
    index = bisect.bisect_left(timestamps, target)
    if index >= len(timestamps) or timestamps[index] - target > MAX_TARGET_LATENESS:
        return (None, None, None)
    return prices[index], timestamps[index], timestamps[index] - target


def fmt(value):
    return "" if value is None else format(value, ".12g")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market-id", default="1897043")
    parser.add_argument("--initiating-record-ids")
    parser.add_argument("--output-stem")
    args = parser.parse_args()
    market_id = args.market_id
    threshold = load_threshold(market_id)
    trades = load_trades(market_id)
    trade_times = [x["timestamp"] for x in trades]
    price_times, prices = load_prices(market_id)
    wallet_indices = defaultdict(list)
    minute_wallet_value = defaultdict(lambda: defaultdict(float))
    minute_signed_value = defaultdict(float)
    minute_gross_value = defaultdict(float)
    for index, trade in enumerate(trades):
        wallet_indices[trade["wallet"]].append(index)
        minute = trade["timestamp"] - trade["timestamp"] % 60
        minute_wallet_value[minute][trade["wallet"]] += trade["value"]
        minute_signed_value[minute] += trade["direction"] * trade["value"]
        minute_gross_value[minute] += trade["value"]

    output = []
    selected_ids = None
    if args.initiating_record_ids:
        with Path(args.initiating_record_ids).open(encoding="utf-8") as handle:
            selected_ids = {line.strip() for line in handle if line.strip()}
    for initiating in trades:
        if selected_ids is None and initiating["value"] < threshold:
            continue
        if selected_ids is not None and initiating["trade_record_id"] not in selected_ids:
            continue
        event_time = initiating["timestamp"]
        row = {
            "market_id": market_id,
            "event_id": initiating["event_id"],
            "question": initiating["question"],
            "initiating_trade_record_id": initiating["trade_record_id"],
            "initiating_timestamp": event_time,
            "initiating_time_utc": initiating["trade_time_utc"],
            "initiating_transaction_hash": initiating["transaction_hash"],
            "initiating_wallet": initiating["wallet"],
            "initiating_direction": initiating["direction"],
            "initiating_trade_value_usdc": fmt(initiating["value"]),
            "p99_threshold_usdc": fmt(threshold),
        }
        minute = event_time - event_time % 60
        wallet_values = sorted(minute_wallet_value[minute].values(), reverse=True)
        minute_gross = minute_gross_value[minute]
        shares = [value / minute_gross for value in wallet_values] if minute_gross else []
        row.update({
            "initiating_minute_gross_value_usdc": fmt(minute_gross),
            "initiating_minute_distinct_wallets": len(wallet_values),
            "initiating_minute_top1_wallet_share": fmt(shares[0] if shares else None),
            "initiating_minute_top3_wallet_share": fmt(sum(shares[:3]) if shares else None),
            "initiating_minute_wallet_hhi": fmt(sum(share * share for share in shares) if shares else None),
            "initiating_minute_absolute_flow_imbalance": fmt(
                abs(minute_signed_value[minute]) / minute_gross if minute_gross else None
            ),
        })
        baseline, baseline_ts, baseline_age = baseline_price(price_times, prices, event_time)
        row.update({"baseline_yes_price": fmt(baseline), "baseline_price_timestamp": baseline_ts or "",
                    "baseline_price_age_seconds": baseline_age if baseline_age is not None else ""})
        for seconds in (300, 900):
            prior, prior_ts, prior_age = baseline_price(price_times, prices, event_time - seconds)
            tag = f"{seconds // 60}m"
            recent_change = None if baseline is None or prior is None else baseline - prior
            alignment = None if recent_change is None or recent_change == 0 else (
                1 if initiating["direction"] * recent_change > 0 else -1
            )
            row.update({
                f"prior_yes_price_{tag}": fmt(prior),
                f"recent_price_change_{tag}": fmt(recent_change),
                f"initiating_trade_trend_alignment_{tag}": "" if alignment is None else alignment,
            })
        for seconds in FOLLOW_WINDOWS:
            left = bisect.bisect_right(trade_times, event_time)
            right = bisect.bisect_right(trade_times, event_time + seconds)
            same_value = opposite_value = 0.0
            same_wallets, opposite_wallets = set(), set()
            same_count = opposite_count = 0
            for follower in trades[left:right]:
                if follower["wallet"] == initiating["wallet"]:
                    continue
                if follower["direction"] == initiating["direction"]:
                    same_value += follower["value"]; same_count += 1; same_wallets.add(follower["wallet"])
                else:
                    opposite_value += follower["value"]; opposite_count += 1; opposite_wallets.add(follower["wallet"])
            total = same_value + opposite_value
            tag = f"{seconds // 60}m"
            row.update({
                f"same_direction_value_{tag}": fmt(same_value),
                f"opposite_direction_value_{tag}": fmt(opposite_value),
                f"follower_imbalance_{tag}": fmt((same_value - opposite_value) / total) if total else "",
                f"same_direction_trade_count_{tag}": same_count,
                f"opposite_direction_trade_count_{tag}": opposite_count,
                f"same_direction_wallet_count_{tag}": len(same_wallets),
                f"opposite_direction_wallet_count_{tag}": len(opposite_wallets),
            })
        initiating_opposite = []
        for index in wallet_indices[initiating["wallet"]]:
            candidate = trades[index]
            if event_time < candidate["timestamp"] <= event_time + 3600 and candidate["direction"] != initiating["direction"]:
                initiating_opposite.append(candidate)
        row["initiating_wallet_opposite_trade_within_60m"] = int(bool(initiating_opposite))
        row["initiating_wallet_opposite_value_60m"] = fmt(sum(x["value"] for x in initiating_opposite))
        first_opposite = initiating_opposite[0] if initiating_opposite else None
        row["seconds_to_first_initiating_wallet_opposite_trade"] = (
            first_opposite["timestamp"] - event_time if first_opposite else ""
        )
        for seconds in PRICE_HORIZONS:
            target, target_ts, lateness = target_price(price_times, prices, event_time + seconds)
            tag = f"{seconds // 60}m"
            change = None if baseline is None or target is None else target - baseline
            directional = None if change is None else initiating["direction"] * change
            row.update({
                f"yes_price_{tag}": fmt(target), f"price_timestamp_{tag}": target_ts or "",
                f"price_lateness_seconds_{tag}": lateness if lateness is not None else "",
                f"price_change_{tag}": fmt(change), f"directional_price_change_{tag}": fmt(directional),
                f"reversal_from_5m_to_{tag}": "",
            })
        first_move = row.get("directional_price_change_5m", "")
        if first_move != "":
            first_move_value = float(first_move)
            for seconds in (900, 1800, 3600):
                tag = f"{seconds // 60}m"
                later = row.get(f"directional_price_change_{tag}", "")
                if later != "" and first_move_value != 0:
                    row[f"reversal_from_5m_to_{tag}"] = int(float(later) < first_move_value)
        output.append(row)

    OUT.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem or f"market_{market_id}"
    target = OUT / f"{stem}_events.csv"
    temporary = target.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader(); writer.writerows(output)
    temporary.replace(target)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "prototype_built_not_frozen", "market_id": market_id,
        "question": trades[0]["question"], "canonical_trade_rows": len(trades),
        "p99_threshold_usdc": threshold, "initiating_p99_events": len(output),
        "selection_mode": "record_id_list" if selected_ids is not None else "within_market_p99",
        "selected_record_ids_requested": len(selected_ids) if selected_ids is not None else None,
        "events_with_baseline_price": sum(bool(x["baseline_yes_price"]) for x in output),
        "events_with_5m_price": sum(bool(x["yes_price_5m"]) for x in output),
        "events_with_any_other_wallet_following_5m": sum(
            int(x["same_direction_trade_count_5m"]) + int(x["opposite_direction_trade_count_5m"]) > 0
            for x in output
        ),
        "events_with_same_direction_other_wallet_flow_5m": sum(float(x["same_direction_value_5m"]) > 0 for x in output),
        "events_with_initiating_wallet_opposite_trade_60m": sum(
            int(x["initiating_wallet_opposite_trade_within_60m"]) for x in output
        ),
        "price_alignment": {"baseline_max_age_seconds": MAX_BASELINE_AGE,
                            "target_max_lateness_seconds": MAX_TARGET_LATENESS,
                            "interpolation": False},
        "output": str(target.relative_to(ROOT)),
    }
    (OUT / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
