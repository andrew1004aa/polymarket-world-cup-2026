#!/usr/bin/env python3
"""Build versioned, resumable market-minute analytical tables.

Inputs under regression_ready/ are read-only. Per-market checkpoints make the
build resumable. This script constructs variables only; it does not estimate a
model, test a hypothesis, trim observations, or interpret results.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = ROOT / "analysis_ready"
SOURCE = ROOT / "regression_ready"

OUTPUT_FIELDS = [
    "analysis_record_id", "analysis_version", "sample", "market_id",
    "condition_id", "event_id", "fifa_match_id", "market_type",
    "market_subtype", "market_role", "question", "country", "stage",
    "home_team", "away_team", "minute_start_timestamp",
    "minute_start_utc", "minute_end_timestamp", "minute_end_utc",
    "date_utc", "hour_of_day_utc", "calendar_hour_utc",
    "calendar_15min_block_utc", "actual_kickoff_utc",
    "minutes_from_actual_kickoff", "resolved_on_timestamp",
    "minutes_to_resolution", "yes_outcome_won", "trade_count",
    "distinct_wallet_count", "gross_volume_usdc", "net_signed_flow_usdc",
    "whale_trade_count", "whale_distinct_wallet_count",
    "whale_gross_volume_usdc", "whale_net_signed_flow_usdc",
    "nonwhale_trade_count", "nonwhale_distinct_wallet_count",
    "nonwhale_gross_volume_usdc", "nonwhale_net_signed_flow_usdc",
    "whale_flow_share", "signed_log_whale_net_flow",
    "signed_log_nonwhale_net_flow", "lagged_30m_gross_volume_usdc",
    "lagged_30m_price_change", "baseline_price_timestamp",
    "baseline_price_utc", "baseline_price_age_seconds", "yes_price_t",
]

HORIZON_FIELDS = [
    "price_timestamp_{h}m", "price_timestamp_{h}m_utc",
    "price_lateness_{h}m_seconds", "yes_price_{h}m", "delta_yes_price_{h}m",
    "directional_impact_{h}m", "brier_improvement_{h}m",
]


def utc_iso(timestamp: int | None) -> str:
    if timestamp is None:
        return ""
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc(value: str) -> int | None:
    value = (value or "").strip()
    if not value:
        return None
    if value.endswith(" UTC"):
        return int(datetime.strptime(value, "%Y-%m-%d %H:%M:%S.%f UTC").replace(tzinfo=timezone.utc).timestamp())
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())


def fmt(value: float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return format(value, ".12g")


def signed_log(value: float) -> float:
    if value == 0:
        return 0.0
    return math.copysign(math.log1p(abs(value)), value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def output_fields(horizons: list[int]) -> list[str]:
    fields = list(OUTPUT_FIELDS)
    for horizon in horizons:
        fields.extend(field.format(h=horizon) for field in HORIZON_FIELDS)
    return fields


def load_config(version: str) -> tuple[Path, dict]:
    out = ANALYSIS_ROOT / version
    config_path = out / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    return out, json.loads(config_path.read_text(encoding="utf-8"))


def build_whale_table(out: Path, config: dict, force: bool = False) -> dict:
    target = out / "whale_wallet_definitions.csv.gz"
    meta_path = out / "whale_wallet_definitions.json"
    if target.exists() and meta_path.exists() and not force:
        return json.loads(meta_path.read_text(encoding="utf-8"))

    source = SOURCE / "tables" / "wallets.csv.gz"
    wallets = []
    with gzip.open(source, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            wallets.append((row["wallet_address"].lower(), float(row["total_volume_usdc"]), int(row["number_of_trades"])))
    wallets.sort(key=lambda item: (-item[1], item[0]))
    total = len(wallets)
    thresholds = sorted(set([float(config["whale_definition"]["primary_top_percent"])] + [float(x) for x in config["whale_definition"]["robustness_top_percent"]]))
    counts = {threshold: max(1, math.ceil(total * threshold / 100.0)) for threshold in thresholds}
    primary = float(config["whale_definition"]["primary_top_percent"])

    temporary = target.with_suffix(target.suffix + ".tmp")
    fields = ["wallet_address", "wallet_volume_rank", "wallet_volume_percentile", "total_volume_usdc", "number_of_trades"]
    fields += [f"top_{str(x).replace('.', '_')}pct" for x in thresholds]
    fields += ["primary_whale"]
    with gzip.open(temporary, "wt", encoding="utf-8", newline="", compresslevel=6) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, (wallet, volume, trades) in enumerate(wallets, 1):
            row = {
                "wallet_address": wallet,
                "wallet_volume_rank": rank,
                "wallet_volume_percentile": fmt(100.0 * rank / total),
                "total_volume_usdc": fmt(volume),
                "number_of_trades": trades,
                "primary_whale": 1 if rank <= counts[primary] else 0,
            }
            for threshold in thresholds:
                row[f"top_{str(threshold).replace('.', '_')}pct"] = 1 if rank <= counts[threshold] else 0
            writer.writerow(row)
    temporary.replace(target)
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "wallets": total,
        "ranking": "total_volume_usdc descending, wallet_address ascending as deterministic tie-break",
        "threshold_counts": {str(k): v for k, v in counts.items()},
        "primary_top_percent": primary,
        "primary_whale_wallets": counts[primary],
        "path": str(target.relative_to(ROOT)),
        "sha256": sha256(target),
    }
    atomic_json(meta_path, metadata)
    return metadata


def load_primary_whales(out: Path) -> set[str]:
    whales = set()
    with gzip.open(out / "whale_wallet_definitions.csv.gz", "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["primary_whale"] == "1":
                whales.add(row["wallet_address"])
    return whales


def load_markets() -> dict[str, dict[str, str]]:
    return {row["market_id"]: row for row in read_csv(SOURCE / "tables" / "markets.csv")}


def load_manifests() -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    trades = read_csv(SOURCE / "trade_files.csv")
    prices = {row["market_id"]: row for row in read_csv(SOURCE / "price_files.csv")}
    return trades, prices


def load_yes_prices(path: Path) -> tuple[list[int], list[float], int]:
    points: dict[int, float] = {}
    no_rows = 0
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["outcome"].upper() == "YES":
                points[int(row["timestamp"])] = float(row["yes_equivalent_price"])
            else:
                no_rows += 1
    timestamps = sorted(points)
    return timestamps, [points[t] for t in timestamps], no_rows


def prior_price(timestamps: list[int], prices: list[float], target: int, max_age_seconds: int) -> tuple[int | None, float | None]:
    pos = bisect.bisect_right(timestamps, target) - 1
    if pos < 0 or target - timestamps[pos] > max_age_seconds:
        return None, None
    return timestamps[pos], prices[pos]


def future_price(timestamps: list[int], prices: list[float], target: int, max_lateness_seconds: int) -> tuple[int | None, float | None]:
    pos = bisect.bisect_left(timestamps, target)
    if pos >= len(timestamps) or timestamps[pos] - target > max_lateness_seconds:
        return None, None
    return timestamps[pos], prices[pos]


def trade_direction(side: str, outcome: str) -> int:
    pair = (side.upper(), outcome.upper())
    directions = {("BUY", "YES"): 1, ("SELL", "NO"): 1, ("SELL", "YES"): -1, ("BUY", "NO"): -1}
    if pair not in directions:
        raise ValueError(f"Unsupported side/outcome pair: {pair}")
    return directions[pair]


def sample_for_trade(market_type: str, timestamp: int, kickoff: int | None) -> str:
    if market_type == "outright":
        return "outright"
    if kickoff is None:
        raise ValueError("Match market lacks actual kickoff")
    return "match_prematch" if timestamp < kickoff else "match_inplay"


def aggregate_trades(path: Path, whales: set[str], market: dict[str, str], resolution: int | None) -> tuple[dict[tuple[str, int], dict], dict]:
    kickoff = None
    groups: dict[tuple[str, int], dict] = {}
    source_rows = 0
    excluded_post_resolution = 0
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            source_rows += 1
            timestamp = int(row["timestamp"])
            if resolution is not None and timestamp >= resolution:
                excluded_post_resolution += 1
                continue
            if market["market_type"] == "match" and kickoff is None:
                kickoff = parse_utc(row["actual_kickoff_utc"])
            sample = sample_for_trade(market["market_type"], timestamp, kickoff)
            minute_start = timestamp - timestamp % 60
            key = (sample, minute_start)
            if key not in groups:
                groups[key] = {
                    "trade_count": 0, "wallets": set(), "gross": 0.0, "net": 0.0,
                    "whale_trade_count": 0, "whale_wallets": set(), "whale_gross": 0.0, "whale_net": 0.0,
                    "nonwhale_trade_count": 0, "nonwhale_wallets": set(), "nonwhale_gross": 0.0, "nonwhale_net": 0.0,
                    "context": row,
                }
            group = groups[key]
            wallet = row["wallet_address"].lower()
            value = float(row["trade_value_usdc"])
            signed = trade_direction(row["side"], row["outcome"]) * value
            group["trade_count"] += 1
            group["wallets"].add(wallet)
            group["gross"] += value
            group["net"] += signed
            if wallet in whales:
                group["whale_trade_count"] += 1
                group["whale_wallets"].add(wallet)
                group["whale_gross"] += value
                group["whale_net"] += signed
            else:
                group["nonwhale_trade_count"] += 1
                group["nonwhale_wallets"].add(wallet)
                group["nonwhale_gross"] += value
                group["nonwhale_net"] += signed
    return groups, {"source_trade_rows": source_rows, "excluded_post_resolution": excluded_post_resolution}


def sample_horizons(config: dict, sample: str) -> list[int]:
    item = config["samples"][sample]
    return sorted(set([int(item["primary_horizon_minutes"])] + [int(x) for x in item["secondary_horizon_minutes"]] + [int(x) for x in item["robustness_horizon_minutes"]]))


def make_rows(groups: dict, market: dict[str, str], timestamps: list[int], prices: list[float], config: dict) -> tuple[dict[str, list[dict]], dict]:
    by_sample = {name: [] for name in config["samples"]}
    missing = defaultdict(lambda: defaultdict(int))
    rolling = {name: deque() for name in config["samples"]}
    rolling_sum = {name: 0.0 for name in config["samples"]}
    resolution = parse_utc(market["resolved_on_timestamp"])

    for (sample, minute_start), group in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
        context = group["context"]
        minute_end = minute_start + 60
        sample_cfg = config["samples"][sample]
        max_age = int(sample_cfg["baseline_max_age_minutes"]) * 60
        p_ts, p0 = prior_price(timestamps, prices, minute_end, max_age)
        if p0 is None:
            missing[sample]["missing_baseline"] += 1

        queue = rolling[sample]
        while queue and queue[0][0] < minute_start - 1800:
            _, old = queue.popleft()
            rolling_sum[sample] -= old
        lagged_volume = rolling_sum[sample]
        lag_ts, lag_p = prior_price(timestamps, prices, minute_end - 1800, max_age)
        lagged_change = None if p0 is None or lag_p is None else p0 - lag_p

        dt = datetime.fromtimestamp(minute_start, timezone.utc)
        kickoff = parse_utc(context["actual_kickoff_utc"])
        yes_won = int(context["yes_outcome_won"]) if context["yes_outcome_won"] in ("0", "1") else None
        row = {
            "analysis_record_id": f"{sample}:{market['market_id']}:{minute_start}",
            "analysis_version": config["version"], "sample": sample,
            "market_id": market["market_id"], "condition_id": market["condition_id"],
            "event_id": context["event_id"], "fifa_match_id": context["fifa_match_id"],
            "market_type": market["market_type"], "market_subtype": market["market_subtype"],
            "market_role": context["market_role"], "question": market["question"],
            "country": market["country"], "stage": context["stage"],
            "home_team": context["home_team"], "away_team": context["away_team"],
            "minute_start_timestamp": minute_start, "minute_start_utc": utc_iso(minute_start),
            "minute_end_timestamp": minute_end, "minute_end_utc": utc_iso(minute_end),
            "date_utc": dt.strftime("%Y-%m-%d"), "hour_of_day_utc": dt.strftime("%H"),
            "calendar_hour_utc": dt.strftime("%Y-%m-%dT%H:00:00Z"),
            "calendar_15min_block_utc": dt.replace(minute=(dt.minute // 15) * 15).strftime("%Y-%m-%dT%H:%M:00Z"),
            "actual_kickoff_utc": context["actual_kickoff_utc"],
            "minutes_from_actual_kickoff": fmt(None if kickoff is None else (minute_end - kickoff) / 60),
            "resolved_on_timestamp": market["resolved_on_timestamp"],
            "minutes_to_resolution": fmt(None if resolution is None else (resolution - minute_end) / 60),
            "yes_outcome_won": "" if yes_won is None else yes_won,
            "trade_count": group["trade_count"], "distinct_wallet_count": len(group["wallets"]),
            "gross_volume_usdc": fmt(group["gross"]), "net_signed_flow_usdc": fmt(group["net"]),
            "whale_trade_count": group["whale_trade_count"], "whale_distinct_wallet_count": len(group["whale_wallets"]),
            "whale_gross_volume_usdc": fmt(group["whale_gross"]), "whale_net_signed_flow_usdc": fmt(group["whale_net"]),
            "nonwhale_trade_count": group["nonwhale_trade_count"], "nonwhale_distinct_wallet_count": len(group["nonwhale_wallets"]),
            "nonwhale_gross_volume_usdc": fmt(group["nonwhale_gross"]), "nonwhale_net_signed_flow_usdc": fmt(group["nonwhale_net"]),
            "whale_flow_share": fmt(group["whale_gross"] / group["gross"] if group["gross"] else 0.0),
            "signed_log_whale_net_flow": fmt(signed_log(group["whale_net"])),
            "signed_log_nonwhale_net_flow": fmt(signed_log(group["nonwhale_net"])),
            "lagged_30m_gross_volume_usdc": fmt(lagged_volume),
            "lagged_30m_price_change": fmt(lagged_change),
            "baseline_price_timestamp": "" if p_ts is None else p_ts,
            "baseline_price_utc": utc_iso(p_ts),
            "baseline_price_age_seconds": "" if p_ts is None else minute_end - p_ts,
            "yes_price_t": fmt(p0),
        }

        for horizon in sample_horizons(config, sample):
            target_ts = minute_end + horizon * 60
            boundary = kickoff if sample == "match_prematch" else resolution
            crossed = boundary is not None and target_ts > boundary
            future_ts = future = None
            if crossed:
                missing[sample][f"{horizon}m_crossed_boundary"] += 1
            elif p0 is None:
                missing[sample][f"{horizon}m_missing_due_to_baseline"] += 1
            else:
                future_ts, future = future_price(timestamps, prices, target_ts, int(sample_cfg["target_max_lateness_minutes"]) * 60)
                if future is None:
                    missing[sample][f"{horizon}m_missing_target"] += 1
            delta = None if p0 is None or future is None else future - p0
            flow_sign = 0 if group["net"] == 0 else (1 if group["net"] > 0 else -1)
            brier = None if delta is None or yes_won is None else (yes_won - p0) ** 2 - (yes_won - future) ** 2
            row.update({
                f"price_timestamp_{horizon}m": "" if future_ts is None else future_ts,
                f"price_timestamp_{horizon}m_utc": utc_iso(future_ts),
                f"price_lateness_{horizon}m_seconds": "" if future_ts is None else future_ts - target_ts,
                f"yes_price_{horizon}m": fmt(future),
                f"delta_yes_price_{horizon}m": fmt(delta),
                f"directional_impact_{horizon}m": fmt(None if delta is None else flow_sign * delta),
                f"brier_improvement_{horizon}m": fmt(brier),
            })
        by_sample[sample].append(row)
        queue.append((minute_start, group["gross"]))
        rolling_sum[sample] += group["gross"]
    return by_sample, {sample: dict(values) for sample, values in missing.items()}


def write_checkpoint(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="", compresslevel=6) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_progress(out: Path) -> dict:
    path = out / "progress.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "completed_markets": {}, "failed": []}


def save_progress(out: Path, progress: dict) -> None:
    progress["updated_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(out / "progress.json", progress)


def process_markets(out: Path, config: dict, whales: set[str], market_ids: set[str] | None = None, max_markets: int | None = None) -> None:
    markets = load_markets()
    trade_manifest, price_manifest = load_manifests()
    progress = load_progress(out)
    selected = [item for item in trade_manifest if market_ids is None or item["market_id"] in market_ids]
    if max_markets is not None:
        selected = selected[:max_markets]
    for index, item in enumerate(selected, 1):
        market_id = item["market_id"]
        prior = progress["completed_markets"].get(market_id)
        paths = {sample: out / "checkpoints" / sample / f"{market_id}.csv.gz" for sample in config["samples"]}
        if prior and all(path.exists() for path in paths.values()):
            print(f"[{index}/{len(selected)}] {market_id} checkpointed", flush=True)
            continue
        try:
            market = markets[market_id]
            resolution = parse_utc(market["resolved_on_timestamp"])
            groups, trade_stats = aggregate_trades(ROOT / item["path"], whales, market, resolution)
            price_item = price_manifest[market_id]
            timestamps, prices, no_rows = load_yes_prices(ROOT / price_item["path"])
            rows_by_sample, missing = make_rows(groups, market, timestamps, prices, config)
            checkpoint_meta = {}
            for sample, rows in rows_by_sample.items():
                fields = output_fields(sample_horizons(config, sample))
                write_checkpoint(paths[sample], rows, fields)
                checkpoint_meta[sample] = {"rows": len(rows), "sha256": sha256(paths[sample])}
            progress["completed_markets"][market_id] = {
                "market_type": market["market_type"], "source_trade_rows": trade_stats["source_trade_rows"],
                "excluded_post_resolution": trade_stats["excluded_post_resolution"],
                "yes_price_points": len(timestamps), "no_price_rows_ignored": no_rows,
                "checkpoints": checkpoint_meta, "missing_prices": missing,
            }
            save_progress(out, progress)
            counts = ", ".join(f"{sample}={meta['rows']:,}" for sample, meta in checkpoint_meta.items() if meta["rows"])
            print(f"[{index}/{len(selected)}] {market_id} {market['market_type']} {counts or '0 rows'}", flush=True)
        except Exception as exc:
            progress["failed"].append({"market_id": market_id, "error": repr(exc), "at": datetime.now(timezone.utc).isoformat()})
            save_progress(out, progress)
            raise


def combine(out: Path, config: dict, require_all: bool = True) -> None:
    markets = load_markets()
    progress = load_progress(out)
    if require_all and len(progress["completed_markets"]) != len(markets):
        raise RuntimeError(f"Cannot combine: {len(progress['completed_markets'])}/{len(markets)} markets completed")
    manifest_rows = []
    sample_flow = []
    errors = []
    for sample in config["samples"]:
        fields = output_fields(sample_horizons(config, sample))
        target = out / f"market_minute_{sample}.csv.gz"
        temporary = target.with_suffix(target.suffix + ".tmp")
        total = 0
        trade_rows = 0
        gross = 0.0
        missing_primary = 0
        primary_h = int(config["samples"][sample]["primary_horizon_minutes"])
        with gzip.open(temporary, "wt", encoding="utf-8", newline="", compresslevel=6) as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            for market_id in sorted(progress["completed_markets"], key=lambda x: int(x)):
                checkpoint = out / "checkpoints" / sample / f"{market_id}.csv.gz"
                if not checkpoint.exists():
                    errors.append(f"missing checkpoint {checkpoint}")
                    continue
                with gzip.open(checkpoint, "rt", encoding="utf-8", newline="") as handle:
                    for row in csv.DictReader(handle):
                        writer.writerow(row)
                        total += 1
                        trade_rows += int(row["trade_count"])
                        gross += float(row["gross_volume_usdc"])
                        if not row[f"delta_yes_price_{primary_h}m"]:
                            missing_primary += 1
        temporary.replace(target)
        meta = {"sample": sample, "path": str(target.relative_to(ROOT)), "rows": total, "trade_rows": trade_rows, "gross_volume_usdc": fmt(gross), "missing_primary_outcome": missing_primary, "sha256": sha256(target)}
        manifest_rows.append(meta)
        sample_flow.append(meta.copy())

    manifest_path = out / "file_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader(); writer.writerows(manifest_rows)
    flow_path = out / "sample_flow.csv"
    with flow_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample_flow[0]))
        writer.writeheader(); writer.writerows(sample_flow)

    source_trade_rows = sum(int(v["source_trade_rows"]) for v in progress["completed_markets"].values())
    excluded = sum(int(v["excluded_post_resolution"]) for v in progress["completed_markets"].values())
    included = sum(int(row["trade_rows"]) for row in manifest_rows)
    if source_trade_rows != excluded + included:
        errors.append(f"trade reconciliation failed: source={source_trade_rows}, excluded={excluded}, included={included}")
    qc = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "version": config["version"],
        "markets_expected": len(markets), "markets_completed": len(progress["completed_markets"]),
        "source_trade_rows": source_trade_rows, "excluded_post_resolution": excluded,
        "included_trade_rows": included, "sample_files": manifest_rows,
        "whale_metadata": json.loads((out / "whale_wallet_definitions.json").read_text(encoding="utf-8")),
        "errors": errors,
    }
    atomic_json(out / "analysis_ready_qc.json", qc)
    if errors:
        raise RuntimeError("QC errors: " + "; ".join(errors[:5]))
    config["status"] = "built_qc_passed" if len(progress["completed_markets"]) == len(markets) else "partial_test_build"
    config["built_at"] = qc["generated_at"]
    atomic_json(out / "config.json", config)
    print(json.dumps(qc, indent=2), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=None, help="Analysis version; defaults to analysis_ready/CURRENT_VERSION")
    parser.add_argument("--stage", choices=["whales", "markets", "combine", "all"], default="all")
    parser.add_argument("--market-id", action="append", help="Process only specified market ID; repeatable")
    parser.add_argument("--max-markets", type=int, help="Process only first N selected markets for validation")
    parser.add_argument("--force-whales", action="store_true")
    parser.add_argument("--allow-partial-combine", action="store_true")
    args = parser.parse_args()
    version = args.version or (ANALYSIS_ROOT / "CURRENT_VERSION").read_text(encoding="utf-8").strip()
    out, config = load_config(version)
    if config["version"] != version:
        raise ValueError("Config version mismatch")
    if args.stage in ("whales", "all"):
        meta = build_whale_table(out, config, args.force_whales)
        print(json.dumps(meta, indent=2), flush=True)
    if args.stage in ("markets", "all"):
        if not (out / "whale_wallet_definitions.csv.gz").exists():
            build_whale_table(out, config)
        whales = load_primary_whales(out)
        process_markets(out, config, whales, set(args.market_id) if args.market_id else None, args.max_markets)
    if args.stage in ("combine", "all"):
        combine(out, config, not args.allow_partial_combine)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
