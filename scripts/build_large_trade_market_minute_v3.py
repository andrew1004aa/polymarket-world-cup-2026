#!/usr/bin/env python3
"""Build resumable v3 market-minute panels with within-market P99/P95 flows.

The immutable v2 checkpoints supply timing and price-alignment fields. Canonical
trade files are independently re-aggregated to classify large versus ordinary
trades. Every output market is reconciled against both sources before checkpointing.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "analysis_ready/v2"
OUT = ROOT / "analysis_ready/v3"
THRESHOLDS = ROOT / "large_trade_diagnostics/v3/market_thresholds.csv"
TRADE_MANIFEST = ROOT / "regression_ready/trade_files.csv"
DROP_V2 = {
    "whale_trade_count", "whale_distinct_wallet_count",
    "whale_gross_volume_usdc", "whale_net_signed_flow_usdc",
    "nonwhale_trade_count", "nonwhale_distinct_wallet_count",
    "nonwhale_gross_volume_usdc", "nonwhale_net_signed_flow_usdc",
    "whale_flow_share", "signed_log_whale_net_flow",
    "signed_log_nonwhale_net_flow",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def signed_log(value: float) -> float:
    return math.copysign(math.log1p(abs(value)), value) if value else 0.0


def direction(side: str, outcome: str) -> int:
    # YES-equivalent direction: YES buys and NO sells are positive.
    side = side.strip().upper(); outcome = outcome.strip().upper()
    if side not in {"BUY", "SELL"} or outcome not in {"YES", "NO"}:
        raise ValueError(f"Unexpected side/outcome: {side}/{outcome}")
    return 1 if (side == "BUY") == (outcome == "YES") else -1


def fmt(value: float) -> str:
    return format(value, ".12g")


def flow_fields() -> list[str]:
    fields = ["p99_threshold_usdc", "p95_threshold_usdc", "sparse_p99_market"]
    for tag in ("p99", "p95"):
        for group in ("large", "ordinary"):
            fields += [
                f"{tag}_{group}_trade_count", f"{tag}_{group}_distinct_wallet_count",
                f"{tag}_{group}_gross_volume_usdc", f"{tag}_{group}_net_signed_flow_usdc",
                f"{tag}_signed_log_{group}_net_flow",
            ]
        fields.append(f"{tag}_large_gross_volume_share")
    return fields


def load_table(path: Path, key: str) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {row[key]: row for row in csv.DictReader(handle)}


def empty_group() -> dict:
    result = {}
    for tag in ("p99", "p95"):
        for group in ("large", "ordinary"):
            result[f"{tag}_{group}_count"] = 0
            result[f"{tag}_{group}_wallets"] = set()
            result[f"{tag}_{group}_gross"] = 0.0
            result[f"{tag}_{group}_net"] = 0.0
    return result


def aggregate_trades(path: Path, p99: float, p95: float) -> tuple[dict[int, dict], dict]:
    groups: dict[int, dict] = defaultdict(empty_group)
    stats = {"rows": 0, "gross": 0.0, "net": 0.0}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            timestamp = int(row["timestamp"]); minute = timestamp - timestamp % 60
            value = float(row["trade_value_usdc"])
            signed = direction(row["side"], row["outcome"]) * value
            wallet = row["wallet_address"].strip().lower()
            stats["rows"] += 1; stats["gross"] += value; stats["net"] += signed
            g = groups[minute]
            for tag, threshold in (("p99", p99), ("p95", p95)):
                label = "large" if value >= threshold else "ordinary"
                g[f"{tag}_{label}_count"] += 1
                g[f"{tag}_{label}_wallets"].add(wallet)
                g[f"{tag}_{label}_gross"] += value
                g[f"{tag}_{label}_net"] += signed
    return groups, stats


def augment(row: dict[str, str], group: dict, threshold: dict[str, str]) -> dict[str, str]:
    result = {k: v for k, v in row.items() if k not in DROP_V2}
    result["analysis_version"] = "v3"
    result["analysis_record_id"] = result["analysis_record_id"].replace("all:", "v3:", 1)
    result["p99_threshold_usdc"] = threshold["p99_threshold_usdc"]
    result["p95_threshold_usdc"] = threshold["p95_threshold_usdc"]
    result["sparse_p99_market"] = str(int(int(threshold["p99_trade_count"]) < 5))
    gross_total = float(row["gross_volume_usdc"])
    for tag in ("p99", "p95"):
        for label in ("large", "ordinary"):
            count = group[f"{tag}_{label}_count"]
            gross = group[f"{tag}_{label}_gross"]
            net = group[f"{tag}_{label}_net"]
            result[f"{tag}_{label}_trade_count"] = str(count)
            result[f"{tag}_{label}_distinct_wallet_count"] = str(len(group[f"{tag}_{label}_wallets"]))
            result[f"{tag}_{label}_gross_volume_usdc"] = fmt(gross)
            result[f"{tag}_{label}_net_signed_flow_usdc"] = fmt(net)
            result[f"{tag}_signed_log_{label}_net_flow"] = fmt(signed_log(net))
        result[f"{tag}_large_gross_volume_share"] = fmt(
            group[f"{tag}_large_gross"] / gross_total if gross_total else 0.0
        )
    return result


def process_market(mid: str, source: Path, threshold: dict[str, str], force: bool) -> dict:
    target = OUT / "checkpoints" / f"{mid}.csv.gz"
    meta_path = OUT / "market_checkpoints" / f"{mid}.json"
    source_hash = sha256(source)
    threshold_hash = sha256(THRESHOLDS)
    if target.exists() and meta_path.exists() and not force:
        prior = json.loads(meta_path.read_text())
        if (prior.get("status") == "complete" and prior.get("source_trade_sha256") == source_hash
                and prior.get("threshold_table_sha256") == threshold_hash
                and prior.get("source_v2_sha256") == sha256(V2 / "checkpoints" / f"{mid}.csv.gz")):
            return prior

    groups, stats = aggregate_trades(source, float(threshold["p99_threshold_usdc"]),
                                     float(threshold["p95_threshold_usdc"]))
    v2path = V2 / "checkpoints" / f"{mid}.csv.gz"
    temp = target.with_suffix(target.suffix + ".tmp")
    target.parent.mkdir(parents=True, exist_ok=True); meta_path.parent.mkdir(parents=True, exist_ok=True)
    panel_rows = panel_trades = 0; panel_gross = panel_net = 0.0
    with gzip.open(v2path, "rt", encoding="utf-8", newline="") as src:
        reader = csv.DictReader(src)
        output_fields = [x for x in reader.fieldnames if x not in DROP_V2] + flow_fields()
        with gzip.open(temp, "wt", encoding="utf-8", newline="", compresslevel=6) as dst:
            writer = csv.DictWriter(dst, fieldnames=output_fields); writer.writeheader()
            for row in reader:
                minute = int(row["minute_start_timestamp"]); group = groups.pop(minute, empty_group())
                outrow = augment(row, group, threshold); writer.writerow(outrow)
                panel_rows += 1; panel_trades += int(row["trade_count"])
                panel_gross += float(row["gross_volume_usdc"]); panel_net += float(row["net_signed_flow_usdc"])
    errors = []
    if groups: errors.append(f"{len(groups)} trade minutes absent from v2")
    if panel_trades != stats["rows"]: errors.append(f"trade rows {panel_trades} != {stats['rows']}")
    tolerance = max(1e-5, abs(stats["gross"]) * 1e-10)
    if abs(panel_gross - stats["gross"]) > tolerance: errors.append("gross value does not reconcile")
    if abs(panel_net - stats["net"]) > tolerance: errors.append("net flow does not reconcile")
    if errors:
        temp.unlink(missing_ok=True); raise RuntimeError(f"Market {mid}: {'; '.join(errors)}")
    temp.replace(target)
    meta = {
        "status": "complete", "version": "v3", "market_id": int(mid),
        "source_trade_path": str(source.relative_to(ROOT)), "source_trade_sha256": source_hash,
        "source_v2_sha256": sha256(v2path), "threshold_table_sha256": threshold_hash,
        "trade_rows": stats["rows"], "market_minute_rows": panel_rows,
        "gross_volume_usdc": stats["gross"], "net_signed_flow_usdc": stats["net"],
        "output_sha256": sha256(target), "completed_at": now(), "errors": [],
    }
    atomic_json(meta_path, meta); return meta


def combine(progress: dict, require_all: bool = True) -> None:
    completed = progress["completed_markets"]
    if require_all and len(completed) != 360:
        raise RuntimeError(f"Expected 360 completed markets, observed {len(completed)}")
    first = OUT / "checkpoints" / f"{sorted(completed, key=int)[0]}.csv.gz"
    with gzip.open(first, "rt", encoding="utf-8", newline="") as handle:
        fieldnames = csv.DictReader(handle).fieldnames
    target = OUT / "market_minute_all_horizons.csv.gz"; temp = target.with_suffix(target.suffix + ".tmp")
    rows = trades = 0
    with gzip.open(temp, "wt", encoding="utf-8", newline="", compresslevel=6) as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames); writer.writeheader()
        for mid in sorted(completed, key=int):
            with gzip.open(OUT / "checkpoints" / f"{mid}.csv.gz", "rt", encoding="utf-8", newline="") as src:
                for row in csv.DictReader(src):
                    writer.writerow(row); rows += 1; trades += int(row["trade_count"])
    temp.replace(target)
    qc = {
        "version": "v3", "status": "built_qc_passed", "generated_at": now(),
        "markets_expected": 360, "markets_completed": len(completed),
        "market_minute_rows": rows, "trade_rows": trades,
        "threshold_table_sha256": sha256(THRESHOLDS), "output_sha256": sha256(target),
        "output_path": str(target.relative_to(ROOT)), "errors": [],
    }
    atomic_json(OUT / "analysis_ready_qc.json", qc)
    print(json.dumps(qc, indent=2), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["markets", "combine", "all"], default="all")
    parser.add_argument("--market-id", action="append")
    parser.add_argument("--max-markets", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-partial-combine", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    thresholds = load_table(THRESHOLDS, "market_id")
    manifest = load_table(TRADE_MANIFEST, "market_id")
    progress_path = OUT / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {
        "version": "v3", "created_at": now(), "completed_markets": {}, "failed": []}
    selected = sorted(manifest, key=int)
    if args.market_id: selected = [x for x in selected if x in set(args.market_id)]
    if args.max_markets is not None: selected = selected[:args.max_markets]
    if args.stage in ("markets", "all"):
        for index, mid in enumerate(selected, 1):
            try:
                meta = process_market(mid, ROOT / manifest[mid]["path"], thresholds[mid], args.force)
                progress["completed_markets"][mid] = meta
                progress["updated_at"] = now(); atomic_json(progress_path, progress)
                print(f"[{index}/{len(selected)}] {mid} rows={meta['market_minute_rows']:,}", flush=True)
            except Exception as exc:
                progress["failed"].append({"market_id": mid, "error": repr(exc), "at": now()})
                atomic_json(progress_path, progress); raise
    if args.stage in ("combine", "all"):
        combine(progress, not args.allow_partial_combine)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
