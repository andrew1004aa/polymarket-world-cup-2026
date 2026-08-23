#!/usr/bin/env python3
"""H2a stratified direction-permutation tests with deterministic RNG."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "analysis_ready/v4/h2a_prepost_flow/h2a_p99_prepost_events_all.csv"
MARKETS = ROOT / "regression_ready/tables/markets.csv"
MAPPING = ROOT / "regression_ready/tables/event_market_mapping.csv"
OUT = ROOT / "regression_results/v4/h2a"
WINDOWS = ("1m", "5m", "15m")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024): digest.update(chunk)
    return digest.hexdigest()


def signed_log(value: float) -> float:
    return math.copysign(math.log1p(abs(value)), value) if value else 0.0


def load_metadata():
    market_type = {}
    with MARKETS.open(newline="") as handle:
        for row in csv.DictReader(handle): market_type[row["market_id"]] = row["market_type"]
    kickoff = {}
    with MAPPING.open(newline="") as handle:
        for row in csv.DictReader(handle): kickoff[row["market_id"]] = int(datetime.fromisoformat(row["actual_kickoff_utc"].replace("Z", "+00:00")).timestamp())
    return market_type, kickoff


def load_events(nonoverlap_seconds=0):
    market_type, kickoff = load_metadata(); events = []
    with SOURCE.open(newline="") as handle:
        for row in csv.DictReader(handle):
            timestamp = int(row["initiating_timestamp"]); mid = row["market_id"]; direction = int(row["initiating_direction"])
            family = market_type[mid]
            phase = "outright" if family == "outright" else ("pre" if timestamp < kickoff[mid] else "post")
            item = {"market_id": mid, "family": family, "phase": phase, "date": datetime.fromtimestamp(timestamp, timezone.utc).date().isoformat(), "timestamp": timestamp, "direction": direction}
            for tag in WINDOWS:
                same_pre = float(row[f"other_wallet_pre_same_value_{tag}"]); opp_pre = float(row[f"other_wallet_pre_opposite_value_{tag}"])
                same_post = float(row[f"other_wallet_post_same_value_{tag}"]); opp_post = float(row[f"other_wallet_post_opposite_value_{tag}"])
                if direction == 1:
                    plus_pre, minus_pre, plus_post, minus_post = same_pre, opp_pre, same_post, opp_post
                else:
                    plus_pre, minus_pre, plus_post, minus_post = opp_pre, same_pre, opp_post, same_post
                item[tag] = (plus_pre, minus_pre, plus_post, minus_post)
            events.append(item)
    if nonoverlap_seconds:
        retained=[]; last_by_market={}
        for event in sorted(events,key=lambda x:(int(x["market_id"]), x["timestamp"])):
            last=last_by_market.get(event["market_id"])
            if last is None or event["timestamp"]-last >= nonoverlap_seconds:
                retained.append(event); last_by_market[event["market_id"]]=event["timestamp"]
        events=retained
    return events


def statistic(event, tag, assigned_direction):
    plus_pre, minus_pre, plus_post, minus_post = event[tag]
    pre = assigned_direction * (plus_pre - minus_pre)
    post = assigned_direction * (plus_post - minus_post)
    return signed_log(post) - signed_log(pre), post - pre


def test(events, family, tag, permutations, seed):
    sample = [event for event in events if event["family"] == family]
    strata = defaultdict(list)
    for index, event in enumerate(sample): strata[(event["market_id"], event["date"], event["phase"])].append(index)
    observed_values = [statistic(event, tag, event["direction"]) for event in sample]
    observed_log = sum(value[0] for value in observed_values) / len(sample)
    observed_raw = sum(value[1] for value in observed_values) / len(sample)
    rng = random.Random(seed); null = []
    assigned = [event["direction"] for event in sample]
    groups = [(indices, [sample[index]["direction"] for index in indices]) for indices in strata.values()]
    # For absolute YES-equivalent flows, changing the assigned initiating
    # direction only changes the sign of the transformed post-minus-pre value.
    # Precompute the value for direction +1 to avoid repeated log transforms.
    base = [statistic(event, tag, 1)[0] for event in sample]
    for _ in range(permutations):
        for indices, original in groups:
            shuffled = original.copy(); rng.shuffle(shuffled)
            for index, value in zip(indices, shuffled): assigned[index] = value
        value = sum(assigned[index] * base[index] for index in range(len(sample))) / len(sample)
        null.append(value)
    null_mean = sum(null) / len(null)
    null_sd = math.sqrt(sum((value - null_mean) ** 2 for value in null) / max(1, len(null) - 1))
    upper_p = (1 + sum(value >= observed_log for value in null)) / (permutations + 1)
    lower_p = (1 + sum(value <= observed_log for value in null)) / (permutations + 1)
    two_p = min(1.0, 2 * min(upper_p, lower_p))
    return {
        "market_family": family, "window": tag, "events": len(sample), "strata": len(strata),
        "observed_mean_post_minus_pre_signed_log_directional_net": observed_log,
        "observed_mean_post_minus_pre_directional_net_usdc": observed_raw,
        "null_mean": null_mean, "null_sd": null_sd,
        "standardized_distance_from_null": (observed_log - null_mean) / null_sd if null_sd else None,
        "one_sided_upper_p": upper_p, "two_sided_p": two_p,
        "permutations": permutations, "seed": seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--permutations", type=int, default=1999); parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--nonoverlap-seconds", type=int, default=0)
    args = parser.parse_args(); events = load_events(args.nonoverlap_seconds); results = []
    family_labels = sorted({event["family"] for event in events})
    for family_index, family in enumerate(family_labels):
        for window_index, tag in enumerate(WINDOWS):
            result = test(events, family, tag, args.permutations, args.seed + family_index * 100 + window_index)
            results.append(result); print(json.dumps(result), flush=True)
    for family in family_labels:
        family_rows = [row for row in results if row["market_family"] == family]
        for field, output_field in (("one_sided_upper_p", "holm_one_sided_p"), ("two_sided_p", "holm_two_sided_p")):
            ordered = sorted(family_rows, key=lambda row: row[field]); running = 0.0; total = len(ordered)
            for rank, row in enumerate(ordered):
                adjusted = min(1.0, (total - rank) * row[field]); running = max(running, adjusted)
                row[output_field] = running
    output_dir = OUT if not args.nonoverlap_seconds else OUT / f"nonoverlap_{args.nonoverlap_seconds}s"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "h2a_permutation_results.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    summary = {"generated_at": datetime.now(timezone.utc).isoformat(), "status": "estimated",
               "hypothesis": "H2a", "test": "market-date-phase stratified initiating-direction permutation",
               "nonoverlap_seconds": args.nonoverlap_seconds,
               "source_path": str(SOURCE.relative_to(ROOT)), "source_sha256": sha256(SOURCE),
               "results": results,
               "interpretation": "Association/sequence evidence only; initiating direction is permuted while timestamps and observed other-wallet flows remain fixed."}
    (output_dir / "h2a_permutation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = ["# V4 H2a stratified permutation test", "", f"Permutations: {args.permutations}; base seed: {args.seed}.", "",
              "The initiating direction is shuffled within market × UTC date × kickoff phase. Event timestamps and other-wallet flows remain fixed.", "",
              "| Family | Window | N | Observed signed-log difference | Raw USDC difference | Null z-distance | Upper p | Holm upper p | Two-sided p | Holm two-sided p |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in results:
        report.append(f"| {row['market_family']} | {row['window']} | {row['events']:,} | {row['observed_mean_post_minus_pre_signed_log_directional_net']:.6f} | {row['observed_mean_post_minus_pre_directional_net_usdc']:.2f} | {row['standardized_distance_from_null']:.2f} | {row['one_sided_upper_p']:.4g} | {row['holm_one_sided_p']:.4g} | {row['two_sided_p']:.4g} | {row['holm_two_sided_p']:.4g} |")
    report += ["", "These results do not identify causality, trader intent, market making or manipulation."]
    (output_dir / "h2a_permutation_report.md").write_text("\n".join(report) + "\n")
    return 0


if __name__ == "__main__": raise SystemExit(main())
