#!/usr/bin/env python3
"""Collapse transaction actions into 60-second execution sequences and retest continuation."""

from __future__ import annotations

import json
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "model_samples/v4/wallet_sequences/checkpoints"
COMBINED_ACTIONS = ROOT / "model_samples/v4/wallet_sequences/wallet_transaction_sequences_match_prematch.csv.gz"
HIGH_ACTIVITY_CLASSIFICATIONS = ROOT / "regression_results/v4/wallet_sequences_dynamic/daily_high_activity_wallets.csv.gz"
WHALE_CLASSIFICATIONS = ROOT / "regression_results/v4/past_only_whale_classifications/daily_whale_wallets.csv.gz"
OUT = ROOT / "regression_results/v6/wallet_60s_sequences"
START = pd.Timestamp("2026-06-15", tz="UTC")
END = pd.Timestamp("2026-07-23", tz="UTC")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["all", "high_activity", "whale_volume"],
                        default="all")
    args = parser.parse_args()
    output_dir = OUT if args.family == "all" else OUT.parent / f"wallet_60s_sequences_{args.family}"
    output_dir.mkdir(parents=True, exist_ok=True)
    high = pd.read_csv(HIGH_ACTIVITY_CLASSIFICATIONS,
                       usecols=["specification", "evaluation_day", "wallet_address"])
    high = high[high.specification.isin(["rolling_7d", "expanding"])]
    whale = pd.read_csv(WHALE_CLASSIFICATIONS,
                        usecols=["specification", "evaluation_day", "wallet_address"])
    whale = whale[whale.specification.isin(["expanding_top1_primary", "rolling7_top1"])]
    classes = pd.concat([
        high.assign(classification_family="high_activity"),
        whale.assign(classification_family="whale_volume"),
    ], ignore_index=True)
    if args.family != "all":
        classes = classes[classes.classification_family.eq(args.family)]
    class_sets = {(family, spec): set(zip(part.evaluation_day.astype(str),
                                          part.wallet_address.astype(str)))
                  for (family, spec), part in classes.groupby(["classification_family", "specification"])}
    cell_rows = []; source_actions = collapsed_actions = 0
    files = [COMBINED_ACTIONS] if COMBINED_ACTIONS.exists() else sorted(CHECKPOINTS.glob("*.csv.gz"))
    for index, path in enumerate(files, 1):
        cols = ["market_id", "event_id", "timestamp", "wallet_address", "direction", "large_action"]
        data = pd.read_csv(path, usecols=cols)
        time = pd.to_datetime(data.timestamp, unit="s", utc=True)
        data = data[(time >= START) & (time < END)].copy(); source_actions += len(data)
        data = data.sort_values(["market_id", "wallet_address", "timestamp"], kind="mergesort")
        grouping = [data.market_id, data.wallet_address]
        prior_ts = data.groupby(["market_id", "wallet_address"]).timestamp.shift()
        prior_dir = data.groupby(["market_id", "wallet_address"]).direction.shift()
        new_sequence = prior_ts.isna() | data.direction.ne(prior_dir) | data.timestamp.sub(prior_ts).gt(60)
        data["sequence_number"] = new_sequence.groupby(grouping).cumsum()
        seq = data.groupby(["market_id", "wallet_address", "sequence_number"], as_index=False).agg(
            event_id=("event_id", "first"),
            timestamp=("timestamp", "min"), direction=("direction", "first"),
            large_action=("large_action", "max"), transaction_actions=("timestamp", "size"))
        collapsed_actions += len(seq)
        seq = seq.sort_values(["market_id", "wallet_address", "timestamp"], kind="mergesort")
        next_ts = seq.groupby(["market_id", "wallet_address"]).timestamp.shift(-1)
        next_direction = seq.groupby(["market_id", "wallet_address"]).direction.shift(-1)
        seq["any_next_action_5m"] = next_ts.sub(seq.timestamp).between(0, 300, inclusive="both").astype(int)
        seq["same_direction_next_action_5m"] = (seq.any_next_action_5m.eq(1) & next_direction.eq(seq.direction)).astype(int)
        seq["evaluation_day"] = pd.to_datetime(seq.timestamp, unit="s", utc=True).dt.strftime("%Y-%m-%d")
        for (family, spec), selected in class_sets.items():
            seq["dynamic_high_activity"] = [int((d, w) in selected) for d, w in zip(seq.evaluation_day, seq.wallet_address)]
            grouped = seq.groupby(["event_id", "market_id", "large_action", "dynamic_high_activity"], as_index=False).agg(
                actions=("timestamp", "size"), any_next=("any_next_action_5m", "sum"),
                same_next=("same_direction_next_action_5m", "sum"),
                source_transaction_actions=("transaction_actions", "sum"))
            grouped["any_next_action_5m"] = grouped.any_next / grouped.actions
            grouped["same_direction_given_next_5m"] = grouped.same_next / grouped.any_next.replace(0, np.nan)
            grouped.insert(0, "specification", spec)
            grouped.insert(0, "classification_family", family)
            cell_rows.append(grouped)
        if index % 40 == 0 or index == len(files): print(f"[{index}/{len(files)}] {path.stem}", flush=True)
    market = pd.concat(cell_rows, ignore_index=True)
    event = market.groupby(["classification_family", "specification", "event_id", "large_action", "dynamic_high_activity"], as_index=False).agg(
        actions=("actions", "sum"), any_next=("any_next", "sum"), same_next=("same_next", "sum"))
    event["any_next_action_5m"] = event.any_next / event.actions
    event["same_direction_given_next_5m"] = event.same_next / event.any_next.replace(0, np.nan)
    tests = []
    for (family, spec), part in event.groupby(["classification_family", "specification"]):
        for large, label in ((0, "ordinary"), (1, "large")):
            selected = part[part.large_action.eq(large)]
            for outcome in ("any_next_action_5m", "same_direction_given_next_5m"):
                wide = selected.pivot(index="event_id", columns="dynamic_high_activity", values=outcome).dropna()
                if 0 not in wide or 1 not in wide: continue
                diff = wide[1] - wide[0]; n = len(diff); mean = diff.mean(); se = diff.std(ddof=1) / np.sqrt(n)
                t = mean / se; p_t = 2 * stats.t.sf(abs(t), n - 1)
                try: p_w = stats.wilcoxon(diff, zero_method="wilcox").pvalue
                except ValueError: p_w = np.nan
                tests.append({"classification_family": family, "specification": spec,
                              "trade_class": label, "outcome": outcome,
                              "events": n, "high_minus_other_mean": mean, "std_error": se,
                              "paired_t_p_value": p_t, "wilcoxon_p_value": p_w})
    market.to_csv(output_dir / "market_group_rates.csv", index=False)
    event.to_csv(output_dir / "event_group_rates.csv", index=False)
    pd.DataFrame(tests).to_csv(output_dir / "event_level_tests.csv", index=False)
    summary = {"generated_at": datetime.now(timezone.utc).isoformat(), "status": "complete_qc_passed",
               "source_transaction_actions": source_actions, "collapsed_60s_sequences": collapsed_actions,
               "collapse_share": 1 - collapsed_actions / source_actions,
               "sequence_rule": "same wallet, market and YES-equivalent direction with adjacent transaction actions no more than 60 seconds apart",
               "specifications": sorted([f"{family}:{spec}" for family, spec in class_sets]),
               "interpretation": "fragmented-execution robustness for high-activity and cumulative-gross-volume wallet classifications"}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(pd.DataFrame(tests).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
