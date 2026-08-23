#!/usr/bin/env python3
"""Past-only high-activity-wallet robustness for transaction-sequence tests."""

from __future__ import annotations

import csv
import gzip
import io
import json
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "model_samples/v4/wallet_sequences/checkpoints"
OUT = ROOT / "regression_results/v4/wallet_sequences_past_only"
SPECS = {
    "baseline_7d": int(datetime(2026, 6, 8, tzinfo=timezone.utc).timestamp()),
    "baseline_14d": int(datetime(2026, 6, 15, tzinfo=timezone.utc).timestamp()),
}
OUTCOMES = {
    "any_next_action_5m": ("all", "any_next_action_5m"),
    "same_direction_given_next_5m": ("any_next_action_5m", "same_direction_next_action_5m"),
    "opposite_action_60m": ("all", "opposite_action_within_60m"),
    "favorable_given_opposite_60m": ("opposite_action_within_60m", "favorable_opposite_action_within_60m"),
}
GROUPS = {"large_high": (1, 1), "large_other": (1, 0),
          "ordinary_high": (0, 1), "ordinary_other": (0, 0)}
CONTRASTS = {
    "high_activity_effect_among_large": {"large_high": 1, "large_other": -1},
    "high_activity_effect_among_ordinary": {"ordinary_high": 1, "ordinary_other": -1},
    "large_effect_among_high_activity": {"large_high": 1, "ordinary_high": -1},
    "large_effect_among_other": {"large_other": 1, "ordinary_other": -1},
    "difference_in_differences": {"large_high": 1, "large_other": -1,
                                   "ordinary_high": -1, "ordinary_other": 1},
}


def now(): return datetime.now(timezone.utc).isoformat()


def read_checkpoint(path: Path) -> list[dict]:
    for attempt in range(1, 6):
        try:
            with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
                return list(csv.DictReader(io.StringIO(f.read())))
        except TimeoutError:
            if attempt == 5: raise
            time.sleep(attempt)


def exact_top_one_percent(volumes: dict[str, float]):
    ranked = sorted(volumes.items(), key=lambda x: (-x[1], x[0]))
    count = max(1, math.ceil(len(ranked) * .01)); selected = {w for w, _ in ranked[:count]}
    threshold = ranked[count - 1][1]
    boundary_ties = sum(abs(v - threshold) <= 1e-12 for _, v in ranked)
    return selected, threshold, count, boundary_ties


def holm(values: pd.Series):
    p = values.to_numpy(float); order = np.argsort(p)
    adjusted_sorted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    adjusted = np.empty_like(p); adjusted[order] = np.minimum(adjusted_sorted, 1)
    return adjusted


def test_rates(market: pd.DataFrame, specification: str):
    rate_cols = list(OUTCOMES)
    event = market.groupby(["event_id", "large_action", "past_only_high_activity"], as_index=False)[rate_cols].mean()
    event["group"] = event.apply(lambda r: next(k for k, v in GROUPS.items()
        if v == (int(r.large_action), int(r.past_only_high_activity))), axis=1)
    event.insert(0, "specification", specification)
    wide = event.pivot(index="event_id", columns="group", values=rate_cols)
    tests = []
    for outcome in rate_cols:
        for label, weights in CONTRASTS.items():
            available = wide[outcome][list(weights)].dropna()
            diff = sum(available[g] * w for g, w in weights.items())
            n = len(diff); mean = float(diff.mean()); sd = float(diff.std(ddof=1)); se = sd / math.sqrt(n)
            t = mean / se if se else np.nan; df = n - 1; critical = float(stats.t.ppf(.975, df))
            p_t = float(2 * stats.t.sf(abs(t), df)) if math.isfinite(t) else np.nan
            try:
                test = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided", method="auto")
                w_stat, p_w = float(test.statistic), float(test.pvalue)
            except ValueError:
                w_stat, p_w = np.nan, np.nan
            tests.append({"specification": specification, "outcome": outcome, "contrast": label,
                "events": n, "mean_event_difference": mean, "median_event_difference": float(diff.median()),
                "std_error": se, "t_statistic": t, "t_df": df, "paired_t_p_value": p_t,
                "conf_low": mean - critical * se, "conf_high": mean + critical * se,
                "wilcoxon_statistic": w_stat, "wilcoxon_p_value": p_w,
                "weights": json.dumps(weights, sort_keys=True)})
    tests = pd.DataFrame(tests)
    tests["holm_p_paired_t_within_contrast"] = tests.groupby("contrast")["paired_t_p_value"].transform(holm)
    tests["holm_p_wilcoxon_within_contrast"] = tests.groupby("contrast")["wilcoxon_p_value"].transform(holm)
    return event, tests


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(CHECKPOINTS.glob("*.csv.gz"))
    if len(files) != 312: raise RuntimeError(f"Expected 312 checkpoints, found {len(files)}")
    volumes = {name: defaultdict(float) for name in SPECS}; ex_post = {}
    print("Pass 1/2: build past-only wallet classifications", flush=True)
    for i, path in enumerate(files, 1):
        for row in read_checkpoint(path):
            ts = int(row["timestamp"]); wallet = row["wallet_address"]
            ex_post[wallet] = int(row["high_activity_wallet"])
            value = float(row["action_value_usdc"])
            for name, cutoff in SPECS.items():
                if ts < cutoff: volumes[name][wallet] += value
        if i % 50 == 0: print(f"  {i}/312 markets", flush=True)
    selected, classification_rows, diagnostics = {}, [], []
    for name in SPECS:
        chosen, threshold, count, ties = exact_top_one_percent(volumes[name]); selected[name] = chosen
        active = set(volumes[name]); ex_active = {w for w in active if ex_post.get(w, 0) == 1}; overlap = chosen & ex_active
        diagnostics.append({"specification": name, "cutoff_timestamp": SPECS[name],
            "cutoff_utc": datetime.fromtimestamp(SPECS[name], timezone.utc).isoformat(),
            "baseline_active_wallets": len(active), "past_only_high_wallets": count,
            "threshold_usdc": threshold, "wallets_tied_at_boundary": ties,
            "overlap_with_ex_post_high": len(overlap), "precision_vs_ex_post": len(overlap) / len(chosen),
            "recall_among_baseline_active_ex_post": len(overlap) / len(ex_active) if ex_active else np.nan,
            "jaccard_among_baseline_active": len(overlap) / len(chosen | ex_active)})
        for wallet, value in volumes[name].items():
            classification_rows.append({"specification": name, "wallet_address": wallet,
                "baseline_volume_usdc": value, "past_only_high_activity": int(wallet in chosen),
                "ex_post_high_activity": ex_post.get(wallet, 0)})
    pd.DataFrame(classification_rows).to_csv(OUT / "past_only_wallet_classifications.csv.gz", index=False, compression="gzip")
    pd.DataFrame(diagnostics).to_csv(OUT / "classification_diagnostics.csv", index=False)

    cells = {name: defaultdict(lambda: defaultdict(int)) for name in SPECS}
    evaluation_actions = defaultdict(int)
    print("Pass 2/2: aggregate out-of-baseline evaluation actions", flush=True)
    for i, path in enumerate(files, 1):
        for row in read_checkpoint(path):
            ts = int(row["timestamp"]); wallet = row["wallet_address"]
            for name, cutoff in SPECS.items():
                if ts < cutoff: continue
                evaluation_actions[name] += 1
                high = int(wallet in selected[name])
                key = (row["event_id"], row["market_id"], int(row["large_action"]), high)
                cell = cells[name][key]; cell["all"] += 1
                cell["any_next_action_5m"] += int(row["any_next_action_5m"])
                cell["same_direction_next_action_5m"] += int(row["same_direction_next_action_5m"])
                cell["opposite_action_within_60m"] += int(row["opposite_action_within_60m"])
                cell["favorable_opposite_action_within_60m"] += int(row["favorable_opposite_action_within_60m"])
        if i % 50 == 0: print(f"  {i}/312 markets", flush=True)
    all_market, all_event, all_tests = [], [], []
    for name, spec_cells in cells.items():
        rows = []
        for (event_id, market_id, large, high), cell in spec_cells.items():
            row = {"specification": name, "event_id": event_id, "market_id": market_id,
                   "large_action": large, "past_only_high_activity": high, "actions": cell["all"]}
            for outcome, (denominator, numerator) in OUTCOMES.items():
                den = cell[denominator]; row[f"{outcome}_denominator"] = den
                row[outcome] = cell[numerator] / den if den else np.nan
            rows.append(row)
        market = pd.DataFrame(rows); event, tests = test_rates(market, name)
        all_market.append(market); all_event.append(event); all_tests.append(tests)
    market = pd.concat(all_market, ignore_index=True); event = pd.concat(all_event, ignore_index=True)
    tests = pd.concat(all_tests, ignore_index=True)
    market.to_csv(OUT / "market_group_sequence_rates.csv", index=False)
    event.to_csv(OUT / "event_group_sequence_rates.csv", index=False)
    tests.to_csv(OUT / "event_level_paired_tests.csv", index=False)
    summary = {"generated_at": now(), "status": "complete_qc_passed",
        "design": "fixed baseline classification followed by strictly later evaluation",
        "tie_rule": "exact ceil(1%) selected; descending volume then ascending wallet address",
        "specifications": diagnostics, "evaluation_actions": dict(evaluation_actions),
        "markets": int(market.market_id.nunique()), "events": int(market.event_id.nunique()),
        "tests": int(len(tests)), "multiplicity": "Holm across four outcomes within specification and contrast",
        "limitations": ["classification uses only observed World Cup match-market activity",
                        "baseline-inactive future wallets are classified as other",
                        "favourable opposite action is not realised profit"]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(tests[(tests.outcome == "same_direction_given_next_5m") |
                (tests.outcome == "favorable_given_opposite_60m")][
        ["specification", "outcome", "contrast", "events", "mean_event_difference",
         "holm_p_paired_t_within_contrast", "holm_p_wilcoxon_within_contrast"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
