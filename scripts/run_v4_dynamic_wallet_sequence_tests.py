#!/usr/bin/env python3
"""Build dynamic wallet classes and re-estimate transaction-sequence contrasts.

Primary: daily trailing-seven-day top 1% by gross USDC among wallets with at
least five actions OR three active days. Robustness: expanding history from
2026-06-01. Ex-post full-window status is descriptive only.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import math
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "model_samples/v4/wallet_sequences/checkpoints"
OUT = ROOT / "regression_results/v4/wallet_sequences_dynamic"
START_HISTORY = date(2026, 6, 1)
START_EVALUATION = date(2026, 6, 15)
END_EVALUATION = date(2026, 7, 22)
SPECS = ("rolling_7d", "expanding", "ex_post_descriptive")
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


def now():
    return datetime.now(timezone.utc).isoformat()


def utc_day(timestamp: int) -> date:
    return datetime.fromtimestamp(timestamp, timezone.utc).date()


def day_range(first: date, last: date):
    current = first
    while current <= last:
        yield current
        current += timedelta(days=1)


def read_checkpoint(path: Path) -> list[dict]:
    for attempt in range(1, 6):
        try:
            with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
                return list(csv.DictReader(io.StringIO(handle.read())))
        except TimeoutError:
            if attempt == 5:
                raise
            time.sleep(attempt)


def select_top_one_percent(stats_by_wallet: dict[str, list[float]]):
    eligible = [(wallet, values[0], int(values[1]), int(values[2]))
                for wallet, values in stats_by_wallet.items()
                if values[1] >= 5 or values[2] >= 3]
    eligible.sort(key=lambda row: (-row[1], row[0]))
    if not eligible:
        return set(), math.nan, 0, 0
    count = max(1, math.ceil(len(eligible) * .01))
    threshold = eligible[count - 1][1]
    selected = {row[0] for row in eligible[:count]}
    ties = sum(abs(row[1] - threshold) <= 1e-12 for row in eligible)
    return selected, threshold, len(eligible), ties


def add_daily(target: dict[str, list[float]], source: dict[str, list[float]], sign=1):
    for wallet, (volume, actions) in source.items():
        item = target.setdefault(wallet, [0.0, 0, 0])
        item[0] += sign * volume
        item[1] += sign * actions
        item[2] += sign
        if item[1] <= 0:
            target.pop(wallet, None)


def holm(series: pd.Series):
    values = series.to_numpy(float)
    order = np.argsort(values)
    adjusted_sorted = np.maximum.accumulate((len(values) - np.arange(len(values))) * values[order])
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(adjusted_sorted, 1)
    return adjusted


def test_rates(market: pd.DataFrame):
    rates = list(OUTCOMES)
    event = market.groupby(["specification", "event_id", "large_action", "dynamic_high_activity"],
                           as_index=False)[rates].mean()
    event["group"] = event.apply(lambda row: next(
        name for name, pair in GROUPS.items()
        if pair == (int(row.large_action), int(row.dynamic_high_activity))), axis=1)
    tests = []
    for specification, part in event.groupby("specification"):
        wide = part.pivot(index="event_id", columns="group", values=rates)
        for outcome in rates:
            for label, weights in CONTRASTS.items():
                available = wide[outcome][list(weights)].dropna()
                difference = sum(available[group] * weight for group, weight in weights.items())
                n = len(difference)
                mean = float(difference.mean())
                sd = float(difference.std(ddof=1))
                se = sd / math.sqrt(n) if n else math.nan
                statistic = mean / se if se and math.isfinite(se) else math.nan
                degrees = n - 1
                p_t = float(2 * stats.t.sf(abs(statistic), degrees)) if degrees > 0 and math.isfinite(statistic) else math.nan
                critical = float(stats.t.ppf(.975, degrees)) if degrees > 0 else math.nan
                try:
                    wilcoxon = stats.wilcoxon(difference, zero_method="wilcox", alternative="two-sided", method="auto")
                    w_stat, p_w = float(wilcoxon.statistic), float(wilcoxon.pvalue)
                except ValueError:
                    w_stat, p_w = math.nan, math.nan
                tests.append({"specification": specification, "outcome": outcome, "contrast": label,
                    "events": n, "mean_event_difference": mean,
                    "median_event_difference": float(difference.median()), "std_error": se,
                    "t_statistic": statistic, "t_df": degrees, "paired_t_p_value": p_t,
                    "conf_low": mean - critical * se, "conf_high": mean + critical * se,
                    "wilcoxon_statistic": w_stat, "wilcoxon_p_value": p_w,
                    "weights": json.dumps(weights, sort_keys=True)})
    tests = pd.DataFrame(tests)
    tests["holm_p_paired_t_within_spec_contrast"] = tests.groupby(
        ["specification", "contrast"])["paired_t_p_value"].transform(holm)
    tests["holm_p_wilcoxon_within_spec_contrast"] = tests.groupby(
        ["specification", "contrast"])["wilcoxon_p_value"].transform(holm)
    return event, tests


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(CHECKPOINTS.glob("*.csv.gz"))
    if len(files) != 312:
        raise RuntimeError(f"Expected 312 checkpoints, found {len(files)}")

    daily: dict[date, dict[str, list[float]]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    ex_post = set()
    print("Pass 1/3: aggregate wallet-day histories", flush=True)
    for index, path in enumerate(files, 1):
        for row in read_checkpoint(path):
            timestamp = int(row["timestamp"])
            day = utc_day(timestamp)
            wallet = row["wallet_address"]
            item = daily[day][wallet]
            item[0] += float(row["action_value_usdc"])
            item[1] += 1
            if int(row["high_activity_wallet"]):
                ex_post.add(wallet)
        if index % 40 == 0:
            print(f"  {index}/312 markets", flush=True)

    classifications: dict[str, dict[date, set[str]]] = {name: {} for name in SPECS}
    diagnostics = []
    rolling: dict[str, list[float]] = {}
    expanding: dict[str, list[float]] = {}
    print("Pass 2/3: freeze daily past-only classifications", flush=True)
    for current in day_range(START_HISTORY, END_EVALUATION):
        previous = current - timedelta(days=1)
        if previous in daily:
            add_daily(expanding, daily[previous])
            add_daily(rolling, daily[previous])
        expired = current - timedelta(days=8)
        if expired in daily:
            add_daily(rolling, daily[expired], sign=-1)
        if current < START_EVALUATION:
            continue
        for specification, history in (("rolling_7d", rolling), ("expanding", expanding)):
            selected, threshold, eligible, ties = select_top_one_percent(history)
            classifications[specification][current] = selected
            diagnostics.append({"specification": specification, "evaluation_day": current.isoformat(),
                "history_start": ((current - timedelta(days=7)) if specification == "rolling_7d" else START_HISTORY).isoformat(),
                "history_end_exclusive": current.isoformat(), "eligible_wallets": eligible,
                "high_activity_wallets": len(selected), "threshold_usdc": threshold,
                "wallets_tied_at_boundary": ties})
        classifications["ex_post_descriptive"][current] = ex_post
        diagnostics.append({"specification": "ex_post_descriptive", "evaluation_day": current.isoformat(),
            "history_start": START_HISTORY.isoformat(), "history_end_exclusive": "full_window",
            "eligible_wallets": math.nan, "high_activity_wallets": len(ex_post),
            "threshold_usdc": math.nan, "wallets_tied_at_boundary": math.nan})
    pd.DataFrame(diagnostics).to_csv(OUT / "daily_classification_diagnostics.csv", index=False)
    high_rows = [{"specification": spec, "evaluation_day": day.isoformat(), "wallet_address": wallet}
                 for spec, by_day in classifications.items() for day, wallets in by_day.items()
                 for wallet in sorted(wallets)]
    pd.DataFrame(high_rows).to_csv(OUT / "daily_high_activity_wallets.csv.gz", index=False, compression="gzip")

    cells = {name: defaultdict(lambda: defaultdict(int)) for name in SPECS}
    evaluation_actions = defaultdict(int)
    print("Pass 3/3: aggregate evaluation-period transaction sequences", flush=True)
    for index, path in enumerate(files, 1):
        for row in read_checkpoint(path):
            day = utc_day(int(row["timestamp"]))
            if day < START_EVALUATION or day > END_EVALUATION:
                continue
            wallet = row["wallet_address"]
            for specification in SPECS:
                high = int(wallet in classifications[specification][day])
                key = (row["event_id"], row["market_id"], int(row["large_action"]), high)
                cell = cells[specification][key]
                cell["all"] += 1
                for _, numerator in OUTCOMES.values():
                    cell[numerator] += int(row[numerator])
                evaluation_actions[specification] += 1
        if index % 40 == 0:
            print(f"  {index}/312 markets", flush=True)

    market_rows = []
    for specification, spec_cells in cells.items():
        for (event_id, market_id, large, high), cell in spec_cells.items():
            output = {"specification": specification, "event_id": event_id, "market_id": market_id,
                      "large_action": large, "dynamic_high_activity": high, "actions": cell["all"]}
            for outcome, (denominator, numerator) in OUTCOMES.items():
                den = cell[denominator]
                output[f"{outcome}_denominator"] = den
                output[outcome] = cell[numerator] / den if den else math.nan
            market_rows.append(output)
    market = pd.DataFrame(market_rows)
    event, tests = test_rates(market)
    market.to_csv(OUT / "market_group_sequence_rates.csv", index=False)
    event.to_csv(OUT / "event_group_sequence_rates.csv", index=False)
    tests.to_csv(OUT / "event_level_paired_tests.csv", index=False)
    summary = {"generated_at": now(), "status": "complete_qc_passed",
        "evaluation_period": {"start_inclusive": START_EVALUATION.isoformat(),
                              "end_inclusive": END_EVALUATION.isoformat()},
        "primary": "daily rolling seven-day top 1% by gross USDC",
        "robustness": "daily expanding-history top 1% by gross USDC",
        "descriptive": "full-window ex-post top 1%",
        "eligibility": "at least five actions OR three active days in the applicable history window",
        "classification_timing": "fixed before each UTC evaluation day",
        "evaluation_actions": dict(evaluation_actions), "markets": int(market.market_id.nunique()),
        "events": int(market.event_id.nunique()), "tests": int(len(tests)),
        "multiplicity": "Holm across four outcomes within specification and contrast",
        "limitations": ["activity is observed only within collected World Cup match markets",
                        "favourable opposite action is not realised profit"]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(tests[tests.outcome.isin(["same_direction_given_next_5m", "favorable_given_opposite_60m"])][
        ["specification", "outcome", "contrast", "events", "mean_event_difference",
         "holm_p_paired_t_within_spec_contrast", "holm_p_wilcoxon_within_spec_contrast"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
