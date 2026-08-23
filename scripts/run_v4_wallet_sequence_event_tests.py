#!/usr/bin/env python3
"""Event-level paired tests for Large x high-activity wallet sequence behaviour."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "model_samples/v4/wallet_sequences/wallet_transaction_sequences_match_prematch.csv.gz"
OUT = ROOT / "regression_results/v4/wallet_sequences"
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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1024 * 1024): h.update(block)
    return h.hexdigest()


def holm(values: pd.Series) -> np.ndarray:
    p = values.to_numpy(float); order = np.argsort(p)
    adjusted_sorted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    adjusted = np.empty_like(p); adjusted[order] = np.minimum(adjusted_sorted, 1)
    return adjusted


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    # key -> denominator plus four outcome numerators/conditional denominators
    cells = defaultdict(lambda: defaultdict(int))
    print("Streaming 2.9m transaction actions", flush=True)
    with gzip.open(SRC, "rt", encoding="utf-8", newline="") as f:
        for i, row in enumerate(csv.DictReader(f), 1):
            key = (row["event_id"], row["market_id"], int(row["large_action"]),
                   int(row["high_activity_wallet"]))
            cell = cells[key]; cell["all"] += 1
            any_next = int(row["any_next_action_5m"])
            opposite = int(row["opposite_action_within_60m"])
            cell["any_next_action_5m"] += any_next
            cell["same_direction_next_action_5m"] += int(row["same_direction_next_action_5m"])
            cell["opposite_action_within_60m"] += opposite
            cell["favorable_opposite_action_within_60m"] += int(row["favorable_opposite_action_within_60m"])
            if i % 500_000 == 0: print(f"  {i:,} actions", flush=True)
    market_rows = []
    for (event, market, large, high), cell in cells.items():
        row = {"event_id": event, "market_id": market, "large_action": large,
               "high_activity_wallet": high, "actions": cell["all"]}
        for outcome, (denominator, numerator) in OUTCOMES.items():
            den = cell[denominator]; row[f"{outcome}_denominator"] = den
            row[outcome] = cell[numerator] / den if den else np.nan
        market_rows.append(row)
    market = pd.DataFrame(market_rows)
    market.to_csv(OUT / "market_group_sequence_rates.csv", index=False)
    # Equal-weight markets within each FIFA event, then inference across events.
    rate_cols = list(OUTCOMES)
    event = market.groupby(["event_id", "large_action", "high_activity_wallet"], as_index=False)[rate_cols].mean()
    event["group"] = event.apply(lambda r: next(k for k, v in GROUPS.items()
        if v == (int(r.large_action), int(r.high_activity_wallet))), axis=1)
    event.to_csv(OUT / "event_group_sequence_rates.csv", index=False)
    wide = event.pivot(index="event_id", columns="group", values=rate_cols)
    rows = []
    for outcome in rate_cols:
        for label, weights in CONTRASTS.items():
            required = list(weights); available = wide[outcome][required].dropna()
            differences = sum(available[group] * weight for group, weight in weights.items())
            n = len(differences); mean = float(differences.mean()); sd = float(differences.std(ddof=1))
            se = sd / math.sqrt(n); t = mean / se if se else np.nan; df = n - 1
            p_t = float(2 * stats.t.sf(abs(t), df)) if math.isfinite(t) else np.nan
            critical = float(stats.t.ppf(.975, df))
            try:
                w = stats.wilcoxon(differences, zero_method="wilcox", alternative="two-sided", method="auto")
                w_stat, p_w = float(w.statistic), float(w.pvalue)
            except ValueError:
                w_stat, p_w = np.nan, np.nan
            rows.append({"outcome": outcome, "contrast": label, "events": n,
                "mean_event_difference": mean, "median_event_difference": float(differences.median()),
                "std_dev_event_difference": sd, "std_error": se, "t_statistic": t,
                "t_df": df, "paired_t_p_value": p_t, "conf_low": mean - critical * se,
                "conf_high": mean + critical * se, "wilcoxon_statistic": w_stat,
                "wilcoxon_p_value": p_w, "weights": json.dumps(weights, sort_keys=True)})
    tests = pd.DataFrame(rows)
    tests["holm_p_paired_t_within_contrast"] = tests.groupby("contrast")["paired_t_p_value"].transform(holm)
    tests["holm_p_wilcoxon_within_contrast"] = tests.groupby("contrast")["wilcoxon_p_value"].transform(holm)
    tests.to_csv(OUT / "event_level_paired_tests.csv", index=False)
    payload = {"generated_at": now(), "status": "complete_qc_passed",
        "source": str(SRC.relative_to(ROOT)), "source_sha256": sha256(SRC),
        "actions": int(market.actions.sum()), "markets": int(market.market_id.nunique()),
        "events": int(market.event_id.nunique()), "market_group_cells": int(len(market)),
        "method": "market-group rates, equal-weighted within event, one-sample paired t and Wilcoxon tests across events",
        "multiplicity": "Holm across four sequence outcomes within each contrast family",
        "outcomes": {"any_next_action_5m": "unconditional",
                     "same_direction_given_next_5m": "conditional on any next action within 5m",
                     "opposite_action_60m": "unconditional first opposite action within 60m",
                     "favorable_given_opposite_60m": "conditional on opposite action within 60m; price-improvement proxy only"},
        "software": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__}}
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(tests[["outcome", "contrast", "events", "mean_event_difference", "paired_t_p_value",
                 "holm_p_paired_t_within_contrast", "wilcoxon_p_value",
                 "holm_p_wilcoxon_within_contrast"]].to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
