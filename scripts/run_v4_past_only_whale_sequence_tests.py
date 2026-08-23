#!/usr/bin/env python3
"""Estimate wallet-sequence contrasts under past-only whale definitions."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

import run_v4_dynamic_wallet_sequence_tests as helper

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ROOT / "regression_results/v4/past_only_whale_classifications/daily_whale_wallets.csv.gz"
CHECKPOINTS = ROOT / "model_samples/v4/wallet_sequences/checkpoints"
OUT = ROOT / "regression_results/v4/past_only_whale_sequences"
START = date(2026, 6, 15); END = date(2026, 7, 22)
SPECS = ("expanding_top1_primary", "rolling7_top1", "expanding_top05",
         "expanding_top2", "ex_post_top1_descriptive")


def now(): return datetime.now(timezone.utc).isoformat()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(CLASSES)
    whales = {(specification, day): set(part.wallet_address)
              for (specification, day), part in table.groupby(["specification", "evaluation_day"])}
    cells = {specification: defaultdict(lambda: defaultdict(int)) for specification in SPECS}
    evaluation_actions = defaultdict(int)
    files = sorted(CHECKPOINTS.glob("*.csv.gz"))
    if len(files) != 312: raise RuntimeError(f"Expected 312 checkpoints, found {len(files)}")
    for index, path in enumerate(files, 1):
        for row in helper.read_checkpoint(path):
            day = helper.utc_day(int(row["timestamp"]))
            if day < START or day > END: continue
            for specification in SPECS:
                whale = int(row["wallet_address"] in whales.get((specification, day.isoformat()), set()))
                key = (row["event_id"], row["market_id"], int(row["large_action"]), whale)
                cell = cells[specification][key]; cell["all"] += 1
                for _, numerator in helper.OUTCOMES.values(): cell[numerator] += int(row[numerator])
                evaluation_actions[specification] += 1
        if index % 40 == 0: print(f"  {index}/312 markets", flush=True)
    rows = []
    for specification, spec_cells in cells.items():
        for (event_id, market_id, large, whale), cell in spec_cells.items():
            output = {"specification": specification, "event_id": event_id, "market_id": market_id,
                      "large_action": large, "dynamic_high_activity": whale, "actions": cell["all"]}
            for outcome, (denominator, numerator) in helper.OUTCOMES.items():
                den = cell[denominator]; output[f"{outcome}_denominator"] = den
                output[outcome] = cell[numerator] / den if den else float("nan")
            rows.append(output)
    market = pd.DataFrame(rows); event, tests = helper.test_rates(market)
    tests["contrast"] = tests["contrast"].str.replace("high_activity", "whale", regex=False)
    market.to_csv(OUT / "market_group_sequence_rates.csv", index=False)
    event.to_csv(OUT / "event_group_sequence_rates.csv", index=False)
    tests.to_csv(OUT / "event_level_paired_tests.csv", index=False)
    summary = {"generated_at": now(), "status": "complete_qc_passed",
        "evaluation_start": START.isoformat(), "evaluation_end_inclusive": END.isoformat(),
        "specifications": list(SPECS), "evaluation_actions": dict(evaluation_actions),
        "markets": int(market.market_id.nunique()), "events": int(market.event_id.nunique()),
        "tests": int(len(tests)), "multiplicity": "Holm across four outcomes within specification and contrast",
        "interpretation": "whale-wallet proxy based on past gross USDC; no activity-count minimum"}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(tests[tests.outcome.isin(["same_direction_given_next_5m", "favorable_given_opposite_60m"])][
        ["specification", "outcome", "contrast", "events", "mean_event_difference",
         "holm_p_paired_t_within_spec_contrast", "holm_p_wilcoxon_within_spec_contrast"]].to_string(index=False))
    return 0


if __name__ == "__main__": raise SystemExit(main())
