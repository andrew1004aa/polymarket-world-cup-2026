#!/usr/bin/env python3
"""Run reversal models for past-only whale-wallet classifications."""

from pathlib import Path
import pandas as pd
import run_v4_large_wallet_2x2_reversal as engine

ROOT = Path(__file__).resolve().parents[1]
SPECS = ("expanding_top1_primary", "rolling7_top1", "expanding_top05",
         "expanding_top2", "ex_post_top1_descriptive")


def main() -> int:
    engine.GROUPS = ("large_high", "large_other", "ordinary_high", "ordinary_other")
    engine.FILTER_TO_PANEL = True
    for specification in SPECS:
        print(f"Running whale reversal models: {specification}", flush=True)
        engine.PANEL = ROOT / f"model_samples/v4/past_only_whale_price_2x2/{specification}_price_2x2.csv.gz"
        engine.SAMPLE_OUT = ROOT / f"model_samples/v4/past_only_whale_reversal/{specification}_sample.csv.gz"
        engine.OUT = ROOT / f"regression_results/v4/past_only_whale_reversal/{specification}"
        engine.main()
        path = engine.OUT / "contrasts.csv"
        contrasts = pd.read_csv(path)
        contrasts["contrast"] = contrasts["contrast"].str.replace("high_activity", "whale", regex=False)
        contrasts.to_csv(path, index=False)
    return 0


if __name__ == "__main__": raise SystemExit(main())
