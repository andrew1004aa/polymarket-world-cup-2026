#!/usr/bin/env python3
"""Run 2x2 reversal models for rolling, expanding, and ex-post wallet status."""

from pathlib import Path

import run_v4_large_wallet_2x2_reversal as engine

ROOT = Path(__file__).resolve().parents[1]
SPECS = ("rolling_7d", "expanding", "ex_post_descriptive")


def main() -> int:
    engine.GROUPS = ("large_high", "large_other", "ordinary_high", "ordinary_other")
    engine.FILTER_TO_PANEL = True
    for specification in SPECS:
        print(f"Running reversal models: {specification}", flush=True)
        engine.PANEL = ROOT / f"model_samples/v4/dynamic_price_2x2/{specification}_price_2x2.csv.gz"
        engine.SAMPLE_OUT = ROOT / f"model_samples/v4/dynamic_wallet_reversal/{specification}_sample.csv.gz"
        engine.OUT = ROOT / f"regression_results/v4/dynamic_wallet_reversal/{specification}"
        engine.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
