#!/usr/bin/env python3
"""Build 2x2 price panels for past-only whale-wallet classifications."""

from pathlib import Path
import build_v4_dynamic_price_2x2_panels as engine

ROOT = Path(__file__).resolve().parents[1]
engine.CLASSES = ROOT / "regression_results/v4/past_only_whale_classifications/daily_whale_wallets.csv.gz"
engine.OUT = ROOT / "model_samples/v4/past_only_whale_price_2x2"
engine.SPECS = ("expanding_top1_primary", "rolling7_top1", "expanding_top05",
                "expanding_top2", "ex_post_top1_descriptive")

if __name__ == "__main__": raise SystemExit(engine.main())
