#!/usr/bin/env python3
"""Estimate price models for past-only whale-wallet classifications."""

import json
from pathlib import Path
import run_v4_past_only_price_2x2_regressions as engine

ROOT = Path(__file__).resolve().parents[1]
engine.INPUT = ROOT / "model_samples/v4/past_only_whale_price_2x2"
engine.OUT = ROOT / "regression_results/v4/past_only_whale_price_2x2"
engine.SPECS = ("expanding_top1_primary", "rolling7_top1", "expanding_top05",
                "expanding_top2", "ex_post_top1_descriptive")

if __name__ == "__main__":
    status = engine.main()
    import pandas as pd
    contrasts_path = engine.OUT / "contrasts.csv"
    contrasts = pd.read_csv(contrasts_path)
    contrasts["contrast"] = contrasts["contrast"].str.replace("high_activity", "whale", regex=False)
    contrasts.to_csv(contrasts_path, index=False)
    path = engine.OUT / "summary.json"; summary = json.loads(path.read_text())
    summary.update({"design": "past-only whale-wallet proxy on a common evaluation sample",
        "primary": "daily expanding cumulative gross USDC top 1%; no activity-count minimum",
        "robustness": ["rolling seven-day top 1%", "expanding top 0.5%", "expanding top 2%"],
        "descriptive": "original full-window top 1%", "interpretation": "conditional associations"})
    path.write_text(json.dumps(summary, indent=2) + "\n"); raise SystemExit(status)
