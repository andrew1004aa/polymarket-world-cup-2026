#!/usr/bin/env python3
"""Estimate dynamic-wallet price models using the audited sparse FWL engine."""

import json
from pathlib import Path

import run_v4_past_only_price_2x2_regressions as engine

ROOT = Path(__file__).resolve().parents[1]
engine.INPUT = ROOT / "model_samples/v4/dynamic_price_2x2"
engine.OUT = ROOT / "regression_results/v4/dynamic_price_2x2"
engine.SPECS = ("rolling_7d", "expanding", "ex_post_descriptive")

if __name__ == "__main__":
    status = engine.main()
    summary_path = engine.OUT / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary.update({
        "design": "daily dynamic wallet classifications on a common evaluation sample",
        "primary": "rolling trailing-seven-day top 1% fixed before each UTC day",
        "robustness": "expanding-history top 1% fixed before each UTC day",
        "descriptive": "full-window ex-post top 1%",
        "interpretation": "conditional associations; ex-post specification is descriptive only",
    })
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    raise SystemExit(status)
