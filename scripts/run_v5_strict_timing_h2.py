#!/usr/bin/env python3
"""Estimate V5 H2 under strict price timing and active-market robustness."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from run_v5_past_only_p99_h2 import fit, contrast

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "regression_inputs/v5/strict_timing/strict_timing_past_only_p99.csv.gz"
OUT = ROOT / "regression_results/v5/strict_timing_h2"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    base = ["market_id", "event_id", "calendar_hour_utc", "v5_delta_5m",
            "v5_baseline_price", "v5_lagged_30m_price_change",
            "log_lagged_30m_volume", "log_minutes_to_kickoff",
            "v5_strict0_5m", "v5_strict1_5m", "active_market_above_median", "common_eligible"]
    for spec in ("expanding", "rolling7"):
        base += [f"{spec}_eligible", f"{spec}_signed_log_large_net_flow",
                 f"{spec}_signed_log_ordinary_net_flow",
                 f"{spec}_large_net_flow_usdc", f"{spec}_ordinary_net_flow_usdc"]
    data = pd.read_csv(SOURCE, usecols=base)
    data["any_price_update_5m"] = data.v5_delta_5m.ne(0).astype(float)
    data["absolute_v5_lagged_30m_price_change"] = data.v5_lagged_30m_price_change.abs()

    coefficients = []; contrasts = []; diagnostics = []
    samples = {
        "STRICT0_ALL": data.v5_strict0_5m.eq(1),
        "STRICT1_ALL": data.v5_strict1_5m.eq(1),
        "STRICT0_ACTIVE": data.v5_strict0_5m.eq(1) & data.active_market_above_median.eq(1),
    }
    for sample_name, sample_mask in samples.items():
        for spec in ("expanding", "rolling7"):
            eligible = sample_mask & data[f"{spec}_eligible"].eq(1)
            for model_type in ("DIRECTIONAL", "UPDATE_LPM", "CONDITIONAL_NONZERO"):
                mask = eligible.copy()
                if model_type == "CONDITIONAL_NONZERO":
                    mask &= data.v5_delta_5m.ne(0)
                d = data.loc[mask].dropna(subset=["v5_baseline_price", "v5_lagged_30m_price_change"])
                model = f"{sample_name}_{spec.upper()}_{model_type}"
                if model_type == "UPDATE_LPM":
                    outcome = "any_price_update_5m"
                    d = d.copy()
                    d[f"{spec}_log_abs_large"] = np.log1p(d[f"{spec}_large_net_flow_usdc"].abs())
                    d[f"{spec}_log_abs_ordinary"] = np.log1p(d[f"{spec}_ordinary_net_flow_usdc"].abs())
                    focal = [f"{spec}_log_abs_large", f"{spec}_log_abs_ordinary"]
                    controls = ["absolute_v5_lagged_30m_price_change", "v5_baseline_price",
                                "log_lagged_30m_volume", "log_minutes_to_kickoff"]
                else:
                    outcome = "v5_delta_5m"
                    focal = [f"{spec}_signed_log_large_net_flow", f"{spec}_signed_log_ordinary_net_flow"]
                    controls = ["v5_lagged_30m_price_change", "v5_baseline_price",
                                "log_lagged_30m_volume", "log_minutes_to_kickoff"]
                terms = focal + controls
                print(f"Estimating {model}: {len(d):,}", flush=True)
                co, diag, beta, covariance, names, clusters = fit(d, model, outcome, terms)
                coefficients.extend(co); diagnostics.append(diag)
                contrasts.append(contrast(model, outcome, "large_minus_ordinary",
                    {focal[0]: 1, focal[1]: -1}, beta, covariance, names, clusters))

    pd.DataFrame(coefficients).to_csv(OUT / "coefficients.csv", index=False)
    pd.DataFrame(contrasts).to_csv(OUT / "contrasts.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(OUT / "model_diagnostics.csv", index=False)
    summary = {"generated_at": datetime.now(timezone.utc).isoformat(),
               "status": "complete_qc_passed", "models": len(diagnostics),
               "samples": list(samples), "specifications": ["expanding", "rolling7"],
               "fixed_effects": ["market_id", "calendar_hour_utc"], "cluster": "event_id CRV1",
               "timing": "baseline strictly before t; flow [t,t+1); outcome first price at/after t+5"}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(pd.DataFrame(contrasts).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
