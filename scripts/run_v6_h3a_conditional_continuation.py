#!/usr/bin/env python3
"""H3a robustness conditional on an observed initial 0-to-5-minute move."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from run_v5_past_only_p99_h2 import fit, contrast

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "regression_inputs/v5/strict_timing/strict_timing_past_only_p99.csv.gz"
OUT = ROOT / "regression_results/v6/h3a_conditional_continuation"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["market_id", "event_id", "calendar_hour_utc", "v5_delta_5m", "v5_delta_15m",
            "v5_strict0_5m", "v5_strict0_15m", "v5_baseline_price",
            "v5_lagged_30m_price_change", "log_lagged_30m_volume", "log_minutes_to_kickoff"]
    for spec in ("expanding", "rolling7"):
        cols += [f"{spec}_eligible", f"{spec}_signed_log_large_net_flow", f"{spec}_signed_log_ordinary_net_flow"]
    data = pd.read_csv(SOURCE, usecols=cols)
    data["delta_5_to_15"] = data.v5_delta_15m - data.v5_delta_5m
    aligned = data.v5_strict0_5m.eq(1) & data.v5_strict0_15m.eq(1) & data.v5_delta_5m.ne(0)
    controls = ["v5_lagged_30m_price_change", "v5_baseline_price", "log_lagged_30m_volume", "log_minutes_to_kickoff"]
    coefficients = []; contrasts = []; diagnostics = []; descriptive = []
    for spec in ("expanding", "rolling7"):
        d = data.loc[aligned & data[f"{spec}_eligible"].eq(1)].dropna(subset=["delta_5_to_15"] + controls).copy()
        large = f"{spec}_signed_log_large_net_flow"; ordinary = f"{spec}_signed_log_ordinary_net_flow"
        model = f"{spec.upper()}_CONDITIONAL_INITIAL_UPDATE"
        print(f"Estimating {model}: {len(d):,}", flush=True)
        co, diag, beta, covariance, names, clusters = fit(d, model, "delta_5_to_15", [large, ordinary] + controls)
        coefficients += co; diagnostics.append(diag)
        con = contrast(model, "delta_5_to_15", "large_minus_ordinary", {large: 1, ordinary: -1}, beta, covariance, names, clusters)
        con["specification"] = spec; contrasts.append(con)
        descriptive.append({"specification": spec, "rows": len(d), "markets": d.market_id.nunique(),
                            "events": d.event_id.nunique(),
                            "same_direction_5_to_15_share": float((d.v5_delta_5m * d.delta_5_to_15 > 0).mean()),
                            "zero_5_to_15_share": float(d.delta_5_to_15.eq(0).mean())})
    pd.DataFrame(coefficients).to_csv(OUT / "coefficients.csv", index=False)
    pd.DataFrame(contrasts).to_csv(OUT / "contrasts.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(OUT / "model_diagnostics.csv", index=False)
    pd.DataFrame(descriptive).to_csv(OUT / "descriptive_rates.csv", index=False)
    summary = {"generated_at": datetime.now(timezone.utc).isoformat(), "status": "complete_qc_passed",
               "outcome": "price change from minute 5 to minute 15",
               "selection": "observed non-zero initial 0-to-5-minute recorded-price movement",
               "interpretation": "descriptive outcome-selected robustness; not the primary estimand"}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(pd.DataFrame(contrasts).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

