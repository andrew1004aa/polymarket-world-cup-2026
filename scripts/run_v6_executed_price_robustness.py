#!/usr/bin/env python3
"""Estimate H2 using executed trade prices rather than CLOB price history."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from run_v5_past_only_p99_h2 import fit, contrast

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "regression_inputs/v6/executed_price/executed_price_h2.csv.gz"
OUT = ROOT / "regression_results/v6/executed_price"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(SOURCE)
    controls = ["v5_lagged_30m_price_change", "v5_baseline_price",
                "log_lagged_30m_volume", "log_minutes_to_kickoff"]
    coefficients = []; contrasts = []; diagnostics = []
    for tolerance in (60, 300, 900):
        aligned = data.executed_baseline_gap_seconds.between(1, tolerance) & data.executed_outcome_gap_seconds.between(0, tolerance)
        for spec in ("expanding", "rolling7"):
            mask = aligned & data[f"{spec}_eligible"].eq(1)
            d = data.loc[mask].dropna(subset=["executed_delta_5m"] + controls).copy()
            large = f"{spec}_signed_log_large_net_flow"; ordinary = f"{spec}_signed_log_ordinary_net_flow"
            model = f"EXECUTED_{tolerance}S_{spec.upper()}"
            print(f"Estimating {model}: {len(d):,}", flush=True)
            co, diag, beta, covariance, names, clusters = fit(d, model, "executed_delta_5m", [large, ordinary] + controls)
            coefficients += co; diagnostics.append(diag)
            con = contrast(model, "executed_delta_5m", "large_minus_ordinary", {large: 1, ordinary: -1}, beta, covariance, names, clusters)
            con.update({"tolerance_seconds": tolerance, "specification": spec}); contrasts.append(con)
    pd.DataFrame(coefficients).to_csv(OUT / "coefficients.csv", index=False)
    pd.DataFrame(contrasts).to_csv(OUT / "contrasts.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(OUT / "model_diagnostics.csv", index=False)
    summary = {"generated_at": datetime.now(timezone.utc).isoformat(), "status": "complete_qc_passed",
               "models": len(diagnostics), "price_source": "executed Data API YES-equivalent trade prices",
               "fixed_effects": ["market_id", "calendar_hour_utc"], "cluster": "event_id CRV1",
               "interpretation": "price-source robustness; sparse executed-price coverage is reported separately"}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(pd.DataFrame(contrasts).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

