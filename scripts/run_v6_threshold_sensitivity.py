#!/usr/bin/env python3
"""Estimate threshold and flow-transformation sensitivities for H2."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from run_v5_past_only_p99_h2 import fit, contrast

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "regression_inputs/v6/threshold_sensitivity/past_only_threshold_sensitivity.csv.gz"
OUT = ROOT / "regression_results/v6/threshold_sensitivity"
PERCENTILES = ("p95", "p975", "p99", "p995")
SPECS = ("expanding", "rolling7")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    base = ["market_id", "event_id", "calendar_hour_utc", "v5_delta_5m", "v5_baseline_price",
            "v5_lagged_30m_price_change", "log_lagged_30m_volume", "log_minutes_to_kickoff",
            "v5_strict0_5m"]
    for spec in SPECS:
        for p in PERCENTILES:
            base += [f"{spec}_{p}_eligible", f"{spec}_{p}_large_flow", f"{spec}_{p}_ordinary_flow"]
    data = pd.read_csv(SOURCE, usecols=base)
    coefficients = []; contrasts = []; diagnostics = []; effects = []; support = []
    controls = ["v5_lagged_30m_price_change", "v5_baseline_price", "log_lagged_30m_volume", "log_minutes_to_kickoff"]
    for spec in SPECS:
        for p in PERCENTILES:
            mask = data.v5_strict0_5m.eq(1) & data[f"{spec}_{p}_eligible"].eq(1)
            d0 = data.loc[mask].dropna(subset=controls + ["v5_delta_5m"]).copy()
            large_raw = d0[f"{spec}_{p}_large_flow"].astype(float)
            ordinary_raw = d0[f"{spec}_{p}_ordinary_flow"].astype(float)
            support.append({"specification": spec, "percentile": p, "rows": len(d0),
                            "markets": d0.market_id.nunique(), "events": d0.event_id.nunique(),
                            "large_nonzero_share": large_raw.ne(0).mean(),
                            "ordinary_nonzero_share": ordinary_raw.ne(0).mean(),
                            "large_flow_sd_usdc": large_raw.std(), "ordinary_flow_sd_usdc": ordinary_raw.std()})
            transformations = {
                "signed_log": (np.sign(large_raw) * np.log1p(large_raw.abs()),
                               np.sign(ordinary_raw) * np.log1p(ordinary_raw.abs()), lambda x: np.log1p(x)),
                "asinh": (np.arcsinh(large_raw), np.arcsinh(ordinary_raw), np.arcsinh),
            }
            # Raw-flow sensitivity is scaled per USD 1,000 and winsorised at
            # common sample tails to avoid a handful of extreme minutes
            # dominating the linear specification.
            combined = pd.concat([large_raw, ordinary_raw], ignore_index=True)
            lo, hi = combined.quantile([.005, .995])
            transformations["raw_scaled_winsor"] = (large_raw.clip(lo, hi) / 1000,
                                                      ordinary_raw.clip(lo, hi) / 1000,
                                                      lambda x: x / 1000)
            for transform, (large, ordinary, scale_value) in transformations.items():
                d = d0.copy(); lterm = "focal_large"; oterm = "focal_ordinary"
                d[lterm] = np.asarray(large); d[oterm] = np.asarray(ordinary)
                model = f"{spec.upper()}_{p.upper()}_{transform.upper()}"
                print(f"Estimating {model}: {len(d):,}", flush=True)
                co, diag, beta, covariance, names, clusters = fit(d, model, "v5_delta_5m", [lterm, oterm] + controls)
                coefficients += co; diagnostics.append(diag)
                con = contrast(model, "v5_delta_5m", "large_minus_ordinary", {lterm: 1, oterm: -1}, beta, covariance, names, clusters)
                con.update({"specification": spec, "percentile": p, "transformation": transform})
                contrasts.append(con)
                for term, values in ((lterm, d[lterm]), (oterm, d[oterm])):
                    estimate = float(beta[names.index(term)])
                    effects.append({"model": model, "specification": spec, "percentile": p,
                                    "transformation": transform, "term": term,
                                    "predictor_sd": values.std(), "one_sd_price_change": estimate * values.std(),
                                    "effect_for_positive_1000_usdc": estimate * float(scale_value(1000))})
    pd.DataFrame(coefficients).to_csv(OUT / "coefficients.csv", index=False)
    pd.DataFrame(contrasts).to_csv(OUT / "contrasts.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(OUT / "model_diagnostics.csv", index=False)
    pd.DataFrame(effects).to_csv(OUT / "standardised_and_common_value_effects.csv", index=False)
    pd.DataFrame(support).to_csv(OUT / "sample_support.csv", index=False)
    summary = {"generated_at": datetime.now(timezone.utc).isoformat(), "status": "complete_qc_passed",
               "models": len(diagnostics), "percentiles": list(PERCENTILES), "specifications": list(SPECS),
               "transformations": ["signed_log", "asinh", "raw_scaled_winsor"],
               "fixed_effects": ["market_id", "calendar_hour_utc"], "cluster": "event_id CRV1"}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

