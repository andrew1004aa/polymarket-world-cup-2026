#!/usr/bin/env python3
"""Estimate H3a concentration and absolute five-minute price-change models."""
import gc
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf
import scipy

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "model_samples/v4/h3a_market_minute/h3a_match_pre_5m.csv.gz"
OUT = ROOT / "regression_results/v4/h3a/match_pre_5m"
COLS = [
    "market_id", "event_id", "calendar_hour_utc", "absolute_price_change_5m",
    "wallet_volume_hhi", "wallet_top1_volume_share", "wallet_top3_volume_share",
    "yes_price_t", "lagged_30m_price_change", "log_lagged_30m_volume",
    "log_minutes_to_kickoff", "log_gross_volume_usdc", "log_trade_count",
    "log_distinct_wallet_count", "absolute_flow_imbalance",
]


def tidy(fit, model):
    frame = fit.tidy().reset_index().rename(columns={
        "Coefficient": "term", "Estimate": "estimate", "Std. Error": "std_error",
        "t value": "t_statistic", "Pr(>|t|)": "p_value",
        "2.5%": "conf_low", "97.5%": "conf_high",
    })
    frame.insert(0, "model", model)
    return frame


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(SRC, usecols=COLS)
    if data.isna().any().any() or (~np.isfinite(data.select_dtypes("number"))).any().any():
        raise RuntimeError("H3a input contains missing or non-finite model values")
    data["absolute_lagged_30m_price_change"] = data.lagged_30m_price_change.abs()
    for column in ["market_id", "event_id", "calendar_hour_utc"]:
        data[column] = data[column].astype("category")
    frozen = "yes_price_t + absolute_lagged_30m_price_change + log_lagged_30m_volume + log_minutes_to_kickoff"
    activity = frozen + " + log_gross_volume_usdc + log_trade_count + absolute_flow_imbalance"
    formulas = {
        "H3A_HHI_MINIMAL": f"absolute_price_change_5m ~ wallet_volume_hhi + {frozen} | market_id + calendar_hour_utc",
        "H3A_HHI_PRIMARY": f"absolute_price_change_5m ~ wallet_volume_hhi + {activity} | market_id + calendar_hour_utc",
        "H3A_TOP1_ROBUST": f"absolute_price_change_5m ~ wallet_top1_volume_share + {activity} | market_id + calendar_hour_utc",
        "H3A_TOP3_ROBUST": f"absolute_price_change_5m ~ wallet_top3_volume_share + {activity} | market_id + calendar_hour_utc",
        "H3A_HHI_PARTICIPATION": f"absolute_price_change_5m ~ wallet_volume_hhi + {activity} + log_distinct_wallet_count | market_id + calendar_hour_utc",
    }
    coefficient_frames, diagnostics = [], []
    for name, formula in formulas.items():
        print(f"Estimating {name} on {len(data):,} rows", flush=True)
        fit = pf.feols(formula, data=data, vcov={"CRV1": "event_id"},
                       copy_data=False, store_data=False, lean=True)
        coefficient_frames.append(tidy(fit, name))
        diagnostics.append({
            "model": name, "observations": int(fit._N),
            "markets": int(data.market_id.nunique()), "events": int(data.event_id.nunique()),
            "calendar_hours": int(data.calendar_hour_utc.nunique()),
            "r_squared": float(fit._r2), "within_r_squared": float(fit._r2_within),
            "formula": formula, "collinear_variables_removed": ";".join(map(str, fit._collin_vars)),
        })
        del fit
        gc.collect()
    coefficients = pd.concat(coefficient_frames, ignore_index=True)
    coefficients.to_csv(OUT / "coefficients.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(OUT / "model_diagnostics.csv", index=False)
    concentration_terms = coefficients[coefficients.term.isin([
        "wallet_volume_hhi", "wallet_top1_volume_share", "wallet_top3_volume_share"
    ])].copy()
    concentration_terms["effect_per_0_1_increase"] = concentration_terms.estimate * 0.1
    concentration_terms.to_csv(OUT / "concentration_coefficients.csv", index=False)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_qc_passed", "rows": len(data),
        "markets": int(data.market_id.nunique()), "events": int(data.event_id.nunique()),
        "outcome": "absolute five-minute YES-price change",
        "primary_model": "H3A_HHI_PRIMARY", "fixed_effects": ["market_id", "calendar_hour_utc"],
        "cluster": "event_id CRV1", "interpretation": "Conditional associations, not causal effects.",
        "software": {"python": platform.python_version(), "pandas": pd.__version__,
                     "numpy": np.__version__, "scipy": scipy.__version__, "pyfixest": pf.__version__},
        "concentration_results": concentration_terms.to_dict(orient="records"),
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(concentration_terms[["model", "term", "estimate", "std_error", "p_value",
                               "conf_low", "conf_high", "effect_per_0_1_increase"]].to_string(index=False))


if __name__ == "__main__":
    main()
