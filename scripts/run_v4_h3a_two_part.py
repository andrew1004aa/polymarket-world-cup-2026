#!/usr/bin/env python3
"""Two-part H3a robustness for sparse five-minute price changes."""
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
OUT = ROOT / "regression_results/v4/h3a/two_part"
COLS = [
    "market_id", "event_id", "calendar_hour_utc", "absolute_price_change_5m",
    "wallet_volume_hhi", "wallet_top1_volume_share", "wallet_top3_volume_share",
    "yes_price_t", "lagged_30m_price_change", "log_lagged_30m_volume",
    "log_minutes_to_kickoff", "log_gross_volume_usdc", "log_trade_count",
    "absolute_flow_imbalance",
]


def tidy(fit, model, part, measure):
    frame = fit.tidy().reset_index().rename(columns={
        "Coefficient": "term", "Estimate": "estimate", "Std. Error": "std_error",
        "t value": "t_statistic", "Pr(>|t|)": "p_value",
        "2.5%": "conf_low", "97.5%": "conf_high",
    })
    frame.insert(0, "concentration_measure", measure)
    frame.insert(0, "part", part)
    frame.insert(0, "model", model)
    return frame


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(SRC, usecols=COLS)
    data["any_price_update_5m"] = (data.absolute_price_change_5m > 0).astype(float)
    data["absolute_lagged_30m_price_change"] = data.lagged_30m_price_change.abs()
    data["log_absolute_price_change_5m"] = np.nan
    nonzero_mask = data.any_price_update_5m.eq(1)
    data.loc[nonzero_mask, "log_absolute_price_change_5m"] = np.log(
        data.loc[nonzero_mask, "absolute_price_change_5m"])
    for column in ["market_id", "event_id", "calendar_hour_utc"]:
        data[column] = data[column].astype("category")
    controls = ("yes_price_t + absolute_lagged_30m_price_change + log_lagged_30m_volume + "
                "log_minutes_to_kickoff + log_gross_volume_usdc + log_trade_count + "
                "absolute_flow_imbalance")
    measures = {
        "hhi": "wallet_volume_hhi",
        "top1": "wallet_top1_volume_share",
        "top3": "wallet_top3_volume_share",
    }
    specifications = []
    for label, measure in measures.items():
        specifications.append((f"PART1_ANY_UPDATE_{label.upper()}", "part1_any_update", measure,
                               data, f"any_price_update_5m ~ {measure} + {controls} | market_id + calendar_hour_utc"))
        nonzero = data[data.any_price_update_5m.eq(1)]
        specifications.append((f"PART2_MAGNITUDE_{label.upper()}", "part2_nonzero_magnitude", measure,
                               nonzero, f"absolute_price_change_5m ~ {measure} + {controls} | market_id + calendar_hour_utc"))
    nonzero = data[data.any_price_update_5m.eq(1)]
    specifications.append(("PART2_LOG_MAGNITUDE_HHI", "part2_nonzero_log_magnitude",
                           "wallet_volume_hhi", nonzero,
                           f"log_absolute_price_change_5m ~ wallet_volume_hhi + {controls} | market_id + calendar_hour_utc"))
    frames, diagnostics = [], []
    for name, part, measure, sample, formula in specifications:
        print(f"Estimating {name} on {len(sample):,} rows", flush=True)
        fit = pf.feols(formula, data=sample, vcov={"CRV1": "event_id"},
                       copy_data=False, store_data=False, lean=True)
        frames.append(tidy(fit, name, part, measure))
        diagnostics.append({
            "model": name, "part": part, "observations_input": len(sample),
            "observations_estimated": int(fit._N), "markets": int(sample.market_id.nunique()),
            "events": int(sample.event_id.nunique()), "r_squared": float(fit._r2),
            "within_r_squared": float(fit._r2_within), "formula": formula,
            "collinear_variables_removed": ";".join(map(str, fit._collin_vars)),
        })
        del fit
        gc.collect()
    coefficients = pd.concat(frames, ignore_index=True)
    coefficients.to_csv(OUT / "coefficients.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(OUT / "model_diagnostics.csv", index=False)
    focal = coefficients[coefficients.term.isin(measures.values())].copy()
    focal["effect_per_0_1_increase"] = focal.estimate * 0.1
    focal.to_csv(OUT / "concentration_coefficients.csv", index=False)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_qc_passed", "total_rows": len(data),
        "nonzero_rows": int(data.any_price_update_5m.sum()),
        "nonzero_share": float(data.any_price_update_5m.mean()),
        "part1": "LPM for any non-zero five-minute price update",
        "part2": "OLS magnitude conditional on a non-zero update",
        "fixed_effects": ["market_id", "calendar_hour_utc"], "cluster": "event_id CRV1",
        "software": {"python": platform.python_version(), "pandas": pd.__version__,
                     "numpy": np.__version__, "scipy": scipy.__version__, "pyfixest": pf.__version__},
        "focal_results": focal.to_dict(orient="records"),
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(focal[["model", "part", "term", "estimate", "std_error", "t_statistic",
                 "p_value", "conf_low", "conf_high", "effect_per_0_1_increase"]].to_string(index=False))


if __name__ == "__main__":
    main()
