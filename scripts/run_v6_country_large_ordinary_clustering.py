#!/usr/bin/env python3
"""Alternative clustering for the secondary country large/ordinary model."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "regression_inputs/v3/country/outright_5m_complete_case.csv.gz"
OUT = ROOT / "regression_results/v6/country_clustering"
FORMULA = ("delta_yes_price ~ p99_signed_log_large_net_flow + "
           "p99_signed_log_ordinary_net_flow + yes_price_t + "
           "lagged_30m_price_change + log_lagged_30m_volume | market_id + calendar_hour_utc")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["market_id", "date_utc", "calendar_hour_utc", "delta_yes_price",
            "p99_signed_log_large_net_flow", "p99_signed_log_ordinary_net_flow",
            "yes_price_t", "lagged_30m_price_change", "log_lagged_30m_volume"]
    data = pd.read_csv(SOURCE, usecols=cols)
    for column in ("market_id", "date_utc", "calendar_hour_utc"):
        data[column] = data[column].astype("category")
    coefficients = []; contrasts = []; diagnostics = []
    specifications = [("market", "market_id", data.market_id.nunique() - 1),
                      ("date", "date_utc", data.date_utc.nunique() - 1),
                      ("market_date_two_way", "market_id + date_utc",
                       min(data.market_id.nunique(), data.date_utc.nunique()) - 1)]
    for label, cluster, df in specifications:
        print(f"Estimating country model with {label} clustering", flush=True)
        result = pf.feols(FORMULA, data=data, vcov={"CRV1": cluster}, copy_data=False, store_data=False, lean=True)
        tidy = result.tidy().reset_index().rename(columns={"Coefficient": "term", "Estimate": "estimate",
            "Std. Error": "std_error", "t value": "t_value", "Pr(>|t|)": "p_value",
            "2.5%": "conf_low", "97.5%": "conf_high"})
        tidy.insert(0, "covariance", label); coefficients.append(tidy)
        names = {name: i for i, name in enumerate(result._coefnames)}
        a = names["p99_signed_log_large_net_flow"]; b = names["p99_signed_log_ordinary_net_flow"]
        cov = np.asarray(result._vcov, float); variance = cov[a, a] + cov[b, b] - 2 * cov[a, b]
        beta = tidy.set_index("term").estimate
        difference = float(beta["p99_signed_log_large_net_flow"] -
                           beta["p99_signed_log_ordinary_net_flow"])
        se = math.sqrt(float(variance)); t = difference / se; critical = stats.t.ppf(.975, df)
        contrasts.append({"covariance": label, "cluster": cluster, "large_minus_ordinary": difference,
                          "std_error": se, "t_value": t, "df": df,
                          "p_value": 2 * stats.t.sf(abs(t), df),
                          "conf_low": difference - critical * se, "conf_high": difference + critical * se})
        diagnostics.append({"covariance": label, "rows": int(result._N), "markets": data.market_id.nunique(),
                            "dates": data.date_utc.nunique(), "calendar_hours": data.calendar_hour_utc.nunique(),
                            "within_r_squared": float(result._r2_within)})
    pd.concat(coefficients, ignore_index=True).to_csv(OUT / "coefficients.csv", index=False)
    pd.DataFrame(contrasts).to_csv(OUT / "contrasts.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(OUT / "model_diagnostics.csv", index=False)
    summary = {"generated_at": datetime.now(timezone.utc).isoformat(), "status": "complete_qc_passed",
               "scope": "secondary 48-country outright extension", "formula": FORMULA,
               "covariances": [x[0] for x in specifications],
               "interpretation": "secondary extension, not independent replication"}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(pd.DataFrame(contrasts).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
