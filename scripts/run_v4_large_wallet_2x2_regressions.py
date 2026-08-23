#!/usr/bin/env python3
"""Estimate the pre-specified V4 Large x high-activity-wallet 2x2 models."""

from __future__ import annotations

import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf
import scipy
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "regression_inputs/v3/primary_prematch_5m_complete_case.csv.gz"
PANEL = ROOT / "model_samples/v4/large_wallet_2x2/large_wallet_2x2_primary_prematch.csv.gz"
OUT = ROOT / "regression_results/v4/large_wallet_2x2"
GROUPS = ("large_high_activity", "large_other", "ordinary_high_activity", "ordinary_other")
EXPECTED_ROWS, EXPECTED_MARKETS, EXPECTED_EVENTS = 748_481, 312, 104


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, obj: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def linear_test(model: str, label: str, weights: dict[str, float], beta: pd.Series,
                vcov: np.ndarray, names: list[str], clusters: int) -> dict:
    index = {name: i for i, name in enumerate(names)}
    missing = sorted(set(weights) - set(index))
    if missing:
        raise RuntimeError(f"Contrast {label} missing terms: {missing}")
    w = np.zeros(len(names))
    for term, value in weights.items():
        w[index[term]] = value
    estimate = float(sum(beta[term] * value for term, value in weights.items()))
    variance = float(w @ vcov @ w)
    if not math.isfinite(variance) or variance <= 0:
        raise RuntimeError(f"Contrast {label} has invalid variance {variance}")
    se = math.sqrt(variance)
    t = estimate / se
    df = clusters - 1
    critical = float(stats.t.ppf(.975, df))
    return {
        "model": model, "contrast": label, "estimate": estimate, "std_error": se,
        "t_statistic": t, "df": df, "p_value": float(2 * stats.t.sf(abs(t), df)),
        "conf_low": estimate - critical * se, "conf_high": estimate + critical * se,
        "weights": json.dumps(weights, sort_keys=True),
    }


def fit(name: str, formula: str, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"Estimating {name}", flush=True)
    result = pf.feols(formula, data=data, vcov={"CRV1": "event_id"},
                      copy_data=False, store_data=False, lean=True)
    tidy = result.tidy().reset_index().rename(columns={
        "Coefficient": "term", "Estimate": "estimate", "Std. Error": "std_error",
        "t value": "t_statistic", "Pr(>|t|)": "p_value",
        "2.5%": "conf_low", "97.5%": "conf_high",
    })
    tidy.insert(0, "model", name)
    names = [str(x) for x in result._coefnames]
    beta = tidy.set_index("term")["estimate"]
    cov = np.asarray(result._vcov, dtype=float)
    prefix = "log_gross_" if name == "PRICE_UPDATE_LPM" else "signed_flow_"
    lh, lo, oh, oo = (prefix + g for g in GROUPS)
    specs = {
        "high_activity_effect_among_large": {lh: 1, lo: -1},
        "high_activity_effect_among_ordinary": {oh: 1, oo: -1},
        "large_effect_among_high_activity": {lh: 1, oh: -1},
        "large_effect_among_other": {lo: 1, oo: -1},
        "difference_in_differences": {lh: 1, lo: -1, oh: -1, oo: 1},
        "average_high_activity_effect": {lh: .5, lo: -.5, oh: .5, oo: -.5},
        "average_large_effect": {lh: .5, lo: .5, oh: -.5, oo: -.5},
    }
    contrasts = pd.DataFrame([
        linear_test(name, label, weights, beta, cov, names, EXPECTED_EVENTS)
        for label, weights in specs.items()
    ])
    return tidy, contrasts


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    log = {
        "version": "v4_large_wallet_2x2_20260815", "status": "running",
        "started_at": now(), "base": str(BASE.relative_to(ROOT)),
        "base_sha256": sha256(BASE), "panel": str(PANEL.relative_to(ROOT)),
        "panel_sha256": sha256(PANEL),
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__,
                     "pyfixest": pf.__version__},
    }
    atomic_json(OUT / "run_log.json", log)
    base_cols = ["model_sample_record_id", "market_id", "event_id", "calendar_hour_utc",
                 "delta_yes_price", "yes_price_t", "lagged_30m_price_change",
                 "log_lagged_30m_volume", "log_minutes_to_kickoff"]
    panel_cols = ["model_sample_record_id"]
    for g in GROUPS:
        panel_cols += [f"{g}_gross_volume_usdc", f"{g}_signed_log_net_flow"]
    print("Loading and joining frozen inputs", flush=True)
    base = pd.read_csv(BASE, usecols=base_cols)
    panel = pd.read_csv(PANEL, usecols=panel_cols)
    if len(base) != EXPECTED_ROWS or len(panel) != EXPECTED_ROWS:
        raise RuntimeError(f"Unexpected row counts: base={len(base)}, panel={len(panel)}")
    data = base.merge(panel, on="model_sample_record_id", how="inner", validate="one_to_one")
    if len(data) != EXPECTED_ROWS or data.market_id.nunique() != EXPECTED_MARKETS or data.event_id.nunique() != EXPECTED_EVENTS:
        raise RuntimeError("Joined sample dimensions do not match the frozen design")
    if data.isna().any().any():
        raise RuntimeError(f"Missing values after join: {data.isna().sum()[data.isna().sum()>0].to_dict()}")
    data["any_price_update_5m"] = data.delta_yes_price.ne(0).astype(float)
    data["absolute_lagged_30m_price_change"] = data.lagged_30m_price_change.abs()
    for g in GROUPS:
        data[f"log_gross_{g}"] = np.log1p(data[f"{g}_gross_volume_usdc"])
        data[f"signed_flow_{g}"] = data[f"{g}_signed_log_net_flow"]
    for col in ("market_id", "event_id", "calendar_hour_utc"):
        data[col] = data[col].astype("category")
    common = "yes_price_t + log_lagged_30m_volume + log_minutes_to_kickoff"
    update_terms = " + ".join(f"log_gross_{g}" for g in GROUPS)
    direction_terms = " + ".join(f"signed_flow_{g}" for g in GROUPS)
    formulas = {
        "PRICE_UPDATE_LPM": f"any_price_update_5m ~ {update_terms} + absolute_lagged_30m_price_change + {common} | market_id + calendar_hour_utc",
        "DIRECTIONAL_PRICE_CHANGE": f"delta_yes_price ~ {direction_terms} + lagged_30m_price_change + {common} | market_id + calendar_hour_utc",
    }
    coefficients, contrasts = [], []
    for name, formula in formulas.items():
        c, t = fit(name, formula, data)
        coefficients.append(c); contrasts.append(t)
    coef = pd.concat(coefficients, ignore_index=True)
    tests = pd.concat(contrasts, ignore_index=True)
    coef.to_csv(OUT / "coefficients.csv", index=False)
    tests.to_csv(OUT / "contrasts.csv", index=False)
    config = {
        "version": log["version"], "created_at": now(), "definitions": {
            "large": "trade_value_usdc >= within-market P99",
            "high_activity_wallet": "top 1% by canonical-window cumulative USDC volume; operational proxy",
            "unit": "market-minute", "sample": "frozen primary pre-kickoff 5-minute complete case",
        }, "formulas": formulas, "fixed_effects": ["market_id", "calendar_hour_utc"],
        "cluster": "event_id", "contrasts": tests[["contrast", "weights"]].drop_duplicates().to_dict("records"),
    }
    atomic_json(OUT / "config.json", config)
    log.update({"status": "complete_qc_passed", "completed_at": now(), "rows": len(data),
                "markets": int(data.market_id.nunique()), "events": int(data.event_id.nunique()),
                "price_update_rate": float(data.any_price_update_5m.mean()),
                "outputs": ["coefficients.csv", "contrasts.csv", "config.json"]})
    atomic_json(OUT / "run_log.json", log)
    print(json.dumps({k: log[k] for k in ("status", "rows", "markets", "events", "price_update_rate")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
