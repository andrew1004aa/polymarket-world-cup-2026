#!/usr/bin/env python3
"""Estimate the frozen v3 prematch/post-kickoff five-minute interaction models."""

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
PRE = ROOT / "regression_inputs/v3/primary_prematch_5m_complete_case.csv.gz"
POST = ROOT / "regression_inputs/v3/post_kickoff/post_kickoff_pre_resolution_5m_complete_case.csv.gz"
OUT = ROOT / "robustness_results/v3/phase_interaction"
COLS = ["model_sample_record_id", "market_id", "event_id", "calendar_hour_utc",
        "delta_yes_price", "signed_log_total_net_flow",
        "p99_signed_log_large_net_flow", "p99_signed_log_ordinary_net_flow",
        "yes_price_t", "lagged_30m_price_change", "log_lagged_30m_volume"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024): digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, obj: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8"); tmp.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False); tmp.replace(path)


def tidy(model: str, fit) -> pd.DataFrame:
    frame = fit.tidy().reset_index().rename(columns={
        "Coefficient": "term", "Estimate": "estimate", "Std. Error": "std_error",
        "t value": "t_statistic", "Pr(>|t|)": "p_value",
        "2.5%": "conf_low", "97.5%": "conf_high"})
    frame.insert(0, "model", model)
    return frame


def contrast(model: str, label: str, fit, weights: dict[str, float], df: int) -> dict:
    names = [str(x) for x in fit._coefnames]
    pos = {name: i for i, name in enumerate(names)}
    missing = sorted(set(weights) - set(pos))
    if missing: raise RuntimeError(f"Unavailable contrast terms: {missing}")
    beta = np.asarray(fit.coef(), dtype=float)
    vector = np.zeros(len(names))
    for name, weight in weights.items(): vector[pos[name]] = weight
    estimate = float(vector @ beta)
    variance = float(vector @ np.asarray(fit._vcov, dtype=float) @ vector)
    if not math.isfinite(variance) or variance <= 0: raise RuntimeError(f"Invalid variance: {label}")
    se = math.sqrt(variance); statistic = estimate / se
    critical = float(stats.t.ppf(.975, df))
    return {"model": model, "contrast": label,
            "linear_combination": " + ".join(f"{w:g}*{n}" for n, w in weights.items()),
            "estimate": estimate, "std_error": se, "t_statistic": statistic, "df": df,
            "p_value": float(2 * stats.t.sf(abs(statistic), df)),
            "conf_low": estimate - critical * se, "conf_high": estimate + critical * se}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    config = {
        "version": "v3_phase_interaction", "status": "frozen_pending_estimation",
        "prematch_input": str(PRE.relative_to(ROOT)), "prematch_sha256": sha256(PRE),
        "postkickoff_input": str(POST.relative_to(ROOT)), "postkickoff_sha256": sha256(POST),
        "horizon_minutes": 5, "prematch_expected_rows": 748481,
        "postkickoff_expected_rows": 45697,
        "fixed_effects": ["market_id", "calendar_hour_utc"],
        "cluster": "event_id", "vcov": "CRV1",
        "common_controls": ["yes_price_t", "lagged_30m_price_change", "log_lagged_30m_volume"],
        "notes": ["Prematch and post-kickoff samples use the same five-minute horizon and controls.",
                  "Post-kickoff/pre-resolution is not labelled purely in-play."]}
    atomic_json(OUT / "config.json", config)
    log = {"status": "running", "started_at": now(), "completed_models": [], "errors": []}
    atomic_json(OUT / "run_log.json", log)
    try:
        pre = pd.read_csv(PRE, usecols=COLS); post = pd.read_csv(POST, usecols=COLS)
        if len(pre) != 748481 or len(post) != 45697: raise RuntimeError("Unexpected phase row count")
        pre["post_kickoff"] = 0.0; post["post_kickoff"] = 1.0
        data = pd.concat([pre, post], ignore_index=True)
        if data.isna().any().any(): raise RuntimeError("Missing model values")
        if data.model_sample_record_id.duplicated().any(): raise RuntimeError("Duplicate record IDs")
        for flow in ["signed_log_total_net_flow", "p99_signed_log_large_net_flow", "p99_signed_log_ordinary_net_flow"]:
            data[f"post_x_{flow}"] = data["post_kickoff"] * data[flow]
        for col in ["market_id", "event_id", "calendar_hour_utc"]: data[col] = data[col].astype("category")
        controls = "yes_price_t + lagged_30m_price_change + log_lagged_30m_volume"
        formulas = {
            "PHASE_TOTAL": "delta_yes_price ~ signed_log_total_net_flow + post_x_signed_log_total_net_flow + " + controls + " | market_id + calendar_hour_utc",
            "PHASE_P99_SPLIT": "delta_yes_price ~ p99_signed_log_large_net_flow + p99_signed_log_ordinary_net_flow + post_x_p99_signed_log_large_net_flow + post_x_p99_signed_log_ordinary_net_flow + " + controls + " | market_id + calendar_hour_utc"}
        frames=[]; diagnostics=[]; fits={}
        for name, formula in formulas.items():
            print(f"Estimating {name} on {len(data):,} rows", flush=True)
            fit=pf.feols(formula,data=data,vcov={"CRV1":"event_id"},copy_data=False,store_data=False,lean=True)
            fits[name]=fit; frames.append(tidy(name,fit))
            diagnostics.append({"model":name,"observations":int(fit._N),"markets":int(data.market_id.nunique()),
                                "events":int(data.event_id.nunique()),"calendar_hours":int(data.calendar_hour_utc.nunique()),
                                "r_squared":float(fit._r2),"within_r_squared":float(fit._r2_within),
                                "collinear_variables_removed":";".join(map(str,fit._collin_vars)),"formula":formula})
            log["completed_models"].append(name); atomic_json(OUT / "run_log.json",log)
        df=103; split=fits["PHASE_P99_SPLIT"]
        tests=[
            contrast("PHASE_TOTAL","Post-minus-prematch total-flow coefficient",fits["PHASE_TOTAL"],{"post_x_signed_log_total_net_flow":1},df),
            contrast("PHASE_P99_SPLIT","Prematch large-minus-ordinary",split,{"p99_signed_log_large_net_flow":1,"p99_signed_log_ordinary_net_flow":-1},df),
            contrast("PHASE_P99_SPLIT","Post-kickoff large-minus-ordinary",split,{"p99_signed_log_large_net_flow":1,"post_x_p99_signed_log_large_net_flow":1,"p99_signed_log_ordinary_net_flow":-1,"post_x_p99_signed_log_ordinary_net_flow":-1},df),
            contrast("PHASE_P99_SPLIT","Post-minus-prematch large coefficient",split,{"post_x_p99_signed_log_large_net_flow":1},df),
            contrast("PHASE_P99_SPLIT","Post-minus-prematch ordinary coefficient",split,{"post_x_p99_signed_log_ordinary_net_flow":1},df),
            contrast("PHASE_P99_SPLIT","Difference-in-differences: change in large-minus-ordinary gap",split,{"post_x_p99_signed_log_large_net_flow":1,"post_x_p99_signed_log_ordinary_net_flow":-1},df)]
        atomic_csv(OUT / "coefficients.csv",pd.concat(frames,ignore_index=True))
        atomic_csv(OUT / "phase_contrasts.csv",pd.DataFrame(tests))
        atomic_csv(OUT / "model_diagnostics.csv",pd.DataFrame(diagnostics))
        config.update({"status":"complete_qc_passed","completed_at":now(),"combined_rows":len(data)})
        atomic_json(OUT / "config.json",config)
        log.update({"status":"complete_qc_passed","completed_at":config["completed_at"],"errors":[]})
        atomic_json(OUT / "run_log.json",log)
        print(json.dumps(log,indent=2))
        return 0
    except Exception as exc:
        log.update({"status":"failed","failed_at":now(),"errors":[repr(exc)]})
        atomic_json(OUT / "run_log.json",log); raise


if __name__ == "__main__": raise SystemExit(main())
