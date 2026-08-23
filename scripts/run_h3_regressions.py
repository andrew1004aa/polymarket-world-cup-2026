#!/usr/bin/env python3
"""Estimate H3 subsequent-movement and Brier-score models."""

from __future__ import annotations

import argparse
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
DEFAULT_OUT = ROOT / "robustness_results" / "v1"
CONTROLS = "yes_price_t + lagged_30m_price_change + log_lagged_30m_volume + log_minutes_to_kickoff"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024): digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False); temporary.replace(path)


def coef_frame(model: str, fit, clusters: int) -> pd.DataFrame:
    names = [str(value) for value in fit._coefnames]
    estimates = np.asarray(fit.coef(), dtype=float)
    covariance = np.asarray(fit._vcov, dtype=float)
    se = np.sqrt(np.diag(covariance)); statistic = estimates / se
    df = clusters - 1; critical = stats.t.ppf(0.975, df=df)
    return pd.DataFrame({
        "model": model, "term": names, "estimate": estimates,
        "std_error": se, "t_statistic": statistic, "df": df,
        "p_value": 2*stats.t.sf(np.abs(statistic), df=df),
        "conf_low": estimates-critical*se, "conf_high": estimates+critical*se,
    })


def equality(model: str, frame: pd.DataFrame, fit, left: str, right: str, clusters: int) -> dict:
    names = [str(value) for value in fit._coefnames]
    pos = {name: index for index, name in enumerate(names)}
    covariance = np.asarray(fit._vcov, dtype=float)
    il, ir = pos[left], pos[right]
    beta = frame.set_index("term")["estimate"]
    difference = float(beta[left]-beta[right])
    variance = float(covariance[il,il]+covariance[ir,ir]-2*covariance[il,ir])
    if variance <= 0 or not math.isfinite(variance): raise RuntimeError(f"Invalid variance: {model}")
    se = math.sqrt(variance); statistic = difference/se; df = clusters-1
    critical = stats.t.ppf(0.975, df=df)
    return {"model":model,"left_term":left,"right_term":right,
            "coefficient_difference":difference,"std_error":se,
            "t_statistic":statistic,"df":df,
            "p_value":float(2*stats.t.sf(abs(statistic),df=df)),
            "conf_low":difference-critical*se,"conf_high":difference+critical*se}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--force", action="store_true")
    parser.add_argument("--version", default="v1"); parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(); out = ROOT / args.output_dir if args.output_dir else DEFAULT_OUT
    config_path = out / "h3_config.json"; config = json.loads(config_path.read_text(encoding="utf-8"))
    log_path = out / "h3_run_log.json"
    if log_path.exists() and not args.force:
        prior = json.loads(log_path.read_text(encoding="utf-8"))
        if prior.get("status") == "complete_qc_passed":
            print("H3 regressions already completed; use --force to rerun."); return 0
    source = ROOT / config["input"]
    if sha256(source) != config["input_sha256"]: raise SystemExit("H3 input checksum mismatch")
    log = {"version":args.version,"status":"running","started_at":now(),
           "software":{"python":platform.python_version(),"pandas":pd.__version__,
                       "numpy":np.__version__,"scipy":scipy.__version__,"pyfixest":pf.__version__},
           "completed_models":[],"errors":[]}
    atomic_json(log_path,log)
    default_columns = ["market_id","event_id","calendar_hour_utc","delta_yes_price_5m",
               "delta_5_to_15","signed_log_total_net_flow","signed_log_whale_net_flow",
               "signed_log_nonwhale_net_flow","log_abs_whale_net_flow",
               "log_abs_nonwhale_net_flow","yes_price_t","lagged_30m_price_change",
               "log_lagged_30m_volume","log_minutes_to_kickoff","brier_improvement_0_5",
               "brier_improvement_0_15","brier_improvement_5_15"]
    columns = config.get("input_columns", default_columns)
    default_formulas = {
        "R_TOTAL": f"delta_5_to_15 ~ signed_log_total_net_flow + delta_yes_price_5m + {CONTROLS} | market_id + calendar_hour_utc",
        "R_SPLIT": f"delta_5_to_15 ~ signed_log_whale_net_flow + signed_log_nonwhale_net_flow + delta_yes_price_5m + {CONTROLS} | market_id + calendar_hour_utc",
        "B5_SPLIT": f"brier_improvement_0_5 ~ log_abs_whale_net_flow + log_abs_nonwhale_net_flow + {CONTROLS} | market_id + calendar_hour_utc",
        "B15_SPLIT": f"brier_improvement_0_15 ~ log_abs_whale_net_flow + log_abs_nonwhale_net_flow + {CONTROLS} | market_id + calendar_hour_utc",
        "B5_15_SPLIT": f"brier_improvement_5_15 ~ log_abs_whale_net_flow + log_abs_nonwhale_net_flow + brier_improvement_0_5 + {CONTROLS} | market_id + calendar_hour_utc",
    }
    formulas = config.get("formulas", default_formulas)
    try:
        print("Loading H3 sample",flush=True); data=pd.read_csv(source,usecols=columns)
        if len(data)!=int(config["expected_rows"]) or data.isna().any().any():
            raise RuntimeError("H3 input row or missing-value validation failed")
        for name in ["market_id","event_id","calendar_hour_utc"]: data[name]=data[name].astype("category")
        clusters=int(data["event_id"].nunique()); frames=[]; diagnostics=[]; tests=[]
        descriptive={
            "delta_5_to_15_zero":int((data["delta_5_to_15"]==0).sum()),
            "delta_5_to_15_positive":int((data["delta_5_to_15"]>0).sum()),
            "delta_5_to_15_negative":int((data["delta_5_to_15"]<0).sum()),
            "brier_5_to_15_positive":int((data["brier_improvement_5_15"]>0).sum()),
            "brier_5_to_15_zero":int((data["brier_improvement_5_15"]==0).sum()),
            "brier_5_to_15_negative":int((data["brier_improvement_5_15"]<0).sum()),
        }
        for model,formula in formulas.items():
            print(f"Estimating {model}",flush=True)
            fit=pf.feols(formula,data=data,vcov={"CRV1":"event_id"},fixef_rm="singleton",
                         copy_data=False,store_data=False,lean=True)
            frame=coef_frame(model,fit,clusters);frames.append(frame)
            diagnostics.append({"model":model,"source_rows":len(data),"fitted_rows":int(fit._N),
                                "markets":int(data["market_id"].nunique()),"events":clusters,
                                "calendar_hours":int(data["calendar_hour_utc"].nunique()),
                                "r_squared":float(fit._r2),"within_r_squared":float(fit._r2_within),
                                "collinear_variables_removed":";".join(map(str,fit._collin_vars)),"formula":formula})
            terms = config.get("equality_tests", {}).get(model)
            if terms:
                tests.append(equality(model,frame,fit,terms[0],terms[1],clusters))
            elif model=="R_SPLIT":
                tests.append(equality(model,frame,fit,"signed_log_whale_net_flow","signed_log_nonwhale_net_flow",clusters))
            elif model.startswith("B"):
                tests.append(equality(model,frame,fit,"log_abs_whale_net_flow","log_abs_nonwhale_net_flow",clusters))
            log["completed_models"].append(model);atomic_json(log_path,log)
        atomic_csv(out/"h3_coefficients.csv",pd.concat(frames,ignore_index=True))
        atomic_csv(out/"h3_model_diagnostics.csv",pd.DataFrame(diagnostics))
        atomic_csv(out/"h3_equality_tests.csv",pd.DataFrame(tests))
        log.update({"status":"complete_qc_passed","completed_at":now(),"descriptive_counts":descriptive,"outputs":{}})
        for name in ["h3_coefficients.csv","h3_model_diagnostics.csv","h3_equality_tests.csv"]:
            path=out/name;log["outputs"][name]={"sha256":sha256(path),"bytes":path.stat().st_size}
        atomic_json(log_path,log);config["status"]="complete_qc_passed";config["completed_at"]=log["completed_at"]
        atomic_json(config_path,config);print(json.dumps(log,indent=2),flush=True);return 0
    except Exception as error:
        log["status"]="failed";log["failed_at"]=now();log["errors"].append(repr(error));atomic_json(log_path,log);raise


if __name__ == "__main__": raise SystemExit(main())
