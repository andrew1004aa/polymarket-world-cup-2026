#!/usr/bin/env python3
"""Estimate matched-sample 1-, 5- and 15-minute robustness models."""

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
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def results(model: str, horizon: int, fit, clusters: int) -> pd.DataFrame:
    names = [str(value) for value in fit._coefnames]
    estimates = np.asarray(fit.coef(), dtype=float)
    covariance = np.asarray(fit._vcov, dtype=float)
    standard_errors = np.sqrt(np.diag(covariance))
    statistics = estimates / standard_errors
    df = clusters - 1
    critical = stats.t.ppf(0.975, df=df)
    return pd.DataFrame({
        "model": model, "horizon_minutes": horizon, "term": names,
        "estimate": estimates, "std_error": standard_errors,
        "t_statistic": statistics, "df": df,
        "p_value": 2 * stats.t.sf(np.abs(statistics), df=df),
        "conf_low": estimates - critical * standard_errors,
        "conf_high": estimates + critical * standard_errors,
    })


def equality(model: str, horizon: int, frame: pd.DataFrame, fit, clusters: int,
             left: str, right: str) -> dict:
    names = [str(value) for value in fit._coefnames]
    positions = {name: index for index, name in enumerate(names)}
    covariance = np.asarray(fit._vcov, dtype=float)
    il, ir = positions[left], positions[right]
    betas = frame.set_index("term")["estimate"]
    difference = float(betas[left] - betas[right])
    variance = float(covariance[il, il] + covariance[ir, ir] - 2 * covariance[il, ir])
    if variance <= 0 or not math.isfinite(variance):
        raise RuntimeError(f"Invalid equality variance for {model}")
    standard_error = math.sqrt(variance)
    statistic = difference / standard_error
    df = clusters - 1
    critical = stats.t.ppf(0.975, df=df)
    return {
        "model": model, "horizon_minutes": horizon,
        "coefficient_difference": difference, "std_error": standard_error,
        "t_statistic": statistic, "df": df,
        "p_value": float(2 * stats.t.sf(abs(statistic), df=df)),
        "conf_low": difference - critical * standard_error,
        "conf_high": difference + critical * standard_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v1")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    out = ROOT / args.output_dir if args.output_dir else ROOT / "robustness_results" / args.version
    config_path = out / "horizon_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    log_path = out / "horizon_run_log.json"
    if log_path.exists() and not args.force:
        prior = json.loads(log_path.read_text(encoding="utf-8"))
        if prior.get("status") == "complete_qc_passed":
            print("Horizon robustness already completed; use --force to rerun.")
            return 0

    input_path = ROOT / config["input"]
    if sha256(input_path) != config["input_sha256"]:
        raise SystemExit("Matched horizon input checksum mismatch")
    log = {
        "version": args.version, "status": "running", "started_at": now(),
        "software": {"python": platform.python_version(), "pandas": pd.__version__,
                     "numpy": np.__version__, "scipy": scipy.__version__, "pyfixest": pf.__version__},
        "completed_models": [], "errors": [],
    }
    atomic_json(log_path, log)
    split_left, split_right = config.get("split_flow_fields", [
        "signed_log_whale_net_flow", "signed_log_nonwhale_net_flow"])
    columns = [
        "market_id", "event_id", "calendar_hour_utc",
        "signed_log_total_net_flow", split_left, split_right, "yes_price_t",
        "lagged_30m_price_change", "log_lagged_30m_volume",
        "log_minutes_to_kickoff", "delta_yes_price_1m",
        "delta_yes_price_5m", "delta_yes_price_15m",
    ]
    try:
        print("Loading matched 1/5/15-minute sample", flush=True)
        data = pd.read_csv(input_path, usecols=columns)
        if len(data) != int(config["expected_rows"]):
            raise RuntimeError("Matched horizon row count mismatch")
        if data.isna().any().any():
            raise RuntimeError("Unexpected missing value in matched horizon input")
        for name in ["market_id", "event_id", "calendar_hour_utc"]:
            data[name] = data[name].astype("category")
        clusters = int(data["event_id"].nunique())
        diagnostics = []; frames = []; tests = []
        zero_counts = {}
        for horizon in config["horizons_minutes"]:
            outcome = f"delta_yes_price_{horizon}m"
            zero_counts[str(horizon)] = int((data[outcome] == 0).sum())
            formulas = {
                f"H{horizon}_TOTAL": f"{outcome} ~ signed_log_total_net_flow + {CONTROLS} | market_id + calendar_hour_utc",
                f"H{horizon}_SPLIT": f"{outcome} ~ {split_left} + {split_right} + {CONTROLS} | market_id + calendar_hour_utc",
            }
            for model, formula in formulas.items():
                print(f"Estimating {model}", flush=True)
                fit = pf.feols(formula, data=data, vcov={"CRV1": "event_id"},
                               fixef_rm="singleton", copy_data=False,
                               store_data=False, lean=True)
                frame = results(model, horizon, fit, clusters)
                frames.append(frame)
                diagnostics.append({
                    "model": model, "horizon_minutes": horizon,
                    "source_rows": len(data), "fitted_rows": int(fit._N),
                    "zero_outcome_rows": zero_counts[str(horizon)],
                    "zero_outcome_share": zero_counts[str(horizon)] / len(data),
                    "markets": int(data["market_id"].nunique()),
                    "events": clusters,
                    "calendar_hours": int(data["calendar_hour_utc"].nunique()),
                    "r_squared": float(fit._r2), "within_r_squared": float(fit._r2_within),
                    "collinear_variables_removed": ";".join(map(str, fit._collin_vars)),
                    "formula": formula,
                })
                if model.endswith("_SPLIT"):
                    tests.append(equality(model, horizon, frame, fit, clusters, split_left, split_right))
                log["completed_models"].append(model)
                atomic_json(log_path, log)
        coefficients = pd.concat(frames, ignore_index=True)
        atomic_csv(out / "horizon_coefficients.csv", coefficients)
        atomic_csv(out / "horizon_model_diagnostics.csv", pd.DataFrame(diagnostics))
        atomic_csv(out / "horizon_equality_tests.csv", pd.DataFrame(tests))
        log.update({"status": "complete_qc_passed", "completed_at": now(),
                    "zero_outcome_rows": zero_counts, "outputs": {}})
        for name in ["horizon_coefficients.csv", "horizon_model_diagnostics.csv", "horizon_equality_tests.csv"]:
            path = out / name
            log["outputs"][name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        atomic_json(log_path, log)
        config["status"] = "complete_qc_passed"; config["completed_at"] = log["completed_at"]
        atomic_json(config_path, config)
        print(json.dumps(log, indent=2), flush=True)
        return 0
    except Exception as error:
        log["status"] = "failed"; log["failed_at"] = now(); log["errors"].append(repr(error))
        atomic_json(log_path, log)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
