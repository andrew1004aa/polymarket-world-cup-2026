#!/usr/bin/env python3
"""Run frozen zero-outcome and non-overlap robustness regressions."""

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
DEFAULT_INPUT = ROOT / "robustness_inputs" / "v1"
DEFAULT_OUTPUT = ROOT / "robustness_results" / "v1"


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


def coefficient_frame(model: str, fit, clusters: int) -> pd.DataFrame:
    names = [str(value) for value in fit._coefnames]
    estimates = np.asarray(fit.coef(), dtype=float)
    covariance = np.asarray(fit._vcov, dtype=float)
    standard_errors = np.sqrt(np.diag(covariance))
    statistics = estimates / standard_errors
    df = clusters - 1
    p_values = 2 * stats.t.sf(np.abs(statistics), df=df)
    critical = stats.t.ppf(0.975, df=df)
    return pd.DataFrame({
        "model": model, "term": names, "estimate": estimates,
        "std_error": standard_errors, "t_statistic": statistics,
        "df": df, "p_value": p_values,
        "conf_low": estimates - critical * standard_errors,
        "conf_high": estimates + critical * standard_errors,
    })


def equality_test(model: str, frame: pd.DataFrame, fit, left: str, right: str, clusters: int) -> dict:
    names = [str(value) for value in fit._coefnames]
    positions = {name: index for index, name in enumerate(names)}
    il, ir = positions[left], positions[right]
    covariance = np.asarray(fit._vcov, dtype=float)
    estimates = frame.set_index("term")["estimate"]
    difference = float(estimates[left] - estimates[right])
    variance = float(covariance[il, il] + covariance[ir, ir] - 2 * covariance[il, ir])
    if variance <= 0 or not math.isfinite(variance):
        raise RuntimeError(f"Invalid equality-test variance for {model}: {variance}")
    standard_error = math.sqrt(variance)
    statistic = difference / standard_error
    df = clusters - 1
    critical = stats.t.ppf(0.975, df=df)
    return {
        "model": model, "left_term": left, "right_term": right,
        "coefficient_difference": difference, "std_error": standard_error,
        "t_statistic": statistic, "df": df,
        "p_value": float(2 * stats.t.sf(abs(statistic), df=df)),
        "conf_low": difference - critical * standard_error,
        "conf_high": difference + critical * standard_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v1")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    input_dir = ROOT / args.input_dir if args.input_dir else DEFAULT_INPUT
    output_dir = ROOT / args.output_dir if args.output_dir else DEFAULT_OUTPUT
    config_path = output_dir / "config.json"
    qc_path = input_dir / "robustness_input_qc.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    input_qc = json.loads(qc_path.read_text(encoding="utf-8"))
    log_path = output_dir / "run_log.json"
    if log_path.exists() and not args.force:
        prior = json.loads(log_path.read_text(encoding="utf-8"))
        if prior.get("status") == "complete_qc_passed":
            print("Robustness regressions already completed; use --force to rerun.")
            return 0

    log = {
        "version": args.version, "status": "running", "started_at": now(),
        "software": {"python": platform.python_version(), "pandas": pd.__version__,
                     "numpy": np.__version__, "scipy": scipy.__version__,
                     "pyfixest": pf.__version__},
        "completed_models": [], "errors": [],
    }
    atomic_json(log_path, log)
    sample_cache = {}
    coefficient_tables = []
    diagnostics = []
    tests = []
    default_columns = [
        "market_id", "event_id", "calendar_hour_utc", "delta_yes_price",
        "any_price_change", "signed_log_total_net_flow",
        "signed_log_whale_net_flow", "signed_log_nonwhale_net_flow",
        "log_abs_total_net_flow", "log_abs_whale_net_flow",
        "log_abs_nonwhale_net_flow", "yes_price_t",
        "lagged_30m_price_change", "log_lagged_30m_volume",
        "log_minutes_to_kickoff",
    ]
    columns = config.get("input_columns", default_columns)
    try:
        for model, specification in config["models"].items():
            sample = specification["sample"]
            if sample not in sample_cache:
                path = input_dir / f"{sample}.csv.gz"
                expected = input_qc["samples"][sample]
                if sha256(path) != expected["sha256"]:
                    raise RuntimeError(f"Checksum mismatch for {sample}")
                print(f"Loading {sample}", flush=True)
                data = pd.read_csv(path, usecols=columns)
                if len(data) != expected["rows"]:
                    raise RuntimeError(f"Row mismatch for {sample}")
                for name in ["market_id", "event_id", "calendar_hour_utc"]:
                    data[name] = data[name].astype("category")
                sample_cache[sample] = data
            data = sample_cache[sample]
            clusters = int(data["event_id"].nunique())
            print(f"Estimating {model} on {len(data):,} source rows", flush=True)
            fit = pf.feols(
                specification["formula"], data=data,
                vcov={"CRV1": "event_id"}, fixef_rm="singleton",
                copy_data=False, store_data=False, lean=True,
            )
            frame = coefficient_frame(model, fit, clusters)
            coefficient_tables.append(frame)
            diagnostics.append({
                "model": model, "sample": sample,
                "source_rows": len(data), "fitted_rows": int(fit._N),
                "singleton_rows_removed": len(data) - int(fit._N),
                "markets_source": int(data["market_id"].nunique()),
                "events_source": clusters,
                "calendar_hours_source": int(data["calendar_hour_utc"].nunique()),
                "r_squared": float(fit._r2),
                "within_r_squared": float(fit._r2_within),
                "fixed_effects": str(fit._fixef), "vcov_type": str(fit._vcov_type),
                "cluster_variable": "+".join(fit._clustervar),
                "collinear_variables_removed": ";".join(map(str, fit._collin_vars)),
                "formula": specification["formula"],
            })
            equality_terms = config.get("equality_tests", {}).get(model)
            if equality_terms:
                tests.append(equality_test(model, frame, fit, equality_terms[0], equality_terms[1], clusters))
            elif model == "Z2":
                tests.append(equality_test(model, frame, fit, "log_abs_whale_net_flow", "log_abs_nonwhale_net_flow", clusters))
            elif model in {"C2", "N2"}:
                tests.append(equality_test(model, frame, fit, "signed_log_whale_net_flow", "signed_log_nonwhale_net_flow", clusters))
            log["completed_models"].append(model)
            atomic_json(log_path, log)

        coefficients = pd.concat(coefficient_tables, ignore_index=True)
        diagnostics_frame = pd.DataFrame(diagnostics)
        tests_frame = pd.DataFrame(tests)
        atomic_csv(output_dir / "coefficients.csv", coefficients)
        atomic_csv(output_dir / "model_diagnostics.csv", diagnostics_frame)
        atomic_csv(output_dir / "coefficient_equality_tests.csv", tests_frame)
        outputs = {}
        for name in ["coefficients.csv", "model_diagnostics.csv", "coefficient_equality_tests.csv"]:
            path = output_dir / name
            outputs[name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        log.update({"status": "complete_qc_passed", "completed_at": now(), "outputs": outputs})
        atomic_json(log_path, log)
        config["status"] = "complete_qc_passed"
        config["completed_at"] = log["completed_at"]
        atomic_json(config_path, config)
        print(json.dumps(log, indent=2), flush=True)
        return 0
    except Exception as error:
        log["status"] = "failed"
        log["failed_at"] = now()
        log["errors"].append(repr(error))
        atomic_json(log_path, log)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
