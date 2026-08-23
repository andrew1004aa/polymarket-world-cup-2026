#!/usr/bin/env python3
"""Estimate the frozen primary fixed-effects regressions (P1-P3).

The script is deliberately versioned and fail-closed. It verifies the frozen
input checksum and sample dimensions before estimation, never mutates upstream
data, and writes compact auditable results under regression_results/v1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf
import scipy
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "regression_results"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def package_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pyfixest": pf.__version__,
    }


def validate_input(data: pd.DataFrame, config: dict) -> dict:
    default_required = {
        "delta_yes_price", "signed_log_total_net_flow",
        "signed_log_whale_net_flow", "signed_log_nonwhale_net_flow",
        "yes_price_t", "lagged_30m_price_change",
        "log_lagged_30m_volume", "log_minutes_to_kickoff",
        "market_id", "calendar_hour_utc", "event_id",
        "model_sample_record_id",
    }
    required = set(config.get("required_columns", default_required))
    missing_columns = sorted(required - set(data.columns))
    if missing_columns:
        raise RuntimeError(f"Required columns missing: {missing_columns}")

    numeric = sorted(required - {
        "market_id", "calendar_hour_utc", "event_id",
        "model_sample_record_id",
    })
    missing_values = {name: int(data[name].isna().sum()) for name in required}
    missing_values = {key: value for key, value in missing_values.items() if value}
    nonfinite = {
        name: int((~np.isfinite(data[name].to_numpy(dtype=float))).sum())
        for name in numeric
    }
    nonfinite = {key: value for key, value in nonfinite.items() if value}

    result = {
        "rows": int(len(data)),
        "unique_record_ids": int(data["model_sample_record_id"].nunique()),
        "markets": int(data["market_id"].nunique()),
        "events": int(data["event_id"].nunique()),
        "calendar_hours": int(data["calendar_hour_utc"].nunique()),
        "missing_values": missing_values,
        "nonfinite_values": nonfinite,
    }
    expected = {
        "rows": int(config["expected_rows"]),
        "unique_record_ids": int(config["expected_rows"]),
        "markets": int(config["expected_markets"]),
        "events": int(config["expected_events"]),
        "calendar_hours": int(config["expected_calendar_hours"]),
    }
    errors = []
    for key, value in expected.items():
        if result[key] != value:
            errors.append(f"{key}: expected {value}, observed {result[key]}")
    if missing_values:
        errors.append(f"missing values: {missing_values}")
    if nonfinite:
        errors.append(f"non-finite values: {nonfinite}")
    result["errors"] = errors
    if errors:
        raise RuntimeError("Input validation failed: " + "; ".join(errors))
    return result


def fit_one(name: str, formula: str, data: pd.DataFrame) -> tuple[pd.DataFrame, dict, np.ndarray, list[str]]:
    print(f"Estimating {name}: {formula}", flush=True)
    fit = pf.feols(
        formula,
        data=data,
        vcov={"CRV1": "event_id"},
        copy_data=False,
        store_data=False,
        lean=True,
    )
    tidy = fit.tidy().reset_index()
    tidy.insert(0, "model", name)
    tidy = tidy.rename(columns={
        "Coefficient": "term", "Estimate": "estimate",
        "Std. Error": "std_error", "t value": "statistic",
        "Pr(>|t|)": "p_value", "2.5%": "conf_low",
        "97.5%": "conf_high",
    })
    diagnostics = {
        "model": name,
        "formula": formula,
        "observations": int(fit._N),
        "r_squared": float(fit._r2),
        "within_r_squared": float(fit._r2_within),
        "fixed_effects": str(fit._fixef),
        "vcov_type": str(fit._vcov_type),
        "cluster_variables": "+".join(fit._clustervar),
        "event_clusters": int(data["event_id"].nunique()),
        "collinear_variables_removed": ";".join(map(str, fit._collin_vars)),
    }
    covariance = np.asarray(fit._vcov, dtype=float).copy()
    coefficient_names = [str(value) for value in fit._coefnames]
    return tidy, diagnostics, covariance, coefficient_names


def equality_test(
    coefficients: pd.DataFrame,
    covariance: np.ndarray,
    names: list[str],
    cluster_count: int,
    config: dict,
) -> pd.DataFrame:
    whale, nonwhale = config.get("equality_terms", [
        "signed_log_whale_net_flow", "signed_log_nonwhale_net_flow"])
    equality_model = config.get("equality_model", "P3")
    index = {name: position for position, name in enumerate(names)}
    if whale not in index or nonwhale not in index:
        raise RuntimeError("P3 equality-test coefficients are unavailable")
    iw, inn = index[whale], index[nonwhale]
    beta = coefficients.set_index("term")["estimate"]
    difference = float(beta[whale] - beta[nonwhale])
    variance = float(
        covariance[iw, iw] + covariance[inn, inn] - 2 * covariance[iw, inn]
    )
    if not math.isfinite(variance) or variance <= 0:
        raise RuntimeError(f"Invalid equality-test variance: {variance}")
    standard_error = math.sqrt(variance)
    statistic = difference / standard_error
    degrees_freedom = cluster_count - 1
    p_value = float(2 * stats.t.sf(abs(statistic), df=degrees_freedom))
    critical = float(stats.t.ppf(0.975, df=degrees_freedom))
    return pd.DataFrame([{
        "model": equality_model,
        "null_hypothesis": f"{whale} = {nonwhale}",
        "coefficient_difference": difference,
        "std_error": standard_error,
        "t_statistic": statistic,
        "df": degrees_freedom,
        "p_value": p_value,
        "conf_low": difference - critical * standard_error,
        "conf_high": difference + critical * standard_error,
        "covariance_source": f"{equality_model} CRV1 covariance clustered by event_id",
    }])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v1")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output = ROOT / args.output_dir if args.output_dir else RESULTS_ROOT / args.version
    config_path = output / "config.json"
    if not config_path.exists():
        raise SystemExit(f"Missing configuration: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    completed = output / "run_log.json"
    if completed.exists() and not args.force:
        prior = json.loads(completed.read_text(encoding="utf-8"))
        if prior.get("status") == "complete_qc_passed":
            print(f"Regression {args.version} already completed; use --force for a deliberate rerun.")
            return 0

    input_path = ROOT / config["input"]
    observed_sha = sha256(input_path)
    if observed_sha != config["input_sha256"]:
        raise SystemExit(
            f"Frozen input checksum mismatch: expected {config['input_sha256']}, "
            f"observed {observed_sha}"
        )

    run_log = {
        "version": args.version,
        "status": "running",
        "started_at": utc_now(),
        "input": config["input"],
        "input_sha256": observed_sha,
        "software": package_versions(),
        "completed_models": [],
        "errors": [],
    }
    atomic_json(completed, run_log)

    default_columns = [
        "model_sample_record_id", "market_id", "event_id",
        "calendar_hour_utc", "delta_yes_price",
        "signed_log_total_net_flow", "signed_log_whale_net_flow",
        "signed_log_nonwhale_net_flow", "yes_price_t",
        "lagged_30m_price_change", "log_lagged_30m_volume",
        "log_minutes_to_kickoff",
    ]
    columns = config.get("input_columns", default_columns)
    print(f"Loading {input_path.relative_to(ROOT)}", flush=True)
    try:
        data = pd.read_csv(input_path, usecols=columns)
        validation = validate_input(data, config)
        run_log["input_validation"] = validation
        atomic_json(completed, run_log)

        # The record identifier is required only for the pre-fit uniqueness
        # check. Categoricals substantially reduce memory during FE absorption.
        data = data.drop(columns=["model_sample_record_id"])
        for name in ["market_id", "event_id", "calendar_hour_utc"]:
            data[name] = data[name].astype("category")

        coefficient_frames = []
        diagnostic_rows = []
        split_covariance = None
        split_names = None
        split_coefficients = None
        equality_model = config.get("equality_model", "P3")
        for name, formula in config["models"].items():
            tidy, diagnostics, covariance, names = fit_one(name, formula, data)
            coefficient_frames.append(tidy)
            diagnostic_rows.append(diagnostics)
            if name == equality_model:
                split_covariance = covariance
                split_names = names
                split_coefficients = tidy
            run_log["completed_models"].append(name)
            atomic_json(completed, run_log)

        coefficients = pd.concat(coefficient_frames, ignore_index=True)
        diagnostics = pd.DataFrame(diagnostic_rows)
        test = equality_test(
            split_coefficients, split_covariance, split_names,
            int(validation["events"]), config,
        )
        equality_output = config.get("equality_output", "whale_nonwhale_equality_test.csv")
        atomic_csv(output / "coefficients.csv", coefficients)
        atomic_csv(output / "model_diagnostics.csv", diagnostics)
        atomic_csv(output / equality_output, test)

        run_log.update({
            "status": "complete_qc_passed",
            "completed_at": utc_now(),
            "outputs": {
                name: {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
                for name, path in {
                    "coefficients": output / "coefficients.csv",
                    "model_diagnostics": output / "model_diagnostics.csv",
                    "coefficient_equality_test": output / equality_output,
                }.items()
            },
        })
        atomic_json(completed, run_log)
        config["status"] = "complete_qc_passed"
        config["completed_at"] = run_log["completed_at"]
        atomic_json(config_path, config)
        print(json.dumps(run_log, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as error:
        run_log["status"] = "failed"
        run_log["failed_at"] = utc_now()
        run_log["errors"].append(repr(error))
        atomic_json(completed, run_log)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
