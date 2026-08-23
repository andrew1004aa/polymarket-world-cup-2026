#!/usr/bin/env python3
"""Run frozen alternative-cluster and wild-bootstrap checks for P2/P3."""

from __future__ import annotations

import hashlib
import argparse
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
DEFAULT_OUT = ROOT / "robustness_results/v1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, obj: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def scalar(value) -> float:
    array = np.asarray(value, dtype=float).reshape(-1)
    return float(array[0])


def equality_row(fit, coefficient_frame: pd.DataFrame, covariance_name: str,
                 cluster_spec: str, df: int, whale: str, non: str,
                 model: str) -> dict:
    indices = {name: i for i, name in enumerate(fit._coefnames)}
    iw, inn = indices[whale], indices[non]
    covariance = np.asarray(fit._vcov, dtype=float)
    variance = covariance[iw, iw] + covariance[inn, inn] - 2 * covariance[iw, inn]
    if not np.isfinite(variance) or variance <= 0:
        raise RuntimeError(f"Invalid P3 difference variance for {covariance_name}: {variance}")
    estimates = coefficient_frame.set_index("term")["estimate"]
    difference = float(estimates[whale] - estimates[non])
    se = math.sqrt(float(variance))
    t_value = difference / se
    critical = float(stats.t.ppf(0.975, df))
    return {
        "model": model, "covariance": covariance_name,
        "cluster_specification": cluster_spec,
        "null_hypothesis": f"{whale} = {non}",
        "estimate_difference": difference, "std_error": se,
        "t_value": t_value, "df": df,
        "p_value": float(2 * stats.t.sf(abs(t_value), df)),
        "conf_low": difference - critical * se,
        "conf_high": difference + critical * se,
    }


def fit_model(model: str, formula: str, data: pd.DataFrame, covariance: str,
              cluster_spec: str, df: int, keep_data: bool = False):
    print(f"Estimating {model} with {covariance}", flush=True)
    fit = pf.feols(
        formula, data=data, vcov={"CRV1": cluster_spec},
        copy_data=False, store_data=keep_data, lean=not keep_data,
    )
    tidy = fit.tidy().reset_index().rename(columns={
        "Coefficient": "term", "Estimate": "estimate",
        "Std. Error": "std_error", "t value": "t_value",
        "Pr(>|t|)": "p_value", "2.5%": "conf_low", "97.5%": "conf_high",
    })
    tidy.insert(0, "model", model)
    tidy.insert(1, "covariance", covariance)
    tidy.insert(2, "cluster_specification", cluster_spec)
    tidy["df"] = df
    diagnostics = {
        "model": model, "covariance": covariance,
        "cluster_specification": cluster_spec, "observations": int(fit._N),
        "markets": int(data.market_id.nunique()), "events": int(data.event_id.nunique()),
        "calendar_hours": int(data.calendar_hour_utc.nunique()), "df": df,
        "r_squared": float(fit._r2), "within_r_squared": float(fit._r2_within),
        "collinear_variables_removed": ";".join(map(str, fit._collin_vars)),
    }
    return fit, tidy, diagnostics


def bootstrap_row(fit, model: str, param: str, label: str, bootstrap: dict) -> dict:
    print(f"Wild cluster bootstrap: {label} ({bootstrap['reps']} draws)", flush=True)
    result = fit.wildboottest(
        reps=bootstrap["reps"], cluster=bootstrap["cluster"], param=param,
        weights_type=bootstrap["weights_type"], impose_null=bootstrap["impose_null"],
        bootstrap_type=bootstrap["bootstrap_type"], seed=bootstrap["seed"],
        k_adj=bootstrap["k_adj"], G_adj=bootstrap["G_adj"], parallel=False,
    )
    return {
        "model": model, "test": label, "parameter": param,
        "t_value": scalar(result["t value"]),
        "bootstrap_p_value": scalar(result["Pr(>|t|)"]),
        "reps": bootstrap["reps"], "cluster": bootstrap["cluster"],
        "weights_type": bootstrap["weights_type"],
        "bootstrap_type": "WCR" + bootstrap["bootstrap_type"],
        "impose_null": bootstrap["impose_null"], "seed": bootstrap["seed"],
        "k_adj": bootstrap["k_adj"], "G_adj": bootstrap["G_adj"],
    }


def report(coefficients: pd.DataFrame, equality: pd.DataFrame,
           bootstrap: pd.DataFrame, point_diff: float, bootstrap_note: str) -> str:
    focus = coefficients[coefficients.term.str.contains("signed_log_.*net_flow", regex=True)].copy()
    lines = [
        "# Primary-inference robustness report v1", "",
        "The frozen P2/P3 sample and coefficient formulas were unchanged. Only the variance estimator was changed.", "",
        "## Alternative clustered inference", "",
        "| Model | Term | Covariance | Estimate | Clustered SE | 95% CI | p-value |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in focus.itertuples():
        p = "<0.001" if row.p_value < .001 else f"{row.p_value:.3f}"
        lines.append(f"| {row.model} | {row.term} | {row.covariance} | {row.estimate:.8f} | {row.std_error:.8f} | [{row.conf_low:.8f}, {row.conf_high:.8f}] | {p} |")
    lines += ["", "### P3 whale-minus-non-whale test", "",
              "| Covariance | Difference | SE | 95% CI | p-value |", "|---|---:|---:|---:|---:|"]
    for row in equality.itertuples():
        p = "<0.001" if row.p_value < .001 else f"{row.p_value:.3f}"
        lines.append(f"| {row.covariance} | {row.estimate_difference:.8f} | {row.std_error:.8f} | [{row.conf_low:.8f}, {row.conf_high:.8f}] | {p} |")
    lines += ["", "## Event-level wild cluster bootstrap", "",
              "| Test | t-value | Bootstrap p-value | Draws |", "|---|---:|---:|---:|"]
    if bootstrap.empty:
        lines.append(f"| Not estimated | — | — | — |")
    else:
        for row in bootstrap.itertuples():
            p = "<0.001" if row.bootstrap_p_value < .001 else f"{row.bootstrap_p_value:.4f}"
            lines.append(f"| {row.test} | {row.t_value:.4f} | {p} | {row.reps:,} |")
    all_alt_sig = bool((focus.p_value < .05).all() and (equality.p_value < .05).all())
    all_boot_sig = None if bootstrap.empty else bool((bootstrap.bootstrap_p_value < .05).all())
    lines += [
        "", "## Quality control and interpretation", "",
        f"- Maximum point-estimate difference across covariance estimators: `{point_diff:.3e}`.",
        f"- All focal alternative-cluster tests significant at 5%: **{'YES' if all_alt_sig else 'NO'}**.",
        f"- Wild-bootstrap status: **{'NOT ESTIMATED' if all_boot_sig is None else ('all four significant at 5%' if all_boot_sig else 'not all four significant at 5%')}**.",
        f"- Bootstrap note: {bootstrap_note}",
        "- These checks assess inference robustness; they do not establish causality or prove a market-integrity violation.", "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v1")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-bootstrap", action="store_true")
    args = parser.parse_args()
    out = ROOT / args.output_dir if args.output_dir else DEFAULT_OUT
    config_path = out / "inference_config.json"
    config = json.loads(config_path.read_text())
    input_path = ROOT / config["input"]
    observed_hash = sha256(input_path)
    if observed_hash != config["input_sha256"]:
        raise SystemExit("Frozen input checksum mismatch")
    log_path = out / "inference_run_log.json"
    log = {"status": "running", "started_at": now(), "input_sha256": observed_hash,
           "software": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "scipy": scipy.__version__,
                        "pyfixest": pf.__version__}, "completed_fits": []}
    write_json(log_path, log)
    default_columns = ["delta_yes_price", "signed_log_total_net_flow",
               "signed_log_whale_net_flow", "signed_log_nonwhale_net_flow",
               "yes_price_t", "lagged_30m_price_change", "log_lagged_30m_volume",
               "log_minutes_to_kickoff", "market_id", "event_id", "calendar_hour_utc"]
    columns = config.get("input_columns", default_columns)
    equality_model = config.get("equality_model", "P3")
    equality_terms = config.get("equality_terms", ["signed_log_whale_net_flow", "signed_log_nonwhale_net_flow"])
    try:
        print("Loading frozen input", flush=True)
        data = pd.read_csv(input_path, usecols=columns)
        counts = {"rows": len(data), "markets": data.market_id.nunique(),
                  "events": data.event_id.nunique(), "calendar_hours": data.calendar_hour_utc.nunique()}
        for key, expected_key in [("rows", "expected_rows"), ("markets", "expected_markets"),
                                  ("events", "expected_events"), ("calendar_hours", "expected_calendar_hours")]:
            if counts[key] != config[expected_key]:
                raise RuntimeError(f"{key}: {counts[key]} != {config[expected_key]}")
        if data.isna().any().any():
            raise RuntimeError("Unexpected missing values in model columns")
        for col in ["market_id", "event_id", "calendar_hour_utc"]:
            data[col] = data[col].astype("category")

        frames, diagnostic_rows, equality_rows = [], [], []
        cluster_counts = {"event_id": counts["events"], "market_id": counts["markets"],
                          "calendar_hour_utc": counts["calendar_hours"]}
        for covariance, cluster_spec in config["covariance_estimators"].items():
            variables = [part.strip() for part in cluster_spec.split("+")]
            df = min(cluster_counts[x] for x in variables) - 1
            for model, formula in config["models"].items():
                fit, tidy, diagnostics = fit_model(model, formula, data, covariance, cluster_spec, df)
                frames.append(tidy); diagnostic_rows.append(diagnostics)
                if model == equality_model:
                    equality_rows.append(equality_row(fit, tidy, covariance, cluster_spec, df,
                                                      equality_terms[0], equality_terms[1], model))
                log["completed_fits"].append(f"{model}:{covariance}")
                write_json(log_path, log)

        coefficients = pd.concat(frames, ignore_index=True)
        equality = pd.DataFrame(equality_rows)
        diagnostics = pd.DataFrame(diagnostic_rows)
        focal = coefficients[coefficients.term.str.contains("signed_log_.*net_flow", regex=True)]
        point_diff = float(focal.groupby(["model", "term"]).estimate.agg(lambda x: x.max() - x.min()).max())
        if point_diff > 1e-10:
            raise RuntimeError(f"Point estimates changed across vcov fits: {point_diff}")

        # Preserve the completed alternative-cluster results before entering
        # the substantially more memory-intensive bootstrap stage.
        write_csv(out / "inference_coefficients.csv", coefficients)
        write_csv(out / "inference_equality_tests.csv", equality)
        write_csv(out / "inference_model_diagnostics.csv", diagnostics)

        if args.skip_bootstrap:
            bootstrap = pd.DataFrame(columns=[
                "model", "test", "parameter", "t_value", "bootstrap_p_value",
                "reps", "cluster", "weights_type", "bootstrap_type",
                "impose_null", "seed", "k_adj", "G_adj",
            ])
            bootstrap_note = (
                "PyFixest/wildboottest 0.3.2 was attempted with the frozen 9,999-draw "
                "specification but was stopped because its observation-by-draw memory "
                "requirements caused severe memory pressure on this machine. No lower-draw "
                "p-value is substituted."
            )
        else:
            bootstrap_note = "Completed exactly as frozen."

            # Bootstrap fits must retain source data for cluster extraction.
            event_df = counts["events"] - 1
            p2_fit, _, _ = fit_model("P2", config["models"]["P2"], data, "event_crv1_bootstrap", "event_id", event_df, True)
            p3_fit, _, _ = fit_model("P3", config["models"]["P3"], data, "event_crv1_bootstrap", "event_id", event_df, True)
            boot = config["bootstrap"]
            bootstrap_rows = [
                bootstrap_row(p2_fit, "P2", "signed_log_total_net_flow", "P2 total flow = 0", boot),
                bootstrap_row(p3_fit, "P3", "signed_log_whale_net_flow", "P3 whale flow = 0", boot),
                bootstrap_row(p3_fit, "P3", "signed_log_nonwhale_net_flow", "P3 non-whale flow = 0", boot),
            ]
            data["signed_log_common_split_flow"] = data["signed_log_whale_net_flow"] + data["signed_log_nonwhale_net_flow"]
            difference_formula = config["models"]["P3"].replace(
                "signed_log_whale_net_flow + signed_log_nonwhale_net_flow",
                "signed_log_whale_net_flow + signed_log_common_split_flow")
            diff_fit, _, _ = fit_model("P3_DIFFERENCE", difference_formula, data,
                                        "event_crv1_bootstrap", "event_id", event_df, True)
            bootstrap_rows.append(bootstrap_row(diff_fit, "P3_DIFFERENCE", "signed_log_whale_net_flow",
                                                 "P3 whale minus non-whale = 0", boot))
            bootstrap = pd.DataFrame(bootstrap_rows)

        outputs = {"inference_coefficients.csv": coefficients,
                   "inference_equality_tests.csv": equality,
                   "inference_model_diagnostics.csv": diagnostics,
                   "inference_bootstrap.csv": bootstrap}
        for name, frame in outputs.items():
            write_csv(out / name, frame)
        report_path = out / "inference_robustness_report.md"
        report_path.write_text(report(coefficients, equality, bootstrap, point_diff, bootstrap_note), encoding="utf-8")
        final_status = "complete_qc_passed_with_bootstrap_limitation" if args.skip_bootstrap else "complete_qc_passed"
        log.update({"status": final_status, "completed_at": now(), "counts": counts,
                    "maximum_point_estimate_difference": point_diff,
                    "bootstrap_note": bootstrap_note,
                    "outputs": {name: {"sha256": sha256(out / name), "bytes": (out / name).stat().st_size}
                                for name in [*outputs, report_path.name]}})
        write_json(log_path, log)
        config["status"] = final_status; config["completed_at"] = log["completed_at"]
        write_json(config_path, config)
        print(json.dumps(log, indent=2), flush=True)
        return 0
    except Exception as error:
        log.update({"status": "failed", "failed_at": now(), "error": repr(error)})
        write_json(log_path, log)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
