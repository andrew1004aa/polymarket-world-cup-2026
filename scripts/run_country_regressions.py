#!/usr/bin/env python3
"""Estimate frozen country outright O1/O2 models and horizon checks."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf
import scipy
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "regression_inputs/country_v1"
DEFAULT_OUT = ROOT / "regression_results/country_v1"
CONTROLS = "yes_price_t + lagged_30m_price_change + log_lagged_30m_volume"
FORMULAS = {
    "O1": f"delta_yes_price ~ signed_log_total_net_flow + {CONTROLS} | market_id + calendar_hour_utc",
    "O2": f"delta_yes_price ~ signed_log_whale_net_flow + signed_log_nonwhale_net_flow + {CONTROLS} | market_id + calendar_hour_utc",
}
FILES = {
    "primary_5m": "outright_5m_complete_case.csv.gz",
    "matched_5m": "outright_5m_matched_5_30.csv.gz",
    "matched_30m": "outright_30m_matched_5_30.csv.gz",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, obj: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def equality(fit, tidy: pd.DataFrame, sample: str, whale: str, non: str,
             cluster_count: int) -> dict:
    idx = {name: i for i, name in enumerate(fit._coefnames)}
    cov = np.asarray(fit._vcov, dtype=float)
    iw, inn = idx[whale], idx[non]
    variance = float(cov[iw, iw] + cov[inn, inn] - 2 * cov[iw, inn])
    if variance <= 0 or not math.isfinite(variance):
        raise RuntimeError(f"Invalid equality-test variance: {variance}")
    beta = tidy.set_index("term").estimate
    diff, se, df = float(beta[whale] - beta[non]), math.sqrt(variance), cluster_count - 1
    t = diff / se
    critical = float(stats.t.ppf(.975, df))
    return {"sample": sample, "model": "O2", "estimate_difference": diff,
            "std_error": se, "t_value": t, "df": df,
            "p_value": float(2 * stats.t.sf(abs(t), df)),
            "conf_low": diff - critical * se, "conf_high": diff + critical * se}


def fit(sample: str, model: str, formula: str, data: pd.DataFrame):
    print(f"Estimating {sample}:{model}", flush=True)
    result = pf.feols(formula, data=data, vcov={"CRV1": "market_id"},
                      copy_data=False, store_data=False, lean=True)
    tidy = result.tidy().reset_index().rename(columns={
        "Coefficient": "term", "Estimate": "estimate", "Std. Error": "std_error",
        "t value": "t_value", "Pr(>|t|)": "p_value", "2.5%": "conf_low",
        "97.5%": "conf_high",
    })
    tidy.insert(0, "sample", sample); tidy.insert(1, "model", model)
    diagnostics = {"sample": sample, "model": model, "formula": formula,
                   "observations": int(result._N), "markets": int(data.market_id.nunique()),
                   "calendar_hours": int(data.calendar_hour_utc.nunique()),
                   "clusters": int(data.market_id.nunique()), "cluster_variable": "market_id",
                   "r_squared": float(result._r2), "within_r_squared": float(result._r2_within),
                   "collinear_variables_removed": ";".join(map(str, result._collin_vars))}
    return result, tidy, diagnostics


def make_report(coefficients: pd.DataFrame, tests: pd.DataFrame,
                diagnostics: pd.DataFrame, sample_stats: list[dict]) -> str:
    focal = coefficients[coefficients.term.str.contains("signed_log_.*net_flow", regex=True)]
    lines = ["# Country Outright Regression Report v1", "",
             "## Status and scope", "",
             "All models are secondary analyses of 48 country winner markets and are not pooled with match contracts. Models include country-market and UTC calendar-hour fixed effects; CRV1 standard errors are clustered by country market.", "",
             "## Samples", "", "| Sample | Rows | Markets | Calendar hours | Zero price-change share |",
             "|---|---:|---:|---:|---:|"]
    labels = {"primary_5m":"Five-minute complete case", "matched_5m":"Five-minute matched",
              "matched_30m":"Thirty-minute matched"}
    for row in sample_stats:
        lines.append(f"| {labels[row['sample']]} | {row['rows']:,} | {row['markets']} | {row['calendar_hours']:,} | {row['zero_share']:.2%} |")
    lines += ["", "## Flow coefficients", "",
              "| Sample | Model | Term | Estimate | Clustered SE | 95% CI | p-value |",
              "|---|---|---|---:|---:|---:|---:|"]
    for row in focal.itertuples():
        p = "<0.001" if row.p_value < .001 else f"{row.p_value:.3f}"
        lines.append(f"| {labels[row.sample]} | {row.model} | {row.term} | {row.estimate:.8f} | {row.std_error:.8f} | [{row.conf_low:.8f}, {row.conf_high:.8f}] | {p} |")
    lines += ["", "## Whale-minus-non-whale tests", "",
              "| Sample | Difference | SE | 95% CI | p-value |", "|---|---:|---:|---:|---:|"]
    for row in tests.itertuples():
        p = "<0.001" if row.p_value < .001 else f"{row.p_value:.3f}"
        lines.append(f"| {labels[row.sample]} | {row.estimate_difference:.8f} | {row.std_error:.8f} | [{row.conf_low:.8f}, {row.conf_high:.8f}] | {p} |")
    p5 = focal[(focal['sample']=='primary_5m') & (focal['model']=='O2')].set_index('term')
    p30 = focal[(focal['sample']=='matched_30m') & (focal['model']=='O2')].set_index('term')
    lines += ["", "## Interpretation guardrails", "",
              f"- The primary five-minute O2 whale coefficient is `{p5.loc['signed_log_whale_net_flow','estimate']:.8f}`; the non-whale coefficient is `{p5.loc['signed_log_nonwhale_net_flow','estimate']:.8f}`.",
              f"- In the matched thirty-minute model the corresponding coefficients are `{p30.loc['signed_log_whale_net_flow','estimate']:.8f}` and `{p30.loc['signed_log_nonwhale_net_flow','estimate']:.8f}`.",
              "- Separate horizon coefficients are descriptive and are not a formal test of persistence or reversal.",
              "- These associations do not establish causal impact, information advantage, manipulation or a market-integrity breach.", ""]
    return "\n".join(lines)


def main() -> int:
    parser=argparse.ArgumentParser();parser.add_argument("--version",default="country_v1")
    parser.add_argument("--input-dir",default=None);parser.add_argument("--output-dir",default=None);args=parser.parse_args()
    input_dir=ROOT/args.input_dir if args.input_dir else DEFAULT_INPUT;out=ROOT/args.output_dir if args.output_dir else DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((input_dir / "config.json").read_text());v3=args.version.startswith("v3")
    split_terms=("p99_signed_log_large_net_flow","p99_signed_log_ordinary_net_flow") if v3 else ("signed_log_whale_net_flow","signed_log_nonwhale_net_flow")
    formulas={
        "O1": f"delta_yes_price ~ signed_log_total_net_flow + {CONTROLS} | market_id + calendar_hour_utc",
        "O2": f"delta_yes_price ~ {split_terms[0]} + {split_terms[1]} + {CONTROLS} | market_id + calendar_hour_utc",
    }
    expected = {Path(x["path"]).name: x for x in config["files"]}
    log = {"status":"running", "started_at":now(),
           "software":{"python":platform.python_version(), "numpy":np.__version__,
                       "pandas":pd.__version__, "scipy":scipy.__version__, "pyfixest":pf.__version__},
           "completed_models":[]}
    atomic_json(out / "run_log.json", log)
    all_coef, all_diag, all_tests, sample_stats = [], [], [], []
    required = ["delta_yes_price", "signed_log_total_net_flow", *split_terms,
                "yes_price_t", "lagged_30m_price_change",
                "log_lagged_30m_volume", "market_id", "calendar_hour_utc"]
    try:
        for sample, filename in FILES.items():
            path = input_dir / filename
            if sha256(path) != expected[filename]["sha256"]:
                raise RuntimeError(f"Checksum mismatch: {filename}")
            data = pd.read_csv(path, usecols=required)
            if data.isna().any().any() or not np.isfinite(data.select_dtypes('number')).all().all():
                raise RuntimeError(f"Invalid fitted values: {filename}")
            expected_markets=int(config.get("expected_markets",48))
            if len(data) != expected[filename]["rows"] or data.market_id.nunique() != expected_markets:
                raise RuntimeError(f"Unexpected dimensions: {filename}")
            sample_stats.append({"sample":sample, "rows":len(data), "markets":expected_markets,
                                 "calendar_hours":data.calendar_hour_utc.nunique(),
                                 "zero_share":float((data.delta_yes_price==0).mean())})
            data.market_id = data.market_id.astype("category")
            data.calendar_hour_utc = data.calendar_hour_utc.astype("category")
            for model, formula in formulas.items():
                result, tidy, diag = fit(sample, model, formula, data)
                all_coef.append(tidy); all_diag.append(diag)
                if model == "O2": all_tests.append(equality(result, tidy, sample,*split_terms,expected_markets))
                log["completed_models"].append(f"{sample}:{model}")
                atomic_json(out / "run_log.json", log)
        coefficients = pd.concat(all_coef, ignore_index=True)
        diagnostics = pd.DataFrame(all_diag); tests = pd.DataFrame(all_tests)
        atomic_csv(out / "coefficients.csv", coefficients)
        atomic_csv(out / "model_diagnostics.csv", diagnostics)
        equality_name="large_ordinary_equality_tests.csv" if v3 else "whale_nonwhale_equality_tests.csv"
        atomic_csv(out / equality_name, tests)
        report = out / "country_regression_report.md"
        if v3: report.write_text("# Country outright regression report v3\n\nResults built and QC passed. See README.md for the reviewed interpretation.\n",encoding="utf-8")
        else: report.write_text(make_report(coefficients, tests, diagnostics, sample_stats), encoding="utf-8")
        outputs = [out/"coefficients.csv", out/"model_diagnostics.csv",out/equality_name, report]
        log.update({"status":"complete_qc_passed", "completed_at":now(),
                    "sample_stats":sample_stats,
                    "outputs":{p.name:{"sha256":sha256(p),"bytes":p.stat().st_size} for p in outputs}})
        atomic_json(out / "run_log.json", log)
        atomic_json(out / "config.json", {"version":args.version, "status":log["status"],
                    "completed_at":log["completed_at"], "input_config_sha256":sha256(input_dir/"config.json"),
                    "models":formulas, "fixed_effects":["market_id","calendar_hour_utc"],
                    "vcov":"CRV1", "cluster":"market_id", "cluster_count":int(config.get("expected_markets",48))})
        print(json.dumps(log, indent=2), flush=True)
        return 0
    except Exception as error:
        log.update({"status":"failed", "failed_at":now(), "error":repr(error)})
        atomic_json(out / "run_log.json", log)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
