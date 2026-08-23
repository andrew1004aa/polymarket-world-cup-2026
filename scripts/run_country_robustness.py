#!/usr/bin/env python3
"""Run frozen country outright zero, non-overlap and clustering checks."""

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
INPUT = ROOT / "regression_inputs/country_v1/outright_5m_complete_case.csv.gz"
INPUT_CONFIG = ROOT / "regression_inputs/country_v1/config.json"
OUT = ROOT / "robustness_results/country_v1"
CONTROLS = "yes_price_t + lagged_30m_price_change + log_lagged_30m_volume"
FE = "market_id + calendar_hour_utc"
MODELS = {
    "OZ1": f"any_price_change ~ log_abs_total_net_flow + {CONTROLS} | {FE}",
    "OZ2": f"any_price_change ~ log_abs_whale_net_flow + log_abs_nonwhale_net_flow + {CONTROLS} | {FE}",
    "ON1": f"delta_yes_price ~ signed_log_total_net_flow + {CONTROLS} | {FE}",
    "ON2": f"delta_yes_price ~ signed_log_whale_net_flow + signed_log_nonwhale_net_flow + {CONTROLS} | {FE}",
    "O1": f"delta_yes_price ~ signed_log_total_net_flow + {CONTROLS} | {FE}",
    "O2": f"delta_yes_price ~ signed_log_whale_net_flow + signed_log_nonwhale_net_flow + {CONTROLS} | {FE}",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024): h.update(chunk)
    return h.hexdigest()


def json_write(path: Path, obj: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def csv_write(path: Path, frame: pd.DataFrame) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False); tmp.replace(path)


def fit(model: str, family: str, formula: str, data: pd.DataFrame,
        covariance: str, cluster: str, df: int):
    print(f"Estimating {model}:{covariance} on {len(data):,} rows", flush=True)
    result = pf.feols(formula, data=data, vcov={"CRV1": cluster},
                      copy_data=False, store_data=False, lean=True)
    tidy = result.tidy().reset_index().rename(columns={
        "Coefficient":"term", "Estimate":"estimate", "Std. Error":"std_error",
        "t value":"t_value", "Pr(>|t|)":"p_value", "2.5%":"conf_low",
        "97.5%":"conf_high"})
    tidy.insert(0,"family",family); tidy.insert(1,"model",model)
    tidy.insert(2,"covariance",covariance); tidy.insert(3,"cluster_specification",cluster)
    tidy["df"] = df
    diag = {"family":family,"model":model,"covariance":covariance,
            "cluster_specification":cluster,"observations":int(result._N),
            "markets":int(data.market_id.nunique()),"dates":int(data.date_utc.nunique()),
            "calendar_hours":int(data.calendar_hour_utc.nunique()),"df":df,
            "r_squared":float(result._r2),"within_r_squared":float(result._r2_within),
            "collinear_variables_removed":";".join(map(str,result._collin_vars))}
    return result, tidy, diag


def equality(result, tidy: pd.DataFrame, family: str, model: str,
             covariance: str, cluster: str, df: int, terms: tuple[str,str]) -> dict:
    a,b=terms; idx={x:i for i,x in enumerate(result._coefnames)}
    cov=np.asarray(result._vcov,float); ia,ib=idx[a],idx[b]
    variance=float(cov[ia,ia]+cov[ib,ib]-2*cov[ia,ib])
    if variance <= 0 or not math.isfinite(variance):
        raise RuntimeError(f"Invalid equality variance {model}:{covariance}: {variance}")
    beta=tidy.set_index("term").estimate; difference=float(beta[a]-beta[b]); se=math.sqrt(variance)
    t=difference/se; critical=float(stats.t.ppf(.975,df))
    return {"family":family,"model":model,"covariance":covariance,
            "cluster_specification":cluster,"term_a":a,"term_b":b,
            "estimate_difference":difference,"std_error":se,"t_value":t,"df":df,
            "p_value":float(2*stats.t.sf(abs(t),df)),
            "conf_low":difference-critical*se,"conf_high":difference+critical*se}


def report(coef: pd.DataFrame, tests: pd.DataFrame, diag: pd.DataFrame,
           zero_share: float, nonoverlap_rows: int, point_diff: float,
           two_way_nonfinite_terms: list[str]) -> str:
    flow=coef[coef.term.str.contains("flow")].copy()
    lines=["# Country Outright Robustness Report v1","",
           "## Scope","",
           "These secondary checks use the frozen 48-country five-minute sample, country-market and UTC calendar-hour fixed effects, and the original controls. They do not replace O1/O2.","",
           f"- Full-sample zero price-change share: {zero_share:.2%}.",
           f"- UTC-aligned non-overlapping sample: {nonoverlap_rows:,} rows.","",
           "## Any five-minute price update","",
           "| Model | Term | Estimate | SE | 95% CI | p-value |","|---|---|---:|---:|---:|---:|"]
    for r in flow[flow.family=="any_change"].itertuples():
        p="<0.001" if r.p_value<.001 else f"{r.p_value:.3f}"
        lines.append(f"| {r.model} | {r.term} | {r.estimate:.6f} | {r.std_error:.6f} | [{r.conf_low:.6f}, {r.conf_high:.6f}] | {p} |")
    lines += ["","## Non-overlapping five-minute windows","",
              "| Model | Term | Estimate | SE | 95% CI | p-value |","|---|---|---:|---:|---:|---:|"]
    for r in flow[flow.family=="nonoverlap"].itertuples():
        p="<0.001" if r.p_value<.001 else f"{r.p_value:.3f}"
        lines.append(f"| {r.model} | {r.term} | {r.estimate:.8f} | {r.std_error:.8f} | [{r.conf_low:.8f}, {r.conf_high:.8f}] | {p} |")
    lines += ["","## Alternative clustering for original O1/O2","",
              "| Model | Term | Covariance | Estimate | SE | 95% CI | p-value |",
              "|---|---|---|---:|---:|---:|---:|"]
    for r in flow[flow.family=="alternative_cluster"].itertuples():
        p="<0.001" if r.p_value<.001 else f"{r.p_value:.3f}"
        lines.append(f"| {r.model} | {r.term} | {r.covariance} | {r.estimate:.8f} | {r.std_error:.8f} | [{r.conf_low:.8f}, {r.conf_high:.8f}] | {p} |")
    lines += ["","## Coefficient-difference tests","",
              "| Family | Covariance | Difference | SE | p-value |","|---|---|---:|---:|---:|"]
    for r in tests.itertuples():
        p="<0.001" if r.p_value<.001 else f"{r.p_value:.3f}"
        lines.append(f"| {r.family} | {r.covariance} | {r.estimate_difference:.8f} | {r.std_error:.8f} | {p} |")
    lines += ["","## Quality control and interpretation","",
              f"- Maximum O1/O2 point-estimate difference across covariance estimators: `{point_diff:.3e}`.",
              f"- Two-way covariance non-finite standard-error terms: `{'; '.join(two_way_nonfinite_terms) if two_way_nonfinite_terms else 'none'}`.",
              "- The two-way covariance matrix is retained only as a sensitivity check when a control-term variance is non-finite; market-only and date-only inference remain fully finite.",
              "- Any-change coefficients describe update incidence, not directional magnitude.",
              "- Robustness across these checks does not establish causality, private information or manipulation.",""]
    return "\n".join(lines)


def main() -> int:
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(INPUT_CONFIG.read_text()); expected=next(x for x in cfg["files"] if Path(x["path"]).name==INPUT.name)
    if sha256(INPUT)!=expected["sha256"]: raise SystemExit("Frozen input checksum mismatch")
    cols=["market_id","calendar_hour_utc","date_utc","minute_start_timestamp","delta_yes_price",
          "net_signed_flow_usdc","signed_log_total_net_flow","signed_log_whale_net_flow",
          "signed_log_nonwhale_net_flow",
          "yes_price_t","lagged_30m_price_change","log_lagged_30m_volume"]
    data=pd.read_csv(INPUT,usecols=cols)
    if len(data)!=expected["rows"] or data.market_id.nunique()!=48 or data.isna().any().any():
        raise SystemExit("Input validation failed")
    data["any_price_change"]=(data.delta_yes_price!=0).astype(int)
    data["log_abs_total_net_flow"]=np.log1p(np.abs(data.net_signed_flow_usdc))
    # abs(sign(x) * log1p(abs(x))) is exactly log1p(abs(x)); use the
    # frozen signed-log fields without reconstructing or mutating raw inputs.
    data["log_abs_whale_net_flow"]=np.abs(data.signed_log_whale_net_flow)
    data["log_abs_nonwhale_net_flow"]=np.abs(data.signed_log_nonwhale_net_flow)
    nonoverlap=data[data.minute_start_timestamp.mod(300).eq(0)].copy()
    for frame in [data,nonoverlap]:
        for col in ["market_id","calendar_hour_utc","date_utc"]: frame[col]=frame[col].astype("category")
    log={"status":"running","started_at":now(),"input_sha256":sha256(INPUT),
         "software":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,
                     "scipy":scipy.__version__,"pyfixest":pf.__version__},"completed_models":[]}
    json_write(OUT/"run_log.json",log)
    frames=[]; diagnostics=[]; test_rows=[]
    try:
        specs=[("any_change","OZ1",data,"market_crv1","market_id",47),
               ("any_change","OZ2",data,"market_crv1","market_id",47),
               ("nonoverlap","ON1",nonoverlap,"market_crv1","market_id",47),
               ("nonoverlap","ON2",nonoverlap,"market_crv1","market_id",47)]
        for family,model,frame,cov,cluster,df in specs:
            result,tidy,diag=fit(model,family,MODELS[model],frame,cov,cluster,df)
            frames.append(tidy); diagnostics.append(diag)
            if model=="OZ2": test_rows.append(equality(result,tidy,family,model,cov,cluster,df,("log_abs_whale_net_flow","log_abs_nonwhale_net_flow")))
            if model=="ON2": test_rows.append(equality(result,tidy,family,model,cov,cluster,df,("signed_log_whale_net_flow","signed_log_nonwhale_net_flow")))
            log["completed_models"].append(f"{model}:{cov}"); json_write(OUT/"run_log.json",log)
        clusters=[("market_crv1","market_id",47),("date_crv1","date_utc",49),
                  ("market_date_two_way_crv1","market_id + date_utc",47)]
        for cov,cluster,df in clusters:
            for model in ["O1","O2"]:
                result,tidy,diag=fit(model,"alternative_cluster",MODELS[model],data,cov,cluster,df)
                frames.append(tidy); diagnostics.append(diag)
                if model=="O2": test_rows.append(equality(result,tidy,"alternative_cluster",model,cov,cluster,df,("signed_log_whale_net_flow","signed_log_nonwhale_net_flow")))
                log["completed_models"].append(f"{model}:{cov}"); json_write(OUT/"run_log.json",log)
        coefficients=pd.concat(frames,ignore_index=True); tests=pd.DataFrame(test_rows); diag=pd.DataFrame(diagnostics)
        alt=coefficients[(coefficients.family=="alternative_cluster") & coefficients.term.str.contains("flow")]
        point_diff=float(alt.groupby(["model","term"]).estimate.agg(lambda x:x.max()-x.min()).max())
        if point_diff>1e-10: raise RuntimeError(f"Point estimates changed across vcov: {point_diff}")
        if alt[["std_error","p_value","conf_low","conf_high"]].isna().any().any():
            raise RuntimeError("A focal alternative-cluster result is non-finite")
        two_way_nonfinite=coefficients[
            (coefficients.covariance=="market_date_two_way_crv1") & coefficients.std_error.isna()
        ]["term"].drop_duplicates().tolist()
        csv_write(OUT/"coefficients.csv",coefficients); csv_write(OUT/"coefficient_equality_tests.csv",tests)
        csv_write(OUT/"model_diagnostics.csv",diag)
        report_path=OUT/"country_robustness_report.md"
        report_path.write_text(report(coefficients,tests,diag,float((data.delta_yes_price==0).mean()),len(nonoverlap),point_diff,two_way_nonfinite),encoding="utf-8")
        outputs=[OUT/"coefficients.csv",OUT/"coefficient_equality_tests.csv",OUT/"model_diagnostics.csv",report_path]
        status="complete_qc_passed_with_two_way_covariance_limitation" if two_way_nonfinite else "complete_qc_passed"
        log.update({"status":status,"completed_at":now(),"rows":len(data),
                    "nonoverlap_rows":len(nonoverlap),"maximum_point_estimate_difference":point_diff,
                    "two_way_nonfinite_standard_error_terms":two_way_nonfinite,
                    "outputs":{p.name:{"sha256":sha256(p),"bytes":p.stat().st_size} for p in outputs}})
        json_write(OUT/"run_log.json",log); print(json.dumps(log,indent=2),flush=True); return 0
    except Exception as e:
        log.update({"status":"failed","failed_at":now(),"error":repr(e)}); json_write(OUT/"run_log.json",log); raise


if __name__=="__main__": raise SystemExit(main())
