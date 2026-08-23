#!/usr/bin/env python3
"""Estimate pooled-interaction and stratified H4 timing models."""

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


ROOT=Path(__file__).resolve().parents[1];DEFAULT_OUT=ROOT/"robustness_results"/"v1"
CONTROLS="yes_price_t + lagged_30m_price_change + log_lagged_30m_volume + log_minutes_to_kickoff"
NONREF=["12_to_24h","6_to_12h","1_to_6h","last_60m"]


def now():return datetime.now(timezone.utc).isoformat()
def sha256(path):
    d=hashlib.sha256()
    with path.open("rb") as h:
        while c:=h.read(1024*1024):d.update(c)
    return d.hexdigest()
def atomic_json(path,payload):
    t=path.with_suffix(path.suffix+".tmp");t.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");t.replace(path)
def atomic_csv(path,frame):
    t=path.with_suffix(path.suffix+".tmp");frame.to_csv(t,index=False);t.replace(path)


def coef_frame(model,fit,clusters):
    names=[str(v) for v in fit._coefnames];b=np.asarray(fit.coef(),float);v=np.asarray(fit._vcov,float);se=np.sqrt(np.diag(v));tv=b/se
    df=clusters-1;crit=stats.t.ppf(.975,df)
    return pd.DataFrame({"model":model,"term":names,"estimate":b,"std_error":se,"t_statistic":tv,"df":df,
                         "p_value":2*stats.t.sf(np.abs(tv),df),"conf_low":b-crit*se,"conf_high":b+crit*se})


def linear_result(model,label,weights,names,b,v,clusters):
    vector=np.zeros(len(names))
    for term,value in weights.items():vector[names.index(term)]=value
    estimate=float(vector@b);variance=float(vector@v@vector);se=math.sqrt(variance);tv=estimate/se;df=clusters-1;crit=stats.t.ppf(.975,df)
    return {"model":model,"contrast":label,"estimate":estimate,"std_error":se,"t_statistic":tv,"df":df,
            "p_value":float(2*stats.t.sf(abs(tv),df)),"conf_low":estimate-crit*se,"conf_high":estimate+crit*se}


def joint_test(model,label,terms,names,b,v,clusters):
    r=np.zeros((len(terms),len(names)))
    for row,term in enumerate(terms):r[row,names.index(term)]=1
    rb=r@b;rv=r@v@r.T;wald=float(rb@np.linalg.solve(rv,rb));q=len(terms);fstat=wald/q;df2=clusters-1
    return {"model":model,"test":label,"restrictions":q,"f_statistic":fstat,"df_num":q,"df_denom":df2,
            "p_value":float(stats.f.sf(fstat,q,df2))}


def main():
    p=argparse.ArgumentParser();p.add_argument("--force",action="store_true");p.add_argument("--version",default="v1");p.add_argument("--output-dir",default=None);args=p.parse_args()
    out=ROOT/args.output_dir if args.output_dir else DEFAULT_OUT;config_path=out/"h4_timing_config.json";config=json.loads(config_path.read_text())
    log_path=out/"h4_timing_run_log.json"
    if log_path.exists() and not args.force:
        prior=json.loads(log_path.read_text())
        if prior.get("status")=="complete_qc_passed":print("H4 timing already completed; use --force to rerun.");return 0
    source=ROOT/config["input"]
    if sha256(source)!=config["input_sha256"]:raise SystemExit("H4 timing checksum mismatch")
    log={"version":args.version,"status":"running","started_at":now(),"software":{"python":platform.python_version(),"pandas":pd.__version__,"numpy":np.__version__,"scipy":scipy.__version__,"pyfixest":pf.__version__},"completed_models":[],"errors":[]};atomic_json(log_path,log)
    split=config.get("split_flow_fields",["signed_log_whale_net_flow","signed_log_nonwhale_net_flow"]);prefixes=config.get("interaction_prefixes",["whale","nonwhale"])
    base=["market_id","event_id","calendar_hour_utc","time_band","delta_yes_price",*split,"yes_price_t","lagged_30m_price_change","log_lagged_30m_volume","log_minutes_to_kickoff"]
    dummies=[f"band_{x}" for x in NONREF];wx=[f"{prefixes[0]}_x_{x}" for x in NONREF];nx=[f"{prefixes[1]}_x_{x}" for x in NONREF]
    try:
        print("Loading H4 timing sample",flush=True);data=pd.read_csv(source,usecols=base+dummies+wx+nx)
        if len(data)!=int(config["expected_rows"]) or data.isna().any().any():raise RuntimeError("H4 input validation failed")
        for name in ["market_id","event_id","calendar_hour_utc","time_band"]:data[name]=data[name].astype("category")
        clusters=int(data["event_id"].nunique());frames=[];diagnostics=[];marginals=[];joint=[]
        pooled_terms=" + ".join([*split,*dummies,*wx,*nx])
        formula=f"delta_yes_price ~ {pooled_terms} + {CONTROLS} | market_id + calendar_hour_utc"
        print("Estimating H4_POOLED",flush=True);fit=pf.feols(formula,data=data,vcov={"CRV1":"event_id"},fixef_rm="singleton",copy_data=False,store_data=False,lean=True)
        frame=coef_frame("H4_POOLED",fit,clusters);frames.append(frame);names=[str(v) for v in fit._coefnames];b=np.asarray(fit.coef(),float);v=np.asarray(fit._vcov,float)
        diagnostics.append({"model":"H4_POOLED","source_rows":len(data),"fitted_rows":int(fit._N),"markets":int(data.market_id.nunique()),"events":clusters,"calendar_hours":int(data.calendar_hour_utc.nunique()),"r_squared":float(fit._r2),"within_r_squared":float(fit._r2_within),"collinear_variables_removed":";".join(map(str,fit._collin_vars)),"formula":formula})
        for band in config["bands"]:
            ww={split[0]:1};nw={split[1]:1}
            if band!="gt_24h":ww[f"{prefixes[0]}_x_{band}"]=1;nw[f"{prefixes[1]}_x_{band}"]=1
            marginals.append({"band":band,"flow_type":prefixes[0],**linear_result("H4_POOLED",f"{prefixes[0]}_{band}",ww,names,b,v,clusters)})
            marginals.append({"band":band,"flow_type":prefixes[1],**linear_result("H4_POOLED",f"{prefixes[1]}_{band}",nw,names,b,v,clusters)})
            diff=ww.copy()
            for term,value in nw.items():diff[term]=diff.get(term,0)-value
            marginals.append({"band":band,"flow_type":f"{prefixes[0]}_minus_{prefixes[1]}",**linear_result("H4_POOLED",f"difference_{band}",diff,names,b,v,clusters)})
        joint.append(joint_test("H4_POOLED",f"all_{prefixes[0]}_timing_interactions_zero",wx,names,b,v,clusters))
        joint.append(joint_test("H4_POOLED",f"all_{prefixes[1]}_timing_interactions_zero",nx,names,b,v,clusters))
        log["completed_models"].append("H4_POOLED");atomic_json(log_path,log)
        strat_formula=f"delta_yes_price ~ {split[0]} + {split[1]} + {CONTROLS} | market_id + calendar_hour_utc"
        for band in config["bands"]:
            model=f"H4_STRAT_{band}";subset=data.loc[data.time_band==band].copy();print(f"Estimating {model} on {len(subset):,} rows",flush=True)
            sf=pf.feols(strat_formula,data=subset,vcov={"CRV1":"event_id"},fixef_rm="singleton",copy_data=False,store_data=False,lean=True)
            frames.append(coef_frame(model,sf,int(subset.event_id.nunique())))
            diagnostics.append({"model":model,"source_rows":len(subset),"fitted_rows":int(sf._N),"markets":int(subset.market_id.nunique()),"events":int(subset.event_id.nunique()),"calendar_hours":int(subset.calendar_hour_utc.nunique()),"r_squared":float(sf._r2),"within_r_squared":float(sf._r2_within),"collinear_variables_removed":";".join(map(str,sf._collin_vars)),"formula":strat_formula})
            log["completed_models"].append(model);atomic_json(log_path,log)
        atomic_csv(out/"h4_timing_coefficients.csv",pd.concat(frames,ignore_index=True));atomic_csv(out/"h4_timing_marginal_effects.csv",pd.DataFrame(marginals));atomic_csv(out/"h4_timing_joint_tests.csv",pd.DataFrame(joint));atomic_csv(out/"h4_timing_model_diagnostics.csv",pd.DataFrame(diagnostics))
        log.update({"status":"complete_qc_passed","completed_at":now(),"outputs":{}})
        for name in ["h4_timing_coefficients.csv","h4_timing_marginal_effects.csv","h4_timing_joint_tests.csv","h4_timing_model_diagnostics.csv"]:
            path=out/name;log["outputs"][name]={"sha256":sha256(path),"bytes":path.stat().st_size}
        atomic_json(log_path,log);config["status"]="complete_qc_passed";config["completed_at"]=log["completed_at"];atomic_json(config_path,config);print(json.dumps(log,indent=2));return 0
    except Exception as e:
        log["status"]="failed";log["failed_at"]=now();log["errors"].append(repr(e));atomic_json(log_path,log);raise


if __name__=="__main__":raise SystemExit(main())
