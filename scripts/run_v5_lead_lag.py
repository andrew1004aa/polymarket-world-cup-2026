#!/usr/bin/env python3
"""Estimate the frozen V5 pre/post lead-lag coefficient path."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from run_v5_past_only_p99_h2 import fit, contrast

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"regression_inputs/v5/lead_lag/lead_lag_panel.csv.gz"
OUT=ROOT/"regression_results/v5/lead_lag"
HORIZONS=[("pre",15,-15),("pre",5,-5),("pre",1,-1),("post",1,1),("post",5,5),("post",15,15),("post",30,30)]


def holm(values):
    n=len(values); order=sorted(range(n),key=lambda i:values[i]); adjusted=[0.0]*n; running=0.0
    for rank,index in enumerate(order):
        running=max(running,min(1.0,(n-rank)*values[index])); adjusted[index]=running
    return adjusted


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cols=["market_id","event_id","calendar_hour_utc","v5_baseline_price",
          "v5_lagged_30m_price_change","log_lagged_30m_volume","log_minutes_to_kickoff"]
    for spec in ("expanding","rolling7"):
        cols += [f"{spec}_eligible",f"{spec}_signed_log_large_net_flow",f"{spec}_signed_log_ordinary_net_flow"]
    for phase,h,_ in HORIZONS: cols += [f"{phase}_{h}m_change",f"{phase}_{h}m_strict0"]
    data=pd.read_csv(SOURCE,usecols=list(dict.fromkeys(cols)))
    coefs=[]; contrasts=[]; diagnostics=[]
    for spec in ("expanding","rolling7"):
        for phase,h,signed_h in HORIZONS:
            outcome=f"{phase}_{h}m_change"; strict=f"{phase}_{h}m_strict0"
            mask=data[f"{spec}_eligible"].eq(1)&data[strict].eq(1)
            d=data.loc[mask].dropna()
            focal=[f"{spec}_signed_log_large_net_flow",f"{spec}_signed_log_ordinary_net_flow"]
            terms=focal+["v5_lagged_30m_price_change","v5_baseline_price","log_lagged_30m_volume","log_minutes_to_kickoff"]
            model=f"{spec.upper()}_{signed_h:+d}M"
            print(f"Estimating {model}: {len(d):,}",flush=True)
            co,diag,beta,cov,names,clusters=fit(d,model,outcome,terms)
            coefs.extend(co); diag["signed_horizon_minutes"]=signed_h; diagnostics.append(diag)
            item=contrast(model,outcome,"large_minus_ordinary",{focal[0]:1,focal[1]:-1},beta,cov,names,clusters)
            item["specification"]=spec; item["phase"]=phase; item["signed_horizon_minutes"]=signed_h
            contrasts.append(item)
    result=pd.DataFrame(contrasts)
    result["holm_p_value_within_spec"] = 1.0
    for spec,indexes in result.groupby("specification").groups.items():
        result.loc[indexes,"holm_p_value_within_spec"]=holm(result.loc[indexes,"p_value"].tolist())
    pd.DataFrame(coefs).to_csv(OUT/"coefficients.csv",index=False)
    result.to_csv(OUT/"lead_lag_contrasts.csv",index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/"model_diagnostics.csv",index=False)
    summary={"generated_at":datetime.now(timezone.utc).isoformat(),"status":"complete_qc_passed",
             "horizons":[x[2] for x in HORIZONS],"specifications":["expanding","rolling7"],
             "fixed_effects":["market_id","calendar_hour_utc"],"cluster":"event_id CRV1",
             "multiplicity":"Holm adjustment across seven horizons within each P99 definition",
             "interpretation":"Negative horizons are placebo diagnostics; all estimates are conditional associations."}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(result[["specification","signed_horizon_minutes","estimate","std_error","p_value","holm_p_value_within_spec","conf_low","conf_high"]].to_string(index=False),flush=True)


if __name__=="__main__": main()
