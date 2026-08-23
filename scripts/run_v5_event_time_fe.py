#!/usr/bin/env python3
"""Estimate V5 directional H2 with event-by-time fixed effects."""
from __future__ import annotations

import json, math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"regression_inputs/v5/strict_timing/strict_timing_past_only_p99.csv.gz"
OUT=ROOT/"regression_results/v5/event_time_fe"


def residualize(values, a, b, tol=1e-10, maxiter=1000):
    z=values.astype(float).copy(); na=int(a.max())+1; nb=int(b.max())+1
    ca=np.bincount(a,minlength=na).astype(float); cb=np.bincount(b,minlength=nb).astype(float)
    for iteration in range(maxiter):
        old=z.copy()
        for j in range(z.shape[1]): z[:,j]-=np.bincount(a,weights=z[:,j],minlength=na)[a]/ca[a]
        for j in range(z.shape[1]): z[:,j]-=np.bincount(b,weights=z[:,j],minlength=nb)[b]/cb[b]
        if np.max(np.abs(z-old))<tol: return z,iteration+1
    raise RuntimeError("FE alternating projections did not converge")


def estimate(data,model,outcome,terms,cell):
    a=pd.Categorical(data.market_id).codes; b=pd.Categorical(data[cell]).codes
    matrix=data[[outcome]+terms].to_numpy(float); r,it=residualize(matrix,a,b)
    y=r[:,0]; x=r[:,1:]; keep=np.max(np.abs(x),axis=1)>1e-14
    y=y[keep]; x=x[keep]; groups=pd.Categorical(data.event_id.to_numpy()[keep]).codes
    g=int(groups.max())+1; inv=np.linalg.pinv(x.T@x); beta=inv@(x.T@y); error=y-x@beta
    scores=np.zeros((g,len(terms))); np.add.at(scores,groups,x*error[:,None])
    n,k=x.shape; cov=(g/(g-1))*((n-1)/(n-k))*inv@(scores.T@scores)@inv
    rows=[]; df=g-1; critical=stats.t.ppf(.975,df)
    for j,term in enumerate(terms):
        se=math.sqrt(cov[j,j]); t=beta[j]/se
        rows.append({"model":model,"term":term,"estimate":beta[j],"std_error":se,"t_statistic":t,
                     "cluster_df":df,"p_value":2*stats.t.sf(abs(t),df),"conf_low":beta[j]-critical*se,"conf_high":beta[j]+critical*se})
    v=np.zeros(len(terms)); v[0]=1;v[1]=-1; est=float(v@beta);se=math.sqrt(float(v@cov@v));t=est/se
    contrast={"model":model,"estimate":est,"std_error":se,"t_statistic":t,"cluster_df":df,
              "p_value":2*stats.t.sf(abs(t),df),"conf_low":est-critical*se,"conf_high":est+critical*se}
    diag={"model":model,"input_rows":len(data),"identified_rows":len(y),"markets":data.market_id.nunique(),
          "events":g,"time_cells":data[cell].nunique(),"projection_iterations":it}
    return rows,contrast,diag


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cols=["market_id","event_id","minute_start_timestamp","v5_delta_5m","v5_strict0_5m",
          "v5_baseline_price","v5_lagged_30m_price_change","log_lagged_30m_volume","log_minutes_to_kickoff"]
    for s in ("expanding","rolling7"):cols += [f"{s}_eligible",f"{s}_signed_log_large_net_flow",f"{s}_signed_log_ordinary_net_flow"]
    data=pd.read_csv(SOURCE,usecols=cols)
    data["event_minute"]=data.event_id.astype(str)+":"+data.minute_start_timestamp.astype(str)
    data["event_5minute"]=data.event_id.astype(str)+":"+(data.minute_start_timestamp//300).astype(str)
    coefficients=[];contrasts=[];diagnostics=[]
    for spec in ("expanding","rolling7"):
        for cell in ("event_minute","event_5minute"):
            mask=data.v5_strict0_5m.eq(1)&data[f"{spec}_eligible"].eq(1)
            d=data.loc[mask].dropna().copy()
            support=d.groupby(cell).market_id.transform("nunique")
            d=d.loc[support.ge(2)]
            terms=[f"{spec}_signed_log_large_net_flow",f"{spec}_signed_log_ordinary_net_flow",
                   "v5_lagged_30m_price_change","v5_baseline_price","log_lagged_30m_volume","log_minutes_to_kickoff"]
            model=f"{spec.upper()}_{cell.upper()}"
            print(f"Estimating {model}: {len(d):,}",flush=True)
            co,con,diag=estimate(d,model,"v5_delta_5m",terms,cell)
            coefficients+=co;contrasts.append(con);diagnostics.append(diag)
    pd.DataFrame(coefficients).to_csv(OUT/"coefficients.csv",index=False)
    pd.DataFrame(contrasts).to_csv(OUT/"contrasts.csv",index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/"diagnostics.csv",index=False)
    summary={"generated_at":datetime.now(timezone.utc).isoformat(),"status":"complete_qc_passed",
             "fixed_effects":["market + event×minute","market + event×5-minute"],"cluster":"event_id CRV1",
             "single_market_cells":"excluded because they provide no within-cell cross-contract identification"}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(pd.DataFrame(contrasts).to_string(index=False),flush=True)


if __name__=="__main__":main()
