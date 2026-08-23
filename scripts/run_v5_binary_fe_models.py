#!/usr/bin/env python3
"""Fixed-effects logit and cloglog robustness for five-minute price updating."""
from __future__ import annotations

import json, math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.sparse.linalg import splu
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"regression_inputs/v5/strict_timing/strict_timing_past_only_p99.csv.gz"
OUT=ROOT/"regression_results/v5/binary_fe"


def inverse_link(eta,link):
    eta=np.clip(eta,-25,15)
    if link=="logit":
        mu=expit(eta); derivative=mu*(1-mu)
    else:
        ex=np.exp(eta); mu=-np.expm1(-ex); derivative=np.exp(eta-ex)
    mu=np.clip(mu,1e-9,1-1e-9); derivative=np.maximum(derivative,1e-12)
    return mu,derivative


def fit(data,terms,link,model):
    y=data.any_price_update_5m.to_numpy(float); raw=data[terms].to_numpy(float)
    means=raw.mean(axis=0); scales=raw.std(axis=0); scales[scales==0]=1
    exog=(raw-means)/scales; n=len(data); kx=len(terms)
    market=pd.Categorical(data.market_id).codes
    nm=market.max()+1
    d_market=sparse.csr_matrix((np.ones(n),(np.arange(n),market)),shape=(n,nm))
    X=sparse.hstack([sparse.csr_matrix(exog),d_market],format="csr")
    beta=np.zeros(X.shape[1]); p=float(y.mean())
    start=math.log(p/(1-p)) if link=="logit" else math.log(-math.log1p(-p))
    beta[kx:kx+nm]=start
    converged=False
    mu,_=inverse_link(np.asarray(X@beta),link)
    old_ll=float(np.sum(y*np.log(mu)+(1-y)*np.log1p(-mu)))
    for iteration in range(1,51):
        eta=np.asarray(X@beta); mu,derivative=inverse_link(eta,link)
        variance=mu*(1-mu); w=np.maximum(derivative*derivative/variance,1e-10)
        z=eta+(y-mu)/derivative
        normal=(X.T@sparse.diags(w)@X).tocsc(); rhs=np.asarray(X.T@(w*z)).ravel()
        candidate=splu(normal).solve(rhs)
        step=1.0
        while step>=2**-20:
            new=beta+step*(candidate-beta)
            trial_mu,_=inverse_link(np.asarray(X@new),link)
            new_ll=float(np.sum(y*np.log(trial_mu)+(1-y)*np.log1p(-trial_mu)))
            if new_ll>=old_ll-1e-8: break
            step/=2
        change=float(np.max(np.abs(new-beta))); improvement=new_ll-old_ll; beta=new;old_ll=new_ll
        print(f"  {model} iter={iteration} ll_gain={improvement:.6g} max_change={change:.3g} step={step:.3g}",flush=True)
        if change<1e-7 or abs(improvement)<1e-7: converged=True;break
    if not converged:
        raise RuntimeError(f"{model} did not converge; no estimates written")
    eta=np.asarray(X@beta); mu,derivative=inverse_link(eta,link); variance=mu*(1-mu)
    w=np.maximum(derivative*derivative/variance,1e-10)
    D=d_market
    dd=splu((D.T@sparse.diags(w)@D).tocsc())
    fitted=D@dd.solve(np.asarray(D.T@(w[:,None]*exog)))
    xtilde=exog-fitted
    bread=np.linalg.pinv(xtilde.T@(w[:,None]*xtilde))
    score_factor=(y-mu)*derivative/variance
    groups=pd.Categorical(data.event_id).codes; g=groups.max()+1
    scores=np.zeros((g,kx));np.add.at(scores,groups,xtilde*score_factor[:,None])
    cov_scaled=(g/(g-1))*((n-1)/(n-kx))*bread@(scores.T@scores)@bread
    transform=np.diag(1/scales); b=beta[:kx]/scales;cov=transform@cov_scaled@transform
    df=g-1;critical=stats.t.ppf(.975,df);rows=[]
    for j,term in enumerate(terms):
        se=math.sqrt(cov[j,j]);t=b[j]/se
        rows.append({"model":model,"link":link,"term":term,"estimate":b[j],"std_error":se,
                     "t_statistic":t,"cluster_df":df,"p_value":2*stats.t.sf(abs(t),df),
                     "conf_low":b[j]-critical*se,"conf_high":b[j]+critical*se})
    v=np.zeros(kx);v[0]=1;v[1]=-1;est=float(v@b);se=math.sqrt(float(v@cov@v));t=est/se
    average_derivative=float(derivative.mean())
    contrast={"model":model,"link":link,"contrast":"large_minus_ordinary","estimate_link_scale":est,
              "std_error":se,"t_statistic":t,"cluster_df":df,"p_value":2*stats.t.sf(abs(t),df),
              "conf_low":est-critical*se,"conf_high":est+critical*se,
              "average_marginal_effect_difference":average_derivative*est}
    diag={"model":model,"rows":n,"markets":data.market_id.nunique(),"events":g,
          "outcome_rate":p,"iterations":iteration,
          "converged":converged,"log_likelihood":float(np.sum(y*np.log(mu)+(1-y)*np.log1p(-mu)))}
    return rows,contrast,diag


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cols=["market_id","event_id","v5_delta_5m","v5_strict0_5m","v5_baseline_price",
          "v5_lagged_30m_price_change","log_lagged_30m_volume","log_minutes_to_kickoff"]
    for s in ("expanding","rolling7"):
        cols += [f"{s}_eligible",f"{s}_large_net_flow_usdc",f"{s}_ordinary_net_flow_usdc"]
    data=pd.read_csv(SOURCE,usecols=cols);data["any_price_update_5m"]=data.v5_delta_5m.ne(0).astype(float)
    data["absolute_v5_lagged_30m_price_change"]=data.v5_lagged_30m_price_change.abs()
    coefficients=[];contrasts=[];diagnostics=[]
    for spec in ("expanding","rolling7"):
        d=data[data.v5_strict0_5m.eq(1)&data[f"{spec}_eligible"].eq(1)].dropna().copy()
        # Markets with no outcome variation have an infinite binary-response
        # fixed effect and carry no slope information.
        original_rows=len(d)
        keep=d.groupby("market_id").any_price_update_5m.transform("nunique").gt(1)
        d=d.loc[keep].copy()
        d[f"{spec}_log_abs_large"]=np.log1p(d[f"{spec}_large_net_flow_usdc"].abs())
        d[f"{spec}_log_abs_ordinary"]=np.log1p(d[f"{spec}_ordinary_net_flow_usdc"].abs())
        terms=[f"{spec}_log_abs_large",f"{spec}_log_abs_ordinary","absolute_v5_lagged_30m_price_change",
               "v5_baseline_price","log_lagged_30m_volume","log_minutes_to_kickoff"]
        for link in ("logit",):
            model=f"{spec.upper()}_{link.upper()}"
            print(f"Estimating {model}: {len(d):,}",flush=True)
            co,con,diag=fit(d,terms,link,model);diag["unconditional_rows"]=original_rows
            diag["conditional_fe_rows"]=len(d);coefficients+=co;contrasts.append(con);diagnostics.append(diag)
    pd.DataFrame(coefficients).to_csv(OUT/"coefficients.csv",index=False)
    pd.DataFrame(contrasts).to_csv(OUT/"contrasts.csv",index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/"diagnostics.csv",index=False)
    summary={"generated_at":datetime.now(timezone.utc).isoformat(),"status":"complete_qc_passed",
             "links":["logit"],"fixed_effects":["market_id"],
             "cluster":"event_id CRV1","outcome":"any non-zero strictly aligned five-minute price change",
             "failed_not_reported":"cloglog did not produce a finite stable solution under richer or market-only FE; no cloglog estimates retained"}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(pd.DataFrame(contrasts).to_string(index=False),flush=True)


if __name__=="__main__":main()
