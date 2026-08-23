#!/usr/bin/env python3
"""Estimate V5 expanding and rolling past-only P99 H2 models."""

from __future__ import annotations

import json, math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.sparse.linalg import lsmr, splu

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "regression_inputs/v5/past_only_p99/past_only_p99_primary_sample.csv.gz"
OUT = ROOT / "regression_results/v5/past_only_p99_h2"
CONTROLS_DIRECTION = ["lagged_30m_price_change", "yes_price_t", "log_lagged_30m_volume", "log_minutes_to_kickoff"]
CONTROLS_UPDATE = ["absolute_lagged_30m_price_change", "yes_price_t", "log_lagged_30m_volume", "log_minutes_to_kickoff"]


def now(): return datetime.now(timezone.utc).isoformat()


def residualize(data, columns):
    market = pd.Categorical(data.market_id).codes
    hour = pd.Categorical(data.calendar_hour_utc).codes
    n = len(data); nm = int(market.max()) + 1; nh = int(hour.max()) + 1
    rows=[np.arange(n)]; cols=[market]; values=[np.ones(n)]
    nonreference=hour>0
    rows.append(np.flatnonzero(nonreference)); cols.append(nm+hour[nonreference]-1); values.append(np.ones(nonreference.sum()))
    design=sparse.csr_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(cols))),shape=(n,nm+nh-1))
    matrix=data[columns].to_numpy(float)
    try:
        solver=splu((design.T@design).tocsc())
        result=matrix-design@solver.solve(np.asarray(design.T@matrix))
    except RuntimeError:
        # Outcome-selected samples can yield disconnected FE components. LSMR
        # returns the minimum-norm projection without requiring a full-rank
        # normal-equation matrix.
        result=np.empty_like(matrix)
        for column in range(matrix.shape[1]):
            fitted=design@lsmr(design,matrix[:,column],atol=1e-10,btol=1e-10,maxiter=10000)[0]
            result[:,column]=matrix[:,column]-fitted
    return {name: result[:, i] for i, name in enumerate(columns)}


def fit(data, model, outcome, terms):
    columns = list(dict.fromkeys([outcome] + terms)); residual = residualize(data, columns)
    y = residual[outcome]; x = np.column_stack([residual[t] for t in terms])
    keep = np.max(np.abs(x), axis=1) > 1e-14; y = y[keep]; x = x[keep]
    groups = pd.Categorical(data.event_id.to_numpy()[keep]).codes; clusters = int(groups.max()) + 1
    inv = np.linalg.pinv(x.T @ x); beta = inv @ (x.T @ y); error = y - x @ beta
    scores = np.zeros((clusters, len(terms))); np.add.at(scores, groups, x * error[:, None])
    n, k = x.shape; covariance = (clusters/(clusters-1))*((n-1)/(n-k))*inv@(scores.T@scores)@inv
    se = np.sqrt(np.diag(covariance)); df = clusters - 1; critical = stats.t.ppf(.975, df)
    coefficients = []
    for i, term in enumerate(terms):
        statistic = beta[i]/se[i]
        coefficients.append({"model": model, "outcome": outcome, "term": term, "estimate": beta[i],
            "std_error": se[i], "t_statistic": statistic, "cluster_df": df,
            "p_value": 2*stats.t.sf(abs(statistic),df), "conf_low": beta[i]-critical*se[i],
            "conf_high": beta[i]+critical*se[i]})
    diagnostic = {"model": model, "outcome": outcome, "rows": len(y), "markets": data.market_id.nunique(),
        "events": clusters, "calendar_hours": data.calendar_hour_utc.nunique(),
        "within_r_squared": 1-float(error@error)/float(y@y), "condition_number": np.linalg.cond(x)}
    return coefficients, diagnostic, beta, covariance, terms, clusters


def contrast(model, outcome, label, weights, beta, covariance, names, clusters):
    lookup = {name:i for i,name in enumerate(names)}; vector = np.zeros(len(names))
    for term, weight in weights.items(): vector[lookup[term]] = weight
    estimate = float(vector@beta); se = math.sqrt(float(vector@covariance@vector)); df=clusters-1
    statistic=estimate/se; critical=stats.t.ppf(.975,df)
    return {"model":model,"outcome":outcome,"contrast":label,"estimate":estimate,"std_error":se,
        "t_statistic":statistic,"cluster_df":df,"p_value":2*stats.t.sf(abs(statistic),df),
        "conf_low":estimate-critical*se,"conf_high":estimate+critical*se,
        "weights":json.dumps(weights,sort_keys=True)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    columns = ["market_id","event_id","calendar_hour_utc","delta_yes_price","yes_price_t",
        "lagged_30m_price_change","log_lagged_30m_volume","log_minutes_to_kickoff","common_eligible"]
    for spec in ("expanding","rolling7"):
        columns += [f"{spec}_eligible",f"{spec}_large_net_flow_usdc",f"{spec}_ordinary_net_flow_usdc",
                    f"{spec}_signed_log_large_net_flow",f"{spec}_signed_log_ordinary_net_flow"]
    data = pd.read_csv(INPUT,usecols=columns)
    data["any_price_update_5m"] = data.delta_yes_price.ne(0).astype(float)
    data["absolute_lagged_30m_price_change"] = data.lagged_30m_price_change.abs()
    coefficients=[]; contrasts=[]; diagnostics=[]; effects=[]
    samples=[("expanding","separate",data.expanding_eligible.eq(1)),
             ("rolling7","separate",data.rolling7_eligible.eq(1)),
             ("expanding","common",data.common_eligible.eq(1)),
             ("rolling7","common",data.common_eligible.eq(1))]
    for spec,support,mask in samples:
        sample=data.loc[mask].copy()
        large_signed=f"{spec}_signed_log_large_net_flow"; ordinary_signed=f"{spec}_signed_log_ordinary_net_flow"
        large_abs=f"{spec}_log_abs_large_net_flow"; ordinary_abs=f"{spec}_log_abs_ordinary_net_flow"
        sample[large_abs]=np.log1p(sample[f"{spec}_large_net_flow_usdc"].abs())
        sample[ordinary_abs]=np.log1p(sample[f"{spec}_ordinary_net_flow_usdc"].abs())
        model_specs=[("directional", "delta_yes_price", [large_signed,ordinary_signed]+CONTROLS_DIRECTION, large_signed,ordinary_signed),
                     ("update_lpm","any_price_update_5m",[large_abs,ordinary_abs]+CONTROLS_UPDATE,large_abs,ordinary_abs),
                     ("conditional_nonzero","delta_yes_price",[large_signed,ordinary_signed]+CONTROLS_DIRECTION,large_signed,ordinary_signed)]
        for kind,outcome,terms,large,ordinary in model_specs:
            model_sample=sample if kind!="conditional_nonzero" else sample[sample.delta_yes_price.ne(0)]
            model=f"{spec.upper()}_{support.upper()}_{kind.upper()}"; print(f"Estimating {model}: {len(model_sample):,}",flush=True)
            c,d,beta,cov,names,clusters=fit(model_sample,model,outcome,terms)
            coefficients+=c; diagnostics.append(d)
            contrasts.append(contrast(model,outcome,"large_minus_ordinary",{large:1,ordinary:-1},beta,cov,names,clusters))
            for term in (large,ordinary):
                values=model_sample[term]; sd=float(values.std()); iqr=float(values.quantile(.75)-values.quantile(.25))
                estimate=float(beta[names.index(term)])
                effects.append({"model":model,"term":term,"coefficient":estimate,"predictor_sd":sd,
                    "predictor_iqr":iqr,"one_sd_outcome_change":estimate*sd,"iqr_outcome_change":estimate*iqr})
    pd.DataFrame(coefficients).to_csv(OUT/"coefficients.csv",index=False)
    pd.DataFrame(contrasts).to_csv(OUT/"contrasts.csv",index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/"model_diagnostics.csv",index=False)
    pd.DataFrame(effects).to_csv(OUT/"standardised_effects.csv",index=False)
    summary={"generated_at":now(),"status":"complete_qc_passed","models":len(diagnostics),
        "specifications":["expanding","rolling7"],"support":["separate","common"],
        "fixed_effects":["market_id","calendar_hour_utc"],"cluster":"event_id CRV1",
        "conditional_nonzero_interpretation":"descriptive outcome-selected model"}
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(pd.DataFrame(contrasts).to_string(index=False),flush=True)


if __name__=="__main__": raise SystemExit(main())
