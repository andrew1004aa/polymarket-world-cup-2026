#!/usr/bin/env python3
"""Estimate H2c reversal LPMs conditional on an initial positive move."""
import argparse,json,platform
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd,pyfixest as pf,scipy
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'model_samples/v4/h2c/h2c_initial_positive_move.csv.gz';OUT=ROOT/'regression_results/v4/h2c'
COLS=['market_id','event_id','market_type','phase','calendar_hour_utc','initiating_trade_value_usdc','baseline_yes_price','initial_directional_move_5m','follower_imbalance_5m','follower_signed_log_net_5m','initiating_minute_wallet_hhi','initiating_wallet_opposite_trade_within_60m']
def tidy(fit,model):
 x=fit.tidy().reset_index().rename(columns={'Coefficient':'term','Estimate':'estimate','Std. Error':'std_error','t value':'t_statistic','Pr(>|t|)':'p_value','2.5%':'conf_low','97.5%':'conf_high'});x.insert(0,'model',model);return x
def main():
 p=argparse.ArgumentParser();p.add_argument('--sample',choices=['match_pre','match_post','outright'],default='match_pre');p.add_argument('--measure',choices=['joint','imbalance','signed_log'],default='joint');a=p.parse_args();data=pd.read_csv(SRC)
 if a.sample=='match_pre':data=data[(data.market_type=='match')&(data.phase=='pre')].copy();cluster='event_id'
 elif a.sample=='match_post':data=data[(data.market_type=='match')&(data.phase=='post')].copy();cluster='event_id'
 else:data=data[data.market_type=='outright'].copy();cluster='market_id'
 data['log_initiating_trade_value']=np.log1p(data.initiating_trade_value_usdc)
 for c in ['market_id','calendar_hour_utc',cluster]:data[c]=data[c].astype('category')
 out=OUT/a.sample/a.measure;out.mkdir(parents=True,exist_ok=True);frames=[];diags=[]
 follower_terms={'joint':'follower_imbalance_5m + follower_signed_log_net_5m','imbalance':'follower_imbalance_5m','signed_log':'follower_signed_log_net_5m'}[a.measure]
 for horizon in ('15m','30m','60m'):
  for kind in ('partial_reversal','full_reversal'):
   outcome=f'{kind}_{horizon}';cols=COLS+[outcome];sample=data.dropna(subset=[outcome,'follower_imbalance_5m','baseline_yes_price','initial_directional_move_5m','initiating_minute_wallet_hhi','log_initiating_trade_value']).copy()
   formula=f'{outcome} ~ {follower_terms} + initial_directional_move_5m + log_initiating_trade_value + initiating_minute_wallet_hhi + baseline_yes_price | market_id + calendar_hour_utc'
   print(f'Estimating {a.sample} {a.measure} {outcome} on {len(sample):,}',flush=True);fit=pf.feols(formula,data=sample,vcov={'CRV1':cluster},copy_data=False,store_data=False,lean=True);frames.append(tidy(fit,outcome));diags.append({'model':outcome,'rows_input':len(sample),'observations':int(fit._N),'outcome_share':float(sample[outcome].mean()),'markets':int(sample.market_id.nunique()),'clusters':int(sample[cluster].nunique()),'formula':formula})
 pd.concat(frames,ignore_index=True).to_csv(out/'coefficients.csv',index=False);payload={'generated_at':datetime.now(timezone.utc).isoformat(),'status':'complete_qc_passed','sample':a.sample,'measure':a.measure,'cluster':cluster,'diagnostics':diags,'software':{'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__,'scipy':scipy.__version__,'pyfixest':pf.__version__},'interpretation':'Conditional LPM associations. Follower flow and the initial price move share the first five-minute window; controlling for the initial move does not establish causality.'};(out/'summary.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
