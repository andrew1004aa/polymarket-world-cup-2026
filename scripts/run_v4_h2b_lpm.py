#!/usr/bin/env python3
"""Estimate interpretable H2b LPM baselines with fixed effects and clustered SEs."""
import argparse,hashlib,json,math,platform
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd,pyfixest as pf,scipy

ROOT=Path(__file__).resolve().parents[1];IN=ROOT/'model_samples/v4/h2b';OUT=ROOT/'regression_results/v4/h2b'
COLS=['trade_record_id','market_id','event_id','market_type','timestamp','phase','trade_value_usdc','is_p99_large','yes_price_t','recent_price_change','momentum']
def now():return datetime.now(timezone.utc).isoformat()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  while c:=f.read(1024*1024):h.update(c)
 return h.hexdigest()
def tidy(fit):
 return fit.tidy().reset_index().rename(columns={'Coefficient':'term','Estimate':'estimate','Std. Error':'std_error','t value':'t_statistic','Pr(>|t|)':'p_value','2.5%':'conf_low','97.5%':'conf_high'})
def main():
 p=argparse.ArgumentParser();p.add_argument('--horizon',choices=['5m','15m'],default='5m');p.add_argument('--sample',choices=['match_pre','match_post','outright'],default='match_pre');a=p.parse_args()
 source=IN/f'h2b_{a.horizon}_binary.csv.gz';out=OUT/f'{a.sample}_{a.horizon}';out.mkdir(parents=True,exist_ok=True)
 data=pd.read_csv(source,usecols=COLS)
 if a.sample=='match_pre':data=data[(data.market_type=='match')&(data.phase=='pre')].copy();cluster='event_id'
 elif a.sample=='match_post':data=data[(data.market_type=='match')&(data.phase=='post')].copy();cluster='event_id'
 else:data=data[data.market_type=='outright'].copy();cluster='market_id'
 if data.empty:raise RuntimeError('Empty sample')
 data['calendar_hour_utc']=pd.to_datetime(data.timestamp,unit='s',utc=True).dt.floor('h').astype(str)
 data['log_trade_value']=np.log1p(data.trade_value_usdc);data['abs_recent_price_change']=data.recent_price_change.abs()
 for col in ['market_id','calendar_hour_utc',cluster]:data[col]=data[col].astype('category')
 formula='momentum ~ is_p99_large + log_trade_value + abs_recent_price_change + yes_price_t | market_id + calendar_hour_utc'
 config={'version':'v4','hypothesis':'H2b','sample':a.sample,'horizon':a.horizon,'status':'running','source':str(source.relative_to(ROOT)),'source_sha256':sha(source),'rows':len(data),'formula':formula,'cluster':cluster,'interpretation':'Linear-probability conditional association; the sample excludes zero prior price changes and unavailable prices.'};(out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
 print(f'Estimating {a.sample} {a.horizon} on {len(data):,} rows',flush=True)
 fit=pf.feols(formula,data=data,vcov={'CRV1':cluster},copy_data=False,store_data=False,lean=True)
 coef=tidy(fit);coef.to_csv(out/'coefficients.csv',index=False)
 diag={'observations':int(fit._N),'markets':int(data.market_id.nunique()),'clusters':int(data[cluster].nunique()),'calendar_hours':int(data.calendar_hour_utc.nunique()),'momentum_share':float(data.momentum.mean()),'large_share':float(data.is_p99_large.mean()),'r_squared':float(fit._r2),'within_r_squared':float(fit._r2_within),'software':{'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__,'scipy':scipy.__version__,'pyfixest':pf.__version__}}
 config.update({'status':'complete_qc_passed','completed_at':now(),'diagnostics':diag});(out/'config.json').write_text(json.dumps(config,indent=2)+'\n');print(json.dumps(config,indent=2));print(coef.to_string(index=False));return 0
if __name__=='__main__':raise SystemExit(main())
