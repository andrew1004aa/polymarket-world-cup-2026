#!/usr/bin/env python3
"""Build H2c samples conditional on an initial same-direction five-minute move."""
import csv,gzip,hashlib,json,math
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'analysis_ready/v4/h2a_prepost_flow/h2a_p99_prepost_events_all.csv';MARKETS=ROOT/'regression_ready/tables/markets.csv';MAP=ROOT/'regression_ready/tables/event_market_mapping.csv';OUT=ROOT/'model_samples/v4/h2c';OUT.mkdir(parents=True,exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while c:=f.read(1024*1024):h.update(c)
 return h.hexdigest()
types={}
with MARKETS.open(newline='') as f:
 for r in csv.DictReader(f):types[r['market_id']]=r['market_type']
kick={}
with MAP.open(newline='') as f:
 for r in csv.DictReader(f):kick[r['market_id']]=int(datetime.fromisoformat(r['actual_kickoff_utc'].replace('Z','+00:00')).timestamp())
fields=['initiating_trade_record_id','market_id','event_id','market_type','phase','initiating_timestamp','calendar_hour_utc','initiating_trade_value_usdc','baseline_yes_price','initial_directional_move_5m','follower_imbalance_5m','follower_signed_log_net_5m','initiating_minute_top1_wallet_share','initiating_minute_top3_wallet_share','initiating_minute_wallet_hhi','initiating_minute_absolute_flow_imbalance','initiating_wallet_opposite_trade_within_60m']
for tag in ('15m','30m','60m'):fields += [f'directional_move_{tag}',f'partial_reversal_{tag}',f'full_reversal_{tag}',f'reversal_fraction_{tag}']
target=OUT/'h2c_initial_positive_move.csv.gz';tmp=target.with_suffix(target.suffix+'.tmp');counts={tag:{'available':0,'partial_reversal':0,'full_reversal':0} for tag in ('15m','30m','60m')};rows=0;families={}
with SRC.open(newline='') as src,gzip.open(tmp,'wt',newline='',compresslevel=6) as dst:
 reader=csv.DictReader(src);writer=csv.DictWriter(dst,fieldnames=fields);writer.writeheader()
 for r in reader:
  if not r['directional_price_change_5m'] or float(r['directional_price_change_5m'])<=0:continue
  mid=r['market_id'];t=int(r['initiating_timestamp']);family=types[mid];phase='outright' if family=='outright' else ('pre' if t<kick[mid] else 'post');initial=float(r['directional_price_change_5m']);post_net=float(r['other_wallet_post_directional_net_5m']);signed=math.copysign(math.log1p(abs(post_net)),post_net) if post_net else 0.
  out={'initiating_trade_record_id':r['initiating_trade_record_id'],'market_id':mid,'event_id':r['event_id'],'market_type':family,'phase':phase,'initiating_timestamp':t,'calendar_hour_utc':datetime.fromtimestamp(t,timezone.utc).strftime('%Y-%m-%dT%H:00:00Z'),'initiating_trade_value_usdc':r['initiating_trade_value_usdc'],'baseline_yes_price':r['baseline_yes_price'],'initial_directional_move_5m':r['directional_price_change_5m'],'follower_imbalance_5m':r['other_wallet_post_directional_imbalance_5m'],'follower_signed_log_net_5m':format(signed,'.12g'),'initiating_minute_top1_wallet_share':r['initiating_minute_top1_wallet_share'],'initiating_minute_top3_wallet_share':r['initiating_minute_top3_wallet_share'],'initiating_minute_wallet_hhi':r['initiating_minute_wallet_hhi'],'initiating_minute_absolute_flow_imbalance':r['initiating_minute_absolute_flow_imbalance'],'initiating_wallet_opposite_trade_within_60m':r['initiating_wallet_opposite_trade_within_60m']}
  for tag in ('15m','30m','60m'):
   value=r[f'directional_price_change_{tag}']
   if value=='':out.update({f'directional_move_{tag}':'',f'partial_reversal_{tag}':'',f'full_reversal_{tag}':'',f'reversal_fraction_{tag}':''});continue
   later=float(value);partial=int(later<initial);full=int(later<0);fraction=(initial-later)/initial
   out.update({f'directional_move_{tag}':value,f'partial_reversal_{tag}':partial,f'full_reversal_{tag}':full,f'reversal_fraction_{tag}':format(fraction,'.12g')});counts[tag]['available']+=1;counts[tag]['partial_reversal']+=partial;counts[tag]['full_reversal']+=full
  writer.writerow(out);rows+=1;families[f'{family}|{phase}']=families.get(f'{family}|{phase}',0)+1
tmp.replace(target)
payload={'generated_at':datetime.now(timezone.utc).isoformat(),'status':'built_qc_passed','source':str(SRC.relative_to(ROOT)),'source_sha256':sha(SRC),'rows_conditional_initial_positive_5m_move':rows,'family_phase_rows':families,'outcome_counts':counts,'definitions':{'partial_reversal':'later cumulative directional move < positive five-minute directional move','full_reversal':'later cumulative directional move < 0','reversal_fraction':'(initial 5m move - later cumulative move) / initial 5m move'},'output':str(target.relative_to(ROOT)),'output_sha256':sha(target)}
(OUT/'summary.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload,indent=2))
