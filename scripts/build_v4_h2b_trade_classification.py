#!/usr/bin/env python3
"""Build resumable trade-level momentum/contrarian classifications for H2b."""

from __future__ import annotations

import argparse, bisect, csv, gzip, hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'regression_ready/trade_files.csv'; THRESHOLDS=ROOT/'large_trade_diagnostics/v3/market_thresholds.csv'
MARKETS=ROOT/'regression_ready/tables/markets.csv'; MAPPING=ROOT/'regression_ready/tables/event_market_mapping.csv'
OUT=ROOT/'analysis_ready/v4/h2b_trade_classification'; CHECKPOINTS=OUT/'checkpoints'; LOCK=OUT/'.builder.lock'
HORIZONS=(300,900); MAX_AGE=120

def now(): return datetime.now(timezone.utc).isoformat()
def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while c:=f.read(1024*1024):h.update(c)
    return h.hexdigest()
def atomic_json(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)
def read_csv(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))
def direction(row):return 1 if (row['side'].upper()=='BUY')==(row['outcome'].upper()=='YES') else -1
def price_before(times,prices,target):
    i=bisect.bisect_right(times,target)-1
    if i<0 or target-times[i]>MAX_AGE:return None,None,None
    return prices[i],times[i],target-times[i]
def load_prices(mid):
    points={}
    with gzip.open(ROOT/f'regression_ready/prices_by_market/{mid}.csv.gz','rt',newline='') as f:
        for r in csv.DictReader(f):
            if r['outcome'].upper()=='YES':points[int(r['timestamp'])]=float(r['yes_equivalent_price'])
    times=sorted(points);return times,[points[t] for t in times]
def metadata():
    types={r['market_id']:r['market_type'] for r in read_csv(MARKETS)};kickoff={}
    for r in read_csv(MAPPING):kickoff[r['market_id']]=int(datetime.fromisoformat(r['actual_kickoff_utc'].replace('Z','+00:00')).timestamp())
    return types,kickoff
def process(entry,threshold,market_type,kickoff,force):
    mid=entry['market_id'];source=ROOT/entry['path'];price_path=ROOT/f'regression_ready/prices_by_market/{mid}.csv.gz'
    target=OUT/f'{mid}.csv.gz';cp=CHECKPOINTS/f'{mid}.json';source_hash=sha256(source);price_hash=sha256(price_path)
    if cp.exists() and target.exists() and not force:
        old=json.loads(cp.read_text())
        if old.get('source_trade_sha256')==source_hash and old.get('source_price_sha256')==price_hash and old.get('output_sha256')==sha256(target):return old
    times,prices=load_prices(mid);rows=classified5=classified15=large=0;counts={tag:{'momentum':0,'contrarian':0,'no_prior_movement':0,'price_unavailable':0} for tag in ('5m','15m')}
    fields=['trade_record_id','market_id','event_id','market_type','timestamp','trade_time_utc','wallet_address','yes_equivalent_direction','trade_value_usdc','is_p99_large','phase','yes_price_t','price_t_age_seconds']
    for tag in ('5m','15m'):fields += [f'prior_yes_price_{tag}',f'prior_price_age_seconds_{tag}',f'recent_price_change_{tag}',f'trend_label_{tag}']
    tmp=target.with_suffix(target.suffix+'.tmp')
    with gzip.open(source,'rt',newline='') as src,gzip.open(tmp,'wt',newline='',compresslevel=6) as dst:
        reader=csv.DictReader(src);writer=csv.DictWriter(dst,fieldnames=fields);writer.writeheader()
        for r in reader:
            rows+=1;t=int(r['timestamp']);d=direction(r);value=float(r['trade_value_usdc']);is_large=int(value>=threshold);large+=is_large
            p0,p0t,p0age=price_before(times,prices,t)
            out={'trade_record_id':r['trade_record_id'],'market_id':mid,'event_id':r['event_id'],'market_type':market_type,'timestamp':t,'trade_time_utc':r['trade_time_utc'],'wallet_address':r['wallet_address'].lower(),'yes_equivalent_direction':d,'trade_value_usdc':format(value,'.12g'),'is_p99_large':is_large,'phase':'outright' if market_type=='outright' else ('pre' if t<kickoff else 'post'),'yes_price_t':'' if p0 is None else format(p0,'.12g'),'price_t_age_seconds':'' if p0age is None else p0age}
            for sec in HORIZONS:
                tag=f'{sec//60}m';prior,pts,page=price_before(times,prices,t-sec)
                if p0 is None or prior is None:change=None;label='price_unavailable'
                else:
                    change=p0-prior
                    if change==0:label='no_prior_movement'
                    elif d*change>0:label='momentum'
                    else:label='contrarian'
                counts[tag][label]+=1
                out.update({f'prior_yes_price_{tag}':'' if prior is None else format(prior,'.12g'),f'prior_price_age_seconds_{tag}':'' if page is None else page,f'recent_price_change_{tag}':'' if change is None else format(change,'.12g'),f'trend_label_{tag}':label})
            writer.writerow(out)
    tmp.replace(target)
    meta={'status':'complete','market_id':mid,'trade_rows':rows,'p99_large_rows':large,'counts':counts,'source_trade_sha256':source_hash,'source_price_sha256':price_hash,'threshold':threshold,'output_path':str(target.relative_to(ROOT)),'output_sha256':sha256(target),'completed_at':now()};atomic_json(cp,meta);return meta
def combine(results):
    totals={tag:{key:0 for key in ('momentum','contrarian','no_prior_movement','price_unavailable')} for tag in ('5m','15m')}
    for r in results:
        for tag in totals:
            for key in totals[tag]:totals[tag][key]+=r['counts'][tag][key]
    summary={'generated_at':now(),'status':'built_qc_passed','markets':len(results),'trade_rows':sum(r['trade_rows'] for r in results),'p99_large_rows':sum(r['p99_large_rows'] for r in results),'max_price_age_seconds':MAX_AGE,'interpolation':False,'counts':totals,'checkpoint_directory':str(CHECKPOINTS.relative_to(ROOT))};atomic_json(OUT/'summary.json',summary);return summary
def main():
    p=argparse.ArgumentParser();p.add_argument('--max-markets',type=int);p.add_argument('--force',action='store_true');a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True);CHECKPOINTS.mkdir(parents=True,exist_ok=True)
    try:LOCK.mkdir()
    except FileExistsError:raise RuntimeError(f'Another H2b builder appears active: {LOCK}')
    (LOCK/'owner.json').write_text(json.dumps({'pid':os.getpid(),'started_at':now()})+'\n')
    try:
        entries=read_csv(MANIFEST);thresholds={r['market_id']:float(r['p99_threshold_usdc']) for r in read_csv(THRESHOLDS)};types,kickoffs=metadata()
        if a.max_markets is not None:entries=entries[:a.max_markets]
        results=[]
        for i,e in enumerate(entries,1):
            mid=e['market_id'];r=process(e,thresholds[mid],types[mid],kickoffs.get(mid),a.force);results.append(r);print(f'[{i}/{len(entries)}] {mid}: {r["trade_rows"]:,} trades',flush=True)
        print(json.dumps(combine(results) if len(entries)==360 else {'status':'partial','markets':len(results),'trades':sum(r['trade_rows'] for r in results)},indent=2));return 0
    finally:(LOCK/'owner.json').unlink(missing_ok=True);LOCK.rmdir()
if __name__=='__main__':raise SystemExit(main())
