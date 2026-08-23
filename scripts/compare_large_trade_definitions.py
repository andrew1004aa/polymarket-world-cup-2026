#!/usr/bin/env python3
"""Compare within-market P99 with trade value >= 1% of market total volume."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'regression_ready/trade_files.csv'
THRESHOLDS=ROOT/'large_trade_diagnostics/v3/market_thresholds.csv'
OUT=ROOT/'large_trade_diagnostics/v3/definition_comparison'
START=1780272000; END=1785542400

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(1024*1024):h.update(block)
    return h.hexdigest()

def main()->int:
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=pd.read_csv(MANIFEST)
    thresholds=pd.read_csv(THRESHOLDS)
    source=manifest.merge(thresholds[['market_id','market_type','question','p99_threshold_usdc','total_value_usdc']],on='market_id',validate='one_to_one')
    rows=[]
    for index,r in enumerate(source.itertuples(),1):
        data=pd.read_csv(ROOT/r.path,usecols=['timestamp','trade_value_usdc'])
        values=data.loc[(data.timestamp>=START)&(data.timestamp<END),'trade_value_usdc'].to_numpy(float)
        if len(values)==0 or not np.isfinite(values).all():raise RuntimeError(f'Invalid market {r.market_id}')
        total=float(values.sum()); extreme_threshold=.01*total
        if not np.isclose(total,float(r.total_value_usdc),rtol=1e-10,atol=1e-6):raise RuntimeError(f'Volume mismatch {r.market_id}')
        p99=values>=float(r.p99_threshold_usdc); extreme=values>=extreme_threshold
        both=p99&extreme
        rows.append({'market_id':int(r.market_id),'market_type':r.market_type,'question':r.question,
                     'trade_count':len(values),'total_value_usdc':total,
                     'p99_threshold_usdc':float(r.p99_threshold_usdc),'p99_trade_count':int(p99.sum()),
                     'p99_trade_share':float(p99.mean()),'p99_value_usdc':float(values[p99].sum()),
                     'extreme_1pct_volume_threshold_usdc':extreme_threshold,
                     'extreme_trade_count':int(extreme.sum()),'extreme_trade_share':float(extreme.mean()),
                     'extreme_value_usdc':float(values[extreme].sum()),
                     'overlap_trade_count':int(both.sum()),
                     'extreme_not_p99_count':int((extreme&~p99).sum()),
                     'p99_not_extreme_count':int((p99&~extreme).sum())})
        if index%50==0 or index==len(source):print(f'{index}/{len(source)} markets compared',flush=True)
    frame=pd.DataFrame(rows).sort_values(['market_type','market_id'])
    target=OUT/'market_definition_comparison.csv';frame.to_csv(target,index=False)
    total_trades=int(frame.trade_count.sum());p99=int(frame.p99_trade_count.sum());extreme=int(frame.extreme_trade_count.sum());overlap=int(frame.overlap_trade_count.sum())
    payload={'generated_at':datetime.now(timezone.utc).isoformat(),'window':'2026-06-01T00:00:00Z <= timestamp < 2026-08-01T00:00:00Z',
             'markets':len(frame),'eligible_trades':total_trades,
             'p99':{'trades':p99,'trade_share':p99/total_trades,'markets_with_trades':int((frame.p99_trade_count>0).sum()),'value_usdc':float(frame.p99_value_usdc.sum()),'value_share':float(frame.p99_value_usdc.sum()/frame.total_value_usdc.sum())},
             'extreme_1pct_market_volume':{'trades':extreme,'trade_share':extreme/total_trades,'markets_with_trades':int((frame.extreme_trade_count>0).sum()),'markets_without_trades':int((frame.extreme_trade_count==0).sum()),'value_usdc':float(frame.extreme_value_usdc.sum()),'value_share':float(frame.extreme_value_usdc.sum()/frame.total_value_usdc.sum())},
             'overlap':{'trades':overlap,'share_of_extreme':None if extreme==0 else overlap/extreme,'share_of_p99':overlap/p99,'extreme_not_p99':int(frame.extreme_not_p99_count.sum()),'p99_not_extreme':int(frame.p99_not_extreme_count.sum())},
             'by_market_type':{},'comparison_csv_sha256':sha256(target)}
    for typ,g in frame.groupby('market_type'):
        n=int(g.trade_count.sum());payload['by_market_type'][typ]={'eligible_trades':n,'p99_trades':int(g.p99_trade_count.sum()),'p99_share':float(g.p99_trade_count.sum()/n),'extreme_trades':int(g.extreme_trade_count.sum()),'extreme_share':float(g.extreme_trade_count.sum()/n),'markets':len(g),'markets_without_extreme_trades':int((g.extreme_trade_count==0).sum())}
    (OUT/'summary.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    e=payload['extreme_1pct_market_volume'];o=payload['overlap'];
    lines=['# Comparison of Large-Trade Definitions','',f"Canonical eligible trades: **{total_trades:,}** across **{len(frame)}** markets.",'',
           '| Definition | Selected trades | Share of trades | Markets represented | Selected USDC share |','|---|---:|---:|---:|---:|',
           f"| Within-market P99 (inclusive ties) | {p99:,} | {p99/total_trades:.4%} | {payload['p99']['markets_with_trades']} | {payload['p99']['value_share']:.2%} |",
           f"| Trade value $\\geq$ 1% of market total volume | {extreme:,} | {extreme/total_trades:.4%} | {e['markets_with_trades']} | {e['value_share']:.2%} |",'',
           '## Overlap','',f"- Selected by both definitions: {overlap:,} trades.",f"- Share of extreme-definition trades also classified P99: {o['share_of_extreme']:.2%}." if o['share_of_extreme'] is not None else '- No extreme trades were selected.',f"- P99 trades not meeting the 1%-of-volume rule: {o['p99_not_extreme']:,}.",f"- Extreme trades not meeting P99: {o['extreme_not_p99']:,}.",'',
           'The 1%-of-market-volume rule is an extreme-event definition, not the largest one per cent of transactions. It can leave markets with no qualifying trade and should be reported separately from P99.']
    (OUT/'comparison_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(payload,indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
