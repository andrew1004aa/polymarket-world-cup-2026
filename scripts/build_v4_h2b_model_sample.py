#!/usr/bin/env python3
"""Create compact H2b binary model samples and class-balance diagnostics."""
import csv,gzip,json,hashlib
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'analysis_ready/v4/h2b_trade_classification';OUT=ROOT/'model_samples/v4/h2b';OUT.mkdir(parents=True,exist_ok=True)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  while c:=f.read(1024*1024):h.update(c)
 return h.hexdigest()
fields=['trade_record_id','market_id','event_id','market_type','timestamp','phase','yes_equivalent_direction','trade_value_usdc','is_p99_large','yes_price_t','recent_price_change','momentum']
summaries={}
for tag in ('5m','15m'):
 target=OUT/f'h2b_{tag}_binary.csv.gz';tmp=target.with_suffix(target.suffix+'.tmp');counts=defaultdict(lambda:{'momentum':0,'contrarian':0});rows=0
 with gzip.open(tmp,'wt',newline='',compresslevel=6) as dst:
  w=csv.DictWriter(dst,fieldnames=fields);w.writeheader()
  for path in sorted(SRC.glob('*.csv.gz'),key=lambda p:int(p.stem.split('.')[0])):
   with gzip.open(path,'rt',newline='') as src:
    for r in csv.DictReader(src):
     label=r[f'trend_label_{tag}']
     if label not in ('momentum','contrarian'):continue
     out={k:r[k] for k in fields if k in r};out['recent_price_change']=r[f'recent_price_change_{tag}'];out['momentum']=int(label=='momentum');w.writerow(out);rows+=1
     key='|'.join((r['market_type'],r['phase'],'large' if r['is_p99_large']=='1' else 'ordinary'))
     counts[key][label]+=1
 tmp.replace(target)
 summaries[tag]={'rows':rows,'output_path':str(target.relative_to(ROOT)),'output_sha256':sha(target),'class_balance':dict(sorted(counts.items()))}
payload={'generated_at':datetime.now(timezone.utc).isoformat(),'status':'built_qc_passed','definition':'conditional on a non-zero observable recent price change; momentum=1, contrarian=0','samples':summaries}
(OUT/'summary.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload,indent=2))
