#!/usr/bin/env python3
"""Add Holm corrections to an already completed H2a permutation run."""
import csv, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'regression_results/v4/h2a'
path=OUT/'h2a_permutation_summary.json'; payload=json.loads(path.read_text()); rows=payload['results']
for family in sorted({r['market_family'] for r in rows}):
    family_rows=[r for r in rows if r['market_family']==family]
    for field,out in [('one_sided_upper_p','holm_one_sided_p'),('two_sided_p','holm_two_sided_p')]:
        ordered=sorted(family_rows,key=lambda r:r[field]); running=0.; total=len(ordered)
        for rank,row in enumerate(ordered):
            running=max(running,min(1.,(total-rank)*row[field])); row[out]=running
path.write_text(json.dumps(payload,indent=2)+'\n')
with (OUT/'h2a_permutation_results.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
lines=['# V4 H2a stratified permutation test','',f"Permutations: {rows[0]['permutations']}; base seed: 20260813.",'',
'The initiating direction is shuffled within market × UTC date × kickoff phase. Event timestamps and other-wallet flows remain fixed.','',
'| Family | Window | N | Observed signed-log difference | Raw USDC difference | Null z-distance | Upper p | Holm upper p | Two-sided p | Holm two-sided p |',
'|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['market_family']} | {r['window']} | {r['events']:,} | {r['observed_mean_post_minus_pre_signed_log_directional_net']:.6f} | {r['observed_mean_post_minus_pre_directional_net_usdc']:.2f} | {r['standardized_distance_from_null']:.2f} | {r['one_sided_upper_p']:.4g} | {r['holm_one_sided_p']:.4g} | {r['two_sided_p']:.4g} | {r['holm_two_sided_p']:.4g} |")
lines += ['', 'These results do not identify causality, trader intent, market making or manipulation.']
(OUT/'h2a_permutation_report.md').write_text('\n'.join(lines)+'\n')
