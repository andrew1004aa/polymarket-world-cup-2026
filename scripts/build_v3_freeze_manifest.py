#!/usr/bin/env python3
"""Create the immutable-content manifest for the final v3 empirical chapter."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/results/v3_freeze'

FILES={
    'design':[
        'docs/research_design/v3_final_research_design.md',
        'docs/research_design/large_trade_specification.md',
        'overleaf/sections/research_design.tex'],
    'chapter':[
        'overleaf/main.tex',
        'overleaf/sections/results.tex',
        'overleaf/sections/appendix_robustness.tex'],
    'table':[
        'overleaf/tables/table_summary_statistics.tex',
        'overleaf/tables/table_correlation_matrix.tex',
        'overleaf/tables/table_1_match_primary.tex',
        'overleaf/tables/table_2_match_robustness.tex',
        'overleaf/tables/table_3_h3.tex',
        'overleaf/tables/table_4_h4.tex',
        'overleaf/tables/table_5_country.tex',
        'overleaf/tables/table_6_phase.tex',
        'overleaf/tables/appendix_zero_nonoverlap.tex',
        'overleaf/tables/appendix_alternative_clustering.tex',
        'overleaf/tables/table_hypothesis_decisions.tex'],
    'figure':[
        'overleaf/figures/fig_1_horizon_large_vs_ordinary.pdf',
        'overleaf/figures/fig_2_h4_timing_large_vs_ordinary.pdf'],
    'primary_result_source':[
        'regression_results/v3/coefficients.csv',
        'regression_results/v3/model_diagnostics.csv',
        'regression_results/v3/large_ordinary_equality_test.csv'],
    'robustness_result_source':[
        'robustness_results/v3/horizons/horizon_coefficients.csv',
        'robustness_results/v3/horizons/horizon_equality_tests.csv',
        'robustness_results/v3/nonoverlap/coefficients.csv',
        'robustness_results/v3/nonoverlap/coefficient_equality_tests.csv',
        'robustness_results/v3/zero/coefficients.csv',
        'robustness_results/v3/zero/coefficient_equality_tests.csv',
        'robustness_results/v3/h3/h3_coefficients.csv',
        'robustness_results/v3/h3/h3_equality_tests.csv',
        'robustness_results/v3/h4/h4_timing_marginal_effects.csv',
        'robustness_results/v3/h4/h4_timing_joint_tests.csv',
        'robustness_results/v3/inference/inference_coefficients.csv',
        'robustness_results/v3/inference/inference_equality_tests.csv',
        'robustness_results/v3/phase_interaction/coefficients.csv',
        'robustness_results/v3/phase_interaction/phase_contrasts.csv'],
    'descriptive_result_source':[
        'regression_results/v3/descriptive/summary_statistics.csv',
        'regression_results/v3/descriptive/correlation_matrix.csv',
        'regression_results/v3/descriptive/graph_1_horizon_large_vs_ordinary.csv',
        'regression_results/v3/descriptive/graph_2_h4_timing_large_vs_ordinary.csv'],
    'country_result_source':[
        'regression_results/v3/country/coefficients.csv',
        'regression_results/v3/country/large_ordinary_equality_tests.csv',
        'robustness_results/v3/country_exclude_sparse/coefficients.csv',
        'robustness_results/v3/country_exclude_sparse/large_ordinary_equality_tests.csv'],
    'reproduction_script':[
        'scripts/build_overleaf_v3_tables.py',
        'scripts/build_v3_descriptive_stats.py',
        'scripts/build_v3_coefficient_plots.py',
        'scripts/audit_v3_empirical_chapter.py',
        'scripts/run_phase_interaction_v3.py',
        'scripts/prepare_postkickoff_regression_inputs_v3.py'],
    'audit':[
        'docs/results/v3_empirical_chapter_audit.md',
        'docs/results/v3_empirical_chapter_audit.json']}

def digest(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(1024*1024):h.update(block)
    return h.hexdigest()

def main()->int:
    audit=json.loads((ROOT/'docs/results/v3_empirical_chapter_audit.json').read_text())
    if audit.get('status')!='PASS' or audit.get('failed')!=0:
        raise SystemExit('Empirical chapter audit has not passed')
    rows=[];missing=[]
    for role,names in FILES.items():
        for name in names:
            path=ROOT/name
            if not path.is_file():missing.append(name);continue
            rows.append({'role':role,'path':name,'bytes':path.stat().st_size,'sha256':digest(path)})
    if missing:raise SystemExit(f'Missing freeze targets: {missing}')
    OUT.mkdir(parents=True,exist_ok=True)
    csv_path=OUT/'v3_freeze_manifest.csv'
    with csv_path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['role','path','bytes','sha256']);w.writeheader();w.writerows(rows)
    payload={
        'release':'v3','created_at':datetime.now(timezone.utc).isoformat(),
        'status':'content_frozen_pending_overleaf_visual_qa',
        'definition':'within-market P99 individual trade value; inclusive threshold ties',
        'canonical_window':'2026-06-01T00:00:00Z <= timestamp < 2026-08-01T00:00:00Z',
        'primary_sample':'312 pre-kickoff match markets nested in 104 events; five-minute horizon',
        'manifest_entries':len(rows),'manifest_csv_sha256':digest(csv_path),
        'empirical_audit_status':'PASS','empirical_audit_checks':audit['checks'],
        'visual_qa':{'status':'pending','required_checks':['table overflow','float placement','cross-references','page breaks','font and caption consistency']},
        'version_pointer_changed':False,
        'governance':['Do not edit frozen presentation files manually; update the v3 generator and rerun the audit.','Do not move CURRENT_VERSION until Overleaf visual QA is signed PASS.','V1 wallet-based artifacts remain preserved as secondary analysis.']}
    json_path=OUT/'v3_freeze_manifest.json'
    json_path.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    md=['# V3 Empirical Chapter Freeze Manifest','',f"**Status:** `{payload['status']}`",f"**Created:** {payload['created_at']}",f"**Manifest entries:** {len(rows)}",f"**Numerical/text audit:** PASS ({audit['passed']}/{audit['checks']})",'',
        'The empirical content and its source results are hash-locked. Final release remains conditional on visual compilation review in Overleaf. No `CURRENT_VERSION` pointer has been changed.','',
        '## Required visual sign-off','', '- No table extends beyond the text area.','- Table floats and page breaks are acceptable.','- Every table and appendix cross-reference resolves.','- Captions, fonts and significance notes render consistently.','',
        '## Integrity rule','', 'If any listed file changes, regenerate the tables where applicable, rerun the empirical audit, and rebuild this manifest. Do not edit a numerical LaTeX cell independently of its source CSV.','',
        '## Files','', '| Role | Path | Bytes | SHA-256 |','|---|---|---:|---|']
    for x in rows:md.append(f"| {x['role']} | `{x['path']}` | {x['bytes']} | `{x['sha256']}` |")
    (OUT/'README.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(json.dumps({'status':payload['status'],'entries':len(rows),'audit':'PASS','version_pointer_changed':False},indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
