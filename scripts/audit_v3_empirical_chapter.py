#!/usr/bin/env python3
"""Audit the final v3 Overleaf empirical chapter against model CSV outputs."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OVERLEAF = ROOT / "overleaf"
OUT = ROOT / "docs" / "results"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024): digest.update(chunk)
    return digest.hexdigest()


def row(frame: pd.DataFrame, **filters):
    selected = frame
    for name, value in filters.items(): selected = selected[selected[name].eq(value)]
    if len(selected) != 1: raise AssertionError(f"Expected one row for {filters}, found {len(selected)}")
    return selected.iloc[0]


def main() -> int:
    checks=[]
    def check(name, condition, evidence):
        checks.append({"check":name,"passed":bool(condition),"evidence":str(evidence)})

    primary=pd.read_csv(ROOT/'regression_results/v3/coefficients.csv')
    primary_test=pd.read_csv(ROOT/'regression_results/v3/large_ordinary_equality_test.csv').iloc[0]
    horizons=pd.read_csv(ROOT/'robustness_results/v3/horizons/horizon_equality_tests.csv')
    zero=pd.read_csv(ROOT/'robustness_results/v3/zero/coefficient_equality_tests.csv')
    nonover=pd.read_csv(ROOT/'robustness_results/v3/nonoverlap/coefficient_equality_tests.csv').iloc[0]
    h3=pd.read_csv(ROOT/'robustness_results/v3/h3/h3_equality_tests.csv')
    h4j=pd.read_csv(ROOT/'robustness_results/v3/h4/h4_timing_joint_tests.csv')
    country=pd.read_csv(ROOT/'regression_results/v3/country/large_ordinary_equality_tests.csv')
    phase=pd.read_csv(ROOT/'robustness_results/v3/phase_interaction/phase_contrasts.csv')
    inference=pd.read_csv(ROOT/'robustness_results/v3/inference/inference_equality_tests.csv')

    total=row(primary,model='H1_TOTAL_CTRL',term='signed_log_total_net_flow')
    large=row(primary,model='H2_P99_SPLIT',term='p99_signed_log_large_net_flow')
    ordinary=row(primary,model='H2_P99_SPLIT',term='p99_signed_log_ordinary_net_flow')
    check('H1 coefficient positive and significant',total.estimate>0 and total.p_value<.05,f"beta={total.estimate}, p={total.p_value}")
    check('H2 large coefficient exceeds ordinary',large.estimate>ordinary.estimate,f"large={large.estimate}, ordinary={ordinary.estimate}")
    check('H2 equality test significant and positive',primary_test.coefficient_difference>0 and primary_test.p_value<.05,f"difference={primary_test.coefficient_difference}, p={primary_test.p_value}")
    check('All common horizons support positive large-minus-ordinary gap',(horizons.coefficient_difference>0).all() and (horizons.p_value<.05).all(),f"p range={horizons.p_value.min()}..{horizons.p_value.max()}")
    check('Non-overlap conclusion retained',nonover.coefficient_difference>0 and nonover.p_value<.05,f"difference={nonover.coefficient_difference}, p={nonover.p_value}")
    z=row(zero,model='Z_P99_SPLIT'); c=row(zero,model='C_P99_SPLIT')
    check('Update-incidence difference significant',z.coefficient_difference>0 and z.p_value<.05,f"difference={z.coefficient_difference}, p={z.p_value}")
    check('Conditional non-zero difference not significant',c.p_value>=.05,f"difference={c.coefficient_difference}, p={c.p_value}")
    persistence=row(h3,model='R_P99_SPLIT')
    brier=h3[h3.model.str.startswith('B')]
    check('H3 continuation difference positive and significant',persistence.coefficient_difference>0 and persistence.p_value<.05,f"difference={persistence.coefficient_difference}, p={persistence.p_value}")
    check('H3 Brier equality tests not significant',(brier.p_value>=.05).all(),f"minimum p={brier.p_value.min()}")
    check('H4 timing joint tests significant',(h4j.p_value<.05).all(),f"p values={h4j.p_value.tolist()}")
    phase_did=row(phase,contrast='Difference-in-differences: change in large-minus-ordinary gap')
    check('H4 phase gap difference not significant',phase_did.p_value>=.05,f"difference={phase_did.estimate}, p={phase_did.p_value}")
    c5=row(country,sample='primary_5m',model='O2'); c30=row(country,sample='matched_30m',model='O2')
    check('Country 5m gap positive and significant',c5.estimate_difference>0 and c5.p_value<.05,f"difference={c5.estimate_difference}, p={c5.p_value}")
    check('Country 30m gap positive and significant',c30.estimate_difference>0 and c30.p_value<.05,f"difference={c30.estimate_difference}, p={c30.p_value}")
    check('Alternative clustering retains H2 equality result',(inference.estimate_difference>0).all() and (inference.p_value<.05).all(),f"p range={inference.p_value.min()}..{inference.p_value.max()}")

    tex_files=sorted(OVERLEAF.rglob('*.tex')); combined='\n'.join(p.read_text(encoding='utf-8') for p in tex_files)
    # Wallet-level whale analyses were added as explicitly secondary extensions,
    # so a blanket ban on the word "whale" is no longer valid. Guard instead
    # against relabelling the P99 primary treatment as whale flow.
    lower=combined.lower()
    legacy_primary_labels=('p99 whale', 'whale-trade coefficient',
                           'whale flow has a larger', 'whale-versus-ordinary')
    check('Primary P99 treatment is not relabelled as whale flow',
          not any(label in lower for label in legacy_primary_labels),
          f"forbidden primary labels={legacy_primary_labels}")
    check('Primary P99 definition present','within-market P99 individual trades' in combined,'research_design.tex')
    check('H4 difference-in-differences reported','p=0.734' in combined,'results.tex')
    check('Zero-outcome qualification reported','p=0.334' in combined,'results.tex')
    missing_inputs=[]
    for p in tex_files:
        for line in p.read_text(encoding='utf-8').splitlines():
            line=line.strip()
            if line.startswith('\\input{'):
                target=OVERLEAF/(line[7:-1]+'.tex')
                if not target.exists(): missing_inputs.append(str(target))
    check('All LaTeX inputs exist',not missing_inputs,missing_inputs or 'all targets found')
    missing_figures=[]
    for p in tex_files:
        for line in p.read_text(encoding='utf-8').splitlines():
            line=line.strip()
            if line.startswith('\\includegraphics'):
                target=line.split('{',1)[1].rstrip('}')
                if not (OVERLEAF/target).exists(): missing_figures.append(target)
    check('All LaTeX figure targets exist',not missing_figures,missing_figures or 'all figure targets found')
    structural=[]
    for p in tex_files:
        text=p.read_text(encoding='utf-8')
        for left,right,label in [('{','}','braces'),('\\begin{table}','\\end{table}','table'),('\\begin{tabular}','\\end{tabular}','tabular')]:
            if text.count(left)!=text.count(right): structural.append(f'{p}:{label}')
    check('LaTeX lightweight structural checks pass',not structural,structural or 'balanced')

    expected_tables={
        'table_summary_statistics.tex':['748,481','-0.000001','0.396755'],
        'table_correlation_matrix.tex':['1.000','0.920','-0.727'],
        'table_1_match_primary.tex':['0.000146','0.000048','748,481'],
        'table_2_match_robustness.tex':['0.000149','149,438','744,470'],
        'table_3_h3.tex':['0.000033','0.000017','744,470'],
        'table_4_h4.tex':['0.000190','0.000141','11.019','17.340'],
        'table_5_country.tex':['0.000091','0.000068','646,607','645,157'],
        'table_6_phase.tex':['0.000546','-0.000062','0.734'],
        'appendix_zero_nonoverlap.tex':['0.016793','0.010975','0.000417','0.000457'],
        'appendix_alternative_clustering.tex':['Event + hour','0.000146','0.000048'],
        'table_hypothesis_decisions.tex':['0.000071','0.000098','3.02','0.734'],
    }
    for name,tokens in expected_tables.items():
        text=(OVERLEAF/'tables'/name).read_text(encoding='utf-8')
        check(f'Table tokens verified: {name}',all(token in text for token in tokens),tokens)

    failures=[x for x in checks if not x['passed']]
    payload={"generated_at":datetime.now(timezone.utc).isoformat(),"status":"PASS" if not failures else "FAIL",
             "checks":len(checks),"passed":len(checks)-len(failures),"failed":len(failures),
             "items":checks,"source_hashes":{str(p.relative_to(ROOT)):sha256(p) for p in tex_files}}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'v3_empirical_chapter_audit.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    lines=['# V3 Empirical Chapter Audit','',f"**Status:** {payload['status']}",f"**Generated:** {payload['generated_at']}",
           f"**Checks:** {payload['passed']}/{payload['checks']} passed",'', '| Check | Status | Evidence |','|---|---|---|']
    for item in checks:
        evidence=item['evidence'].replace('|','/').replace('\n',' ')
        lines.append(f"| {item['check']} | {'PASS' if item['passed'] else 'FAIL'} | {evidence} |")
    lines += ['', '## Freeze decision','', 'The v3 empirical chapter is eligible to be frozen.' if not failures else 'Do not freeze the v3 empirical chapter until all failed checks are resolved.',
              '', 'This audit verifies numerical and textual consistency. A final visual compilation check must still be completed in Overleaf because no local TeX engine is installed.']
    (OUT/'v3_empirical_chapter_audit.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:payload[k] for k in ('status','checks','passed','failed')},indent=2))
    return 1 if failures else 0


if __name__=='__main__': raise SystemExit(main())
