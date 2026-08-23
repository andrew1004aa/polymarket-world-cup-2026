#!/usr/bin/env python3
"""Generate the final Overleaf regression tables directly from v3 CSV outputs."""

from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'overleaf/tables'

def read(path): return pd.read_csv(ROOT/path)
def star(p): return '***' if p<.01 else ('**' if p<.05 else ('*' if p<.10 else ''))
def num(x): return f'{float(x):.6f}'
def pval(x): return '<0.001' if float(x)<.001 else f'{float(x):.3f}'
def pcell(x): v=pval(x); return f'p{v}' if v.startswith('<') else f'p={v}'
def row(df, **kw):
    z=df
    for k,v in kw.items(): z=z[z[k].eq(v)]
    if len(z)!=1: raise RuntimeError(f'Expected one row for {kw}, got {len(z)}')
    return z.iloc[0]
def cell(r): return num(r.estimate)+star(r.p_value), '('+num(r.std_error)+')'
def write(name, lines): (OUT/name).write_text('\n'.join(lines)+'\n',encoding='utf-8')

primary=read('regression_results/v3/coefficients.csv'); pdg=read('regression_results/v3/model_diagnostics.csv')
models=['H1_TOTAL_UNADJ','H1_TOTAL_CTRL','H2_P99_SPLIT']
labels=[('Signed-log total flow','signed_log_total_net_flow'),('P99 large flow','p99_signed_log_large_net_flow'),('Ordinary flow','p99_signed_log_ordinary_net_flow'),('Baseline YES price','yes_price_t'),('Lagged 30-minute price change','lagged_30m_price_change'),('Log lagged 30-minute volume','log_lagged_30m_volume'),('Log minutes to kickoff','log_minutes_to_kickoff')]
L=[r'\begin{table}[htbp]',r'\centering',r'\caption{Pre-match individual match-market regressions}',r'\label{tab:match-primary}',r'\small',r'\begin{tabular}{lccc}',r'\hline',r' & H1 unadjusted & H1 controlled & H2 P99 split \\',r'\hline']
for label,term in labels:
    vals=[]; ses=[]
    for m in models:
        z=primary[(primary.model==m)&(primary.term==term)]
        if len(z): a,b=cell(z.iloc[0]); vals.append(a); ses.append(b)
        else: vals.append(''); ses.append('')
    L += [label+' & '+' & '.join(vals)+r' \\', ' & '+' & '.join(ses)+r' \\']
L += [r'\hline','Observations & '+' & '.join(f"{int(row(pdg,model=m).observations):,}" for m in models)+r' \\', 'Within $R^2$ & '+' & '.join(f"{row(pdg,model=m).within_r_squared:.4f}" for m in models)+r' \\',r'Market fixed effects & Yes & Yes & Yes \\',r'Calendar-hour fixed effects & Yes & Yes & Yes \\',r'\hline',r'\multicolumn{4}{p{0.92\textwidth}}{\footnotesize Notes: The dependent variable is the five-minute YES-equivalent probability change. CRV1 standard errors clustered by 104 match events are in parentheses. Large trades are individual trades at or above their market-specific P99 threshold. $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.}',r'\end{tabular}',r'\end{table}']; write('table_1_match_primary.tex',L)

h=read('robustness_results/v3/horizons/horizon_coefficients.csv'); n=read('robustness_results/v3/nonoverlap/coefficients.csv')
L=[r'\begin{table}[htbp]',r'\centering',r'\caption{Horizon and non-overlap robustness}',r'\label{tab:match-horizons}',r'\scriptsize',r'\begin{tabular}{lcccccc}',r'\hline',r' & 1m total & 5m total & 15m total & 1m split & 5m split & 15m split \\',r'\hline']
for label,term,split in [('Total flow','signed_log_total_net_flow',False),('P99 large flow','p99_signed_log_large_net_flow',True),('Ordinary flow','p99_signed_log_ordinary_net_flow',True)]:
    vals=[]; ses=[]
    for hz in (1,5,15):
        m=f'H{hz}_{"SPLIT" if split else "TOTAL"}'; a,b=cell(row(h,model=m,term=term)); vals.append(a); ses.append(b)
    vals=(['','','']+vals) if split else (vals+['','','']); ses=(['','','']+ses) if split else (ses+['','',''])
    L += [label+' & '+' & '.join(vals)+r' \\', ' & '+' & '.join(ses)+r' \\']
L += [r'\hline',r'\multicolumn{7}{l}{\textit{Panel B: UTC-aligned non-overlapping five-minute windows}} \\']
for label,m,term in [('Total flow','N_TOTAL','signed_log_total_net_flow'),('P99 large flow','N_P99_SPLIT','p99_signed_log_large_net_flow'),('Ordinary flow','N_P99_SPLIT','p99_signed_log_ordinary_net_flow')]:
    a,b=cell(row(n,model=m,term=term)); L += [label+' & '+a+r' & & & & & \\', ' & '+b+r' & & & & & \\']
L += [r'\hline',r'\multicolumn{7}{p{0.96\textwidth}}{\footnotesize Notes: Horizon models use 744,470 common observations. Non-overlap models use 149,438 observations. All models use frozen controls, market and calendar-hour fixed effects, and event-clustered CRV1 standard errors.}',r'\end{tabular}',r'\end{table}']; write('table_2_match_robustness.tex',L)

h3=read('robustness_results/v3/h3/h3_coefficients.csv')
spec=[('Total signed flow','R_TOTAL','signed_log_total_net_flow',0),('P99 large flow','R_P99_SPLIT','p99_signed_log_large_net_flow',1),('Ordinary flow','R_P99_SPLIT','p99_signed_log_ordinary_net_flow',1),('Absolute P99 large flow','B5_P99_SPLIT','p99_log_abs_large_net_flow',2),('Absolute P99 large flow','B15_P99_SPLIT','p99_log_abs_large_net_flow',3),('Absolute P99 large flow','B5_15_P99_SPLIT','p99_log_abs_large_net_flow',4),('Absolute ordinary flow','B5_P99_SPLIT','p99_log_abs_ordinary_net_flow',2),('Absolute ordinary flow','B15_P99_SPLIT','p99_log_abs_ordinary_net_flow',3),('Absolute ordinary flow','B5_15_P99_SPLIT','p99_log_abs_ordinary_net_flow',4)]
grid={}
for label,m,t,col in spec: grid.setdefault(label,{})[col]=cell(row(h3,model=m,term=t))
L=[r'\begin{table}[htbp]',r'\centering',r'\caption{Persistence and forecast-improvement tests (H3)}',r'\label{tab:h3}',r'\scriptsize',r'\begin{tabular}{lccccc}',r'\hline',r' & Subsequent total & Subsequent split & Brier 0--5 & Brier 0--15 & Brier 5--15 \\',r'\hline']
for label,cells in grid.items():
    vals=[];ses=[]
    for i in range(5):
        a,b=cells.get(i,('','')); vals.append(a);ses.append(b)
    L += [label+' & '+' & '.join(vals)+r' \\', ' & '+' & '.join(ses)+r' \\']
L += [r'\hline',r'\multicolumn{6}{p{0.94\textwidth}}{\footnotesize Notes: Subsequent movement is $p_{t+15}-p_{t+5}$ conditional on the initial response. Positive Brier improvement denotes movement closer to the resolved outcome. All models use 744,470 observations and event-clustered CRV1 standard errors.}',r'\end{tabular}',r'\end{table}'];write('table_3_h3.tex',L)

m=read('robustness_results/v3/h4/h4_timing_marginal_effects.csv'); j=read('robustness_results/v3/h4/h4_timing_joint_tests.csv')
bands=[('gt_24h','$>24$ hours'),('12_to_24h','12--24 hours'),('6_to_12h','6--12 hours'),('1_to_6h','1--6 hours'),('last_60m','Final 60 minutes')]
L=[r'\begin{table}[htbp]',r'\centering',r'\caption{Time-to-kickoff heterogeneity (H4)}',r'\label{tab:h4}',r'\small',r'\begin{tabular}{lccc}',r'\hline',r'Time to kickoff & P99 large flow & Ordinary flow & Difference \\',r'\hline']
for b,label in bands:
    parts=[]
    for typ in ('large','ordinary','large_minus_ordinary'):
        z=row(m,band=b,flow_type=typ); parts.append(f'{num(z.estimate)}{star(z.p_value)} ({num(z.std_error)})')
    L.append(label+' & '+' & '.join(parts)+r' \\')
jl=row(j,test='all_large_timing_interactions_zero'); jo=row(j,test='all_ordinary_timing_interactions_zero')
L += [r'\hline',f'Joint interaction $F$ test & {jl.f_statistic:.3f}{star(jl.p_value)} & {jo.f_statistic:.3f}{star(jo.p_value)} &  '+r'\\',r'\hline',r'\multicolumn{4}{p{0.92\textwidth}}{\footnotesize Notes: Entries are pooled-model implied coefficients with clustered standard errors. Joint tests use $(4,103)$ degrees of freedom. The pattern is non-monotonic.}',r'\end{tabular}',r'\end{table}'];write('table_4_h4.tex',L)

c=read('regression_results/v3/country/coefficients.csv'); samples=[('primary_5m','5m total','O1'),('primary_5m','5m split','O2'),('matched_30m','30m total','O1'),('matched_30m','30m split','O2')]
L=[r'\begin{table}[htbp]',r'\centering',r'\caption{Country outright-market regressions}',r'\label{tab:country}',r'\small',r'\begin{tabular}{lcccc}',r'\hline',' & '+' & '.join(x[1] for x in samples)+r' \\',r'\hline']
for label,term in [('Total flow','signed_log_total_net_flow'),('P99 large flow','p99_signed_log_large_net_flow'),('Ordinary flow','p99_signed_log_ordinary_net_flow')]:
    vals=[];ses=[]
    for sample,_,model in samples:
        z=c[(c['sample']==sample)&(c.model==model)&(c.term==term)]
        if len(z): a,b=cell(z.iloc[0]);vals.append(a);ses.append(b)
        else: vals.append('');ses.append('')
    L += [label+' & '+' & '.join(vals)+r' \\', ' & '+' & '.join(ses)+r' \\']
L += [r'\hline',r'Observations & 646,607 & 646,607 & 645,157 & 645,157 \\',r'Market fixed effects & Yes & Yes & Yes & Yes \\',r'Calendar-hour fixed effects & Yes & Yes & Yes & Yes \\',r'\hline',r'\multicolumn{5}{p{0.94\textwidth}}{\footnotesize Notes: Country markets are separate secondary models. CRV1 standard errors are clustered by 48 markets. The 30-minute models use the common 5/30-minute sample.}',r'\end{tabular}',r'\end{table}'];write('table_5_country.tex',L)

ph=read('robustness_results/v3/phase_interaction/phase_contrasts.csv')
L=[r'\begin{table}[htbp]',r'\centering',r'\caption{Pre-/post-kickoff phase contrasts (H4)}',r'\label{tab:phase}',r'\small',r'\begin{tabular}{p{0.42\textwidth}rrrr}',r'\hline',r'Contrast & Estimate & SE & 95\% CI & $p$-value \\',r'\hline']
for z in ph.itertuples(): L.append(f'{z.contrast} & {num(z.estimate)} & {num(z.std_error)} & [{num(z.conf_low)}, {num(z.conf_high)}] & {pval(z.p_value)} '+r'\\')
L += [r'\hline',r'\multicolumn{5}{p{0.95\textwidth}}{\footnotesize Notes: The pooled five-minute sample contains 794,178 observations. Models use common controls, market and calendar-hour fixed effects, and CRV1 standard errors clustered by 104 events. Post-kickoff means before recorded resolution and is not necessarily purely in-play.}',r'\end{tabular}',r'\end{table}'];write('table_6_phase.tex',L)

term_labels={'signed_log_total_net_flow':'Total flow','p99_signed_log_large_net_flow':'P99 large flow','p99_signed_log_ordinary_net_flow':'Ordinary flow','log_abs_total_net_flow':'Total flow (absolute)','p99_log_abs_large_net_flow':'P99 large flow (absolute)','p99_log_abs_ordinary_net_flow':'Ordinary flow (absolute)'}

z=read('robustness_results/v3/zero/coefficients.csv'); no=read('robustness_results/v3/nonoverlap/coefficients.csv')
entries=[('Update incidence','Z_TOTAL','log_abs_total_net_flow',z),('Update incidence','Z_P99_SPLIT','p99_log_abs_large_net_flow',z),('Update incidence','Z_P99_SPLIT','p99_log_abs_ordinary_net_flow',z),('Conditional non-zero','C_TOTAL','signed_log_total_net_flow',z),('Conditional non-zero','C_P99_SPLIT','p99_signed_log_large_net_flow',z),('Conditional non-zero','C_P99_SPLIT','p99_signed_log_ordinary_net_flow',z),('Non-overlap','N_TOTAL','signed_log_total_net_flow',no),('Non-overlap','N_P99_SPLIT','p99_signed_log_large_net_flow',no),('Non-overlap','N_P99_SPLIT','p99_signed_log_ordinary_net_flow',no)]
L=[r'\begin{table}[htbp]',r'\centering',r'\caption{Zero-outcome and non-overlap robustness}',r'\label{tab:appendix-zero}',r'\scriptsize',r'\begin{tabular}{lllrr}',r'\hline',r'Family & Flow term & Estimate & SE & $p$-value \\',r'\hline']
for fam,model,term,df in entries:
    x=row(df,model=model,term=term); L.append(f'{fam} & {term_labels[term]} & {num(x.estimate)} & {num(x.std_error)} & {pval(x.p_value)} '+r'\\')
L += [r'\hline',r'\end{tabular}',r'\end{table}'];write('appendix_zero_nonoverlap.tex',L)

i=read('robustness_results/v3/inference/inference_coefficients.csv')
model_labels={'H2_P99_SPLIT':'H2 (P99 split)','H1_TOTAL_CTRL':'H1 (controlled)'}
cov_labels={'event_crv1':'Event','market_crv1':'Market','event_hour_two_way_crv1':'Event + hour'}
L=[r'\begin{table}[htbp]',r'\centering',r'\caption{Alternative clustered inference}',r'\label{tab:appendix-clusters}',r'\scriptsize',r'\begin{tabular}{llllrrr}',r'\hline',r'Model & Covariance & Flow term & Estimate & SE & $p$-value & Rows \\',r'\hline']
for x in i[i.term.str.contains('flow')].itertuples(): L.append(f'{model_labels.get(x.model,x.model)} & {cov_labels[x.covariance]} & {term_labels[x.term]} & {num(x.estimate)} & {num(x.std_error)} & {pval(x.p_value)} & 748,481 '+r'\\')
L += [r'\hline',r'\multicolumn{7}{p{0.96\textwidth}}{\footnotesize Notes: Point estimates are unchanged across covariance estimators. Wild-cluster bootstrap was not rerun because the planned 9,999-draw implementation produced severe memory pressure; no lower-draw substitute is reported.}',r'\end{tabular}',r'\end{table}'];write('appendix_alternative_clustering.tex',L)

h2eq=read('regression_results/v3/large_ordinary_equality_test.csv').iloc[0]
h1row=row(primary,model='H1_TOTAL_CTRL',term='signed_log_total_net_flow')
h2l_row=row(primary,model='H2_P99_SPLIT',term='p99_signed_log_large_net_flow')
h2o_row=row(primary,model='H2_P99_SPLIT',term='p99_signed_log_ordinary_net_flow')
ratio=h2l_row.estimate/h2o_row.estimate
h3eq_all=read('robustness_results/v3/h3/h3_equality_tests.csv')
h3_cont=row(h3eq_all,model='R_P99_SPLIT')
h3_brier=h3eq_all[h3eq_all.model.str.startswith('B')]
h4j_all=read('robustness_results/v3/h4/h4_timing_joint_tests.csv')
h4_large_f=row(h4j_all,test='all_large_timing_interactions_zero')
h4_ord_f=row(h4j_all,test='all_ordinary_timing_interactions_zero')
phase_did=row(ph,contrast='Difference-in-differences: change in large-minus-ordinary gap')
decisions=[
    ('H1: total flow and short-horizon price','Supported (conditional association)',
     f'$\\beta={num(h1row.estimate)}$, ${pcell(h1row.p_value)}$'),
    ('H2: large- vs ordinary-flow association','Supported; concentrated in update incidence rather than conditional magnitude',
     f'Diff $={num(h2eq.coefficient_difference)}$, ${pcell(h2eq.p_value)}$; ratio ${ratio:.2f}\\times$'),
    ('H3: persistence, reversal and forecast improvement','Mixed: persistence supported, forecast improvement not supported',
     f'Diff $={num(h3_cont.coefficient_difference)}$, ${pcell(h3_cont.p_value)}$; Brier tests n.s.\\ (min $p={h3_brier.p_value.min():.3f}$)'),
    ('H4: timing and phase heterogeneity','Timing supported (non-monotonic); phase-specific premium change not supported',
     f'Joint $F={h4_large_f.f_statistic:.3f}$/${h4_ord_f.f_statistic:.3f}$, ${pcell(h4_large_f.p_value)}$; phase DiD ${pcell(phase_did.p_value)}$'),
]
L=[r'\begin{table}[htbp]',r'\centering',r'\caption{Hypothesis decisions summary}',r'\label{tab:hypothesis-decisions}',r'\small',r'\begin{tabular}{p{0.24\textwidth}p{0.40\textwidth}p{0.28\textwidth}}',r'\hline',r'Hypothesis & Decision & Key evidence \\',r'\hline']
for h,d,e in decisions: L.append(f'{h} & {d} & {e} '+r'\\ \hline')
L += [r'\multicolumn{3}{p{0.94\textwidth}}{\footnotesize Notes: All decisions describe conditional associations only. No decision implies a causal effect, informed trading, institutional trading, or a market-integrity violation. n.s.\ denotes not significant at the 5\% level.}',r'\end{tabular}',r'\end{table}'];write('table_hypothesis_decisions.tex',L)

print('Generated 9 v3 Overleaf tables')
