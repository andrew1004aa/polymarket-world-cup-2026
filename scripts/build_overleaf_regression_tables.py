#!/usr/bin/env python3
"""Build reproducible LaTeX tables and Results text from frozen CSV outputs."""

from __future__ import annotations

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "overleaf/tables"
SECTIONS = ROOT / "overleaf/sections"


def stars(p: float) -> str:
    return "***" if p < .01 else ("**" if p < .05 else ("*" if p < .10 else ""))


def cell(row, digits=6) -> tuple[str, str]:
    return f"{row.estimate:.{digits}f}{stars(row.p_value)}", f"({row.std_error:.{digits}f})"


def lookup(data, model, term, **filters):
    selected = data[(data.model == model) & (data.term == term)]
    for key, value in filters.items(): selected = selected[selected[key] == value]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one row: {model}, {term}, {filters}; got {len(selected)}")
    return selected.iloc[0]


def write(name: str, text: str) -> None:
    path = TABLES / name
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text.strip() + "\n", encoding="utf-8")
    tmp.replace(path)


def table1(primary: pd.DataFrame, diag: pd.DataFrame) -> str:
    terms = [("Signed-log total flow","signed_log_total_net_flow"),
             ("Signed-log whale flow","signed_log_whale_net_flow"),
             ("Signed-log non-whale flow","signed_log_nonwhale_net_flow"),
             ("Baseline YES price","yes_price_t"),
             ("Lagged 30-minute price change","lagged_30m_price_change"),
             ("Log lagged 30-minute volume","log_lagged_30m_volume"),
             ("Log minutes to kickoff","log_minutes_to_kickoff")]
    models=["P1","P2","P3"]
    lines=[r"\begin{table}[htbp]",r"\centering",r"\caption{Pre-match individual match-market regressions}",r"\label{tab:match-primary}",r"\small",r"\begin{tabular}{lccc}",r"\hline",r" & P1 & P2 & P3 \\",r"\hline"]
    for label,term in terms:
        vals=[]; ses=[]
        for model in models:
            rows=primary[(primary.model==model)&(primary.term==term)]
            if rows.empty: vals.append(""); ses.append("")
            else:
                a,b=cell(rows.iloc[0]); vals.append(a); ses.append(b)
        lines += [label+" & "+" & ".join(vals)+r" \\"," & "+" & ".join(ses)+r" \\"]
    n=[int(diag.loc[diag.model==m,"observations"].iloc[0]) for m in models]
    r2=[diag.loc[diag.model==m,"within_r_squared"].iloc[0] for m in models]
    lines += [r"\hline","Observations & "+" & ".join(f"{x:,}" for x in n)+r" \\",
              "Within $R^2$ & "+" & ".join(f"{x:.4f}" for x in r2)+r" \\",
              r"Market fixed effects & Yes & Yes & Yes \\",r"Calendar-hour fixed effects & Yes & Yes & Yes \\",
              r"\hline",r"\multicolumn{4}{p{0.92\textwidth}}{\footnotesize Notes: The dependent variable is the five-minute change in YES-equivalent probability. CRV1 standard errors clustered by 104 FIFA match events are in parentheses. $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.}",r"\end{tabular}",r"\end{table}"]
    return "\n".join(lines)


def table2(h: pd.DataFrame, nonoverlap: pd.DataFrame) -> str:
    cols=[("1m total","H1_TOTAL","signed_log_total_net_flow"),("5m total","H5_TOTAL","signed_log_total_net_flow"),("15m total","H15_TOTAL","signed_log_total_net_flow"),
          ("1m split","H1_SPLIT","signed_log_whale_net_flow"),("5m split","H5_SPLIT","signed_log_whale_net_flow"),("15m split","H15_SPLIT","signed_log_whale_net_flow")]
    # Actual model names are verified dynamically below.
    horizon_models={("1","total"):"H1_TOTAL", ("5","total"):"H5_TOTAL",
                    ("15","total"):"H15_TOTAL", ("1","split"):"H1_SPLIT",
                    ("5","split"):"H5_SPLIT", ("15","split"):"H15_SPLIT"}
    order=[("1","total"),("5","total"),("15","total"),("1","split"),("5","split"),("15","split")]
    heads=["1m total","5m total","15m total","1m split","5m split","15m split"]
    lines=[r"\begin{table}[htbp]",r"\centering",r"\caption{Match-market horizon and non-overlap robustness}",r"\label{tab:match-horizons}",r"\scriptsize",r"\begin{tabular}{lcccccc}",r"\hline"," & "+" & ".join(heads)+r" \\",r"\hline"]
    for label,term in [("Total flow","signed_log_total_net_flow"),("Whale flow","signed_log_whale_net_flow"),("Non-whale flow","signed_log_nonwhale_net_flow")]:
        vals=[];ses=[]
        for key in order:
            m=horizon_models[key]; rows=h[(h.model==m)&(h.term==term)]
            if rows.empty: vals.append("");ses.append("")
            else: a,b=cell(rows.iloc[0]);vals.append(a);ses.append(b)
        lines += [label+" & "+" & ".join(vals)+r" \\"," & "+" & ".join(ses)+r" \\"]
    lines += [r"\hline",r"\multicolumn{7}{l}{\textit{Panel B: UTC-aligned non-overlapping five-minute windows}} \\"]
    for label,model,term in [("Total flow (N1)","N1","signed_log_total_net_flow"),("Whale flow (N2)","N2","signed_log_whale_net_flow"),("Non-whale flow (N2)","N2","signed_log_nonwhale_net_flow")]:
        a,b=cell(lookup(nonoverlap,model,term)); lines += [f"{label} & {a} &  &  &  &  & "+r" \\",f" & {b} &  &  &  &  & "+r" \\"]
    lines += [r"\hline",r"\multicolumn{7}{p{0.96\textwidth}}{\footnotesize Notes: Horizon models use 744,470 matched observations. Non-overlap models use 149,438 observations. All specifications include the frozen controls, market and calendar-hour fixed effects, and event-clustered CRV1 standard errors.}",r"\end{tabular}",r"\end{table}"]
    return "\n".join(lines)


def table3(h3: pd.DataFrame) -> str:
    models=["R_TOTAL","R_SPLIT","B5_SPLIT","B15_SPLIT","B5_15_SPLIT"]
    heads=["Subseq. total","Subseq. split","Brier 0--5","Brier 0--15","Brier 5--15"]
    terms=[("Total signed flow","signed_log_total_net_flow"),("Whale flow","signed_log_whale_net_flow"),("Non-whale flow","signed_log_nonwhale_net_flow"),("Absolute whale flow","log_abs_whale_net_flow"),("Absolute non-whale flow","log_abs_nonwhale_net_flow")]
    lines=[r"\begin{table}[htbp]",r"\centering",r"\caption{Persistence and forecast-improvement tests (H3)}",r"\label{tab:h3}",r"\scriptsize",r"\begin{tabular}{lccccc}",r"\hline"," & "+" & ".join(heads)+r" \\",r"\hline"]
    for label,term in terms:
        vals=[];ses=[]
        for m in models:
            rows=h3[(h3.model==m)&(h3.term==term)]
            if rows.empty:vals.append("");ses.append("")
            else:a,b=cell(rows.iloc[0]);vals.append(a);ses.append(b)
        lines += [label+" & "+" & ".join(vals)+r" \\"," & "+" & ".join(ses)+r" \\"]
    lines += [r"\hline",r"\multicolumn{6}{p{0.94\textwidth}}{\footnotesize Notes: Subsequent movement is the price change from minute 5 to minute 15, conditional on the initial five-minute change. Positive Brier improvement means movement closer to the resolved binary outcome. All models use 744,470 matched observations and event-clustered CRV1 standard errors.}",r"\end{tabular}",r"\end{table}"]
    return "\n".join(lines)


def table4(marg: pd.DataFrame, joint: pd.DataFrame) -> str:
    labels={"gt_24h":">24 hours","12_to_24h":"12--24 hours","6_to_12h":"6--12 hours","1_to_6h":"1--6 hours","last_60m":"Final 60 minutes"}
    lines=[r"\begin{table}[htbp]",r"\centering",r"\caption{Time-to-kickoff heterogeneity (H4)}",r"\label{tab:h4}",r"\small",r"\begin{tabular}{lccc}",r"\hline",r"Time to kickoff & Whale flow & Non-whale flow & Difference \\",r"\hline"]
    for band in labels:
        vals=[]
        for flow in ["whale","nonwhale","whale_minus_nonwhale"]:
            row=marg[(marg.band==band)&(marg.flow_type==flow)].iloc[0]
            vals.append(f"{row.estimate:.6f}{stars(row.p_value)} ({row.std_error:.6f})")
        lines.append(labels[band]+" & "+" & ".join(vals)+r" \\")
    w=joint[joint.test=="all_whale_timing_interactions_zero"].iloc[0]
    n=joint[joint.test=="all_nonwhale_timing_interactions_zero"].iloc[0]
    lines += [r"\hline",f"Joint interaction $F$ test & {w.f_statistic:.3f}*** & {n.f_statistic:.3f}*** & "+r" \\",
              r"\hline",r"\multicolumn{4}{p{0.92\textwidth}}{\footnotesize Notes: Entries are pooled-model implied coefficients with clustered standard errors in parentheses. The reference band is more than 24 hours. Joint tests have $(4,103)$ degrees of freedom. The timing pattern is non-monotonic.}",r"\end{tabular}",r"\end{table}"]
    return "\n".join(lines)


def table5(country: pd.DataFrame, diag: pd.DataFrame) -> str:
    specs=[("5m O1","primary_5m","O1"),("5m O2","primary_5m","O2"),("30m O1","matched_30m","O1"),("30m O2","matched_30m","O2")]
    terms=[("Total flow","signed_log_total_net_flow"),("Whale flow","signed_log_whale_net_flow"),("Non-whale flow","signed_log_nonwhale_net_flow")]
    lines=[r"\begin{table}[htbp]",r"\centering",r"\caption{Country outright-market regressions}",r"\label{tab:country}",r"\small",r"\begin{tabular}{lcccc}",r"\hline"," & "+" & ".join(x[0] for x in specs)+r" \\",r"\hline"]
    for label,term in terms:
        vals=[];ses=[]
        for _,sample,model in specs:
            rows=country[(country['sample']==sample)&(country.model==model)&(country.term==term)]
            if rows.empty:vals.append("");ses.append("")
            else:a,b=cell(rows.iloc[0]);vals.append(a);ses.append(b)
        lines += [label+" & "+" & ".join(vals)+r" \\"," & "+" & ".join(ses)+r" \\"]
    obs=[]
    for _,sample,model in specs: obs.append(int(diag[(diag['sample']==sample)&(diag.model==model)].observations.iloc[0]))
    lines += [r"\hline","Observations & "+" & ".join(f"{x:,}" for x in obs)+r" \\",r"Market fixed effects & Yes & Yes & Yes & Yes \\",r"Calendar-hour fixed effects & Yes & Yes & Yes & Yes \\",r"\hline",r"\multicolumn{5}{p{0.94\textwidth}}{\footnotesize Notes: Country markets are estimated separately from match contracts. CRV1 standard errors clustered by 48 country markets are in parentheses. Thirty-minute models use the matched 5/30-minute sample.}",r"\end{tabular}",r"\end{table}"]
    return "\n".join(lines)


def appendix_zero(match: pd.DataFrame, country: pd.DataFrame) -> str:
    rows=[]
    for market,label,data,models in [("Match","Match",match,["Z1","Z2","C1","C2"]),("Country","Country",country,["OZ1","OZ2","ON1","ON2"])]:
        for model in models:
            for r in data[(data.model==model)&data.term.str.contains("flow")].itertuples(): rows.append((label,model,r.term,r))
    lines=[r"\begin{table}[htbp]",r"\centering",r"\caption{Zero-outcome and non-overlap robustness}",r"\label{tab:appendix-zero}",r"\scriptsize",r"\begin{tabular}{llllrr}",r"\hline",r"Family & Model & Flow term & Estimate & SE & $p$-value \\",r"\hline"]
    for fam,model,term,r in rows:
        p="<0.001" if r.p_value<.001 else f"{r.p_value:.3f}"
        lines.append(f"{fam} & {model} & {term.replace('_',r'\_')} & {r.estimate:.6f} & {r.std_error:.6f} & {p} "+r"\\")
    lines += [r"\hline",r"\end{tabular}",r"\end{table}"]
    return "\n".join(lines)


def appendix_clusters(match: pd.DataFrame, country: pd.DataFrame) -> str:
    lines=[r"\begin{table}[htbp]",r"\centering",r"\caption{Alternative clustered inference}",r"\label{tab:appendix-clusters}",r"\scriptsize",r"\begin{tabular}{llllrrr}",r"\hline",r"Family & Model & Covariance & Flow term & Estimate & SE & $p$-value \\",r"\hline"]
    mf=match[match.term.str.contains("signed_log_.*net_flow",regex=True)]
    for r in mf.itertuples():
        p="<0.001" if r.p_value<.001 else f"{r.p_value:.3f}"
        term=r.term.replace("_",r"\_")
        covariance=r.covariance.replace("_",r"\_")
        lines.append(f"Match & {r.model} & {covariance} & {term} & {r.estimate:.6f} & {r.std_error:.6f} & {p} "+r"\\")
    cf=country[(country.family=="alternative_cluster") & country.term.str.contains("signed_log_.*net_flow",regex=True)]
    for r in cf.itertuples():
        p="<0.001" if r.p_value<.001 else f"{r.p_value:.3f}"
        term=r.term.replace("_",r"\_")
        covariance=r.covariance.replace("_",r"\_")
        lines.append(f"Country & {r.model} & {covariance} & {term} & {r.estimate:.6f} & {r.std_error:.6f} & {p} "+r"\\")
    lines += [r"\hline",r"\multicolumn{7}{p{0.96\textwidth}}{\footnotesize Notes: Match estimates use event, market, or event--calendar-hour CRV1 covariance. Country estimates use market, UTC-date, or market--date covariance. The country two-way covariance is a sensitivity check because one lagged-volume control variance is non-finite; all focal flow variances reported here are finite.}",r"\end{tabular}",r"\end{table}"]
    return "\n".join(lines)


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True); SECTIONS.mkdir(parents=True,exist_ok=True)
    primary=pd.read_csv(ROOT/"regression_results/v1/coefficients.csv")
    pdiag=pd.read_csv(ROOT/"regression_results/v1/model_diagnostics.csv")
    robust=pd.read_csv(ROOT/"robustness_results/v1/coefficients.csv")
    horizon=pd.read_csv(ROOT/"robustness_results/v1/horizon_coefficients.csv")
    h3=pd.read_csv(ROOT/"robustness_results/v1/h3_coefficients.csv")
    h4m=pd.read_csv(ROOT/"robustness_results/v1/h4_timing_marginal_effects.csv")
    h4j=pd.read_csv(ROOT/"robustness_results/v1/h4_timing_joint_tests.csv")
    country=pd.read_csv(ROOT/"regression_results/country_v1/coefficients.csv")
    cdiag=pd.read_csv(ROOT/"regression_results/country_v1/model_diagnostics.csv")
    crob=pd.read_csv(ROOT/"robustness_results/country_v1/coefficients.csv")
    minf=pd.read_csv(ROOT/"robustness_results/v1/inference_coefficients.csv")
    write("table_1_match_primary.tex",table1(primary,pdiag))
    write("table_2_match_robustness.tex",table2(horizon,robust))
    write("table_3_h3.tex",table3(h3))
    write("table_4_h4.tex",table4(h4m,h4j))
    write("table_5_country.tex",table5(country,cdiag))
    write("appendix_zero_nonoverlap.tex",appendix_zero(robust,crob))
    write("appendix_alternative_clustering.tex",appendix_clusters(minf,crob))
    results=r'''\section{Empirical Results}
\label{sec:results}

\subsection{Pre-match individual match markets}

Table~\ref{tab:match-primary} reports the primary estimates. Signed total flow is positively associated with the subsequent five-minute change in YES-equivalent probability in both the minimally adjusted model (P1) and the controlled model (P2). In P3, the whale-flow coefficient is positive and larger than the non-whale coefficient. A direct event-clustered test rejects their equality ($p<0.001$). These estimates support H1 and H2 as conditional associations, but the observational design does not identify a causal effect of trades on prices.

\input{tables/table_1_match_primary}

\subsection{Horizon and outcome-distribution robustness}

Table~\ref{tab:match-horizons} shows positive flow coefficients at one, five, and fifteen minutes on a common matched sample. The whale coefficient exceeds the non-whale coefficient at every horizon. Results remain positive when only UTC-aligned, non-overlapping five-minute windows are retained. Because 96.31\% of primary five-minute outcomes are zero, Appendix Table~\ref{tab:appendix-zero} separately models price-update incidence and conditional non-zero movement. These checks show that the primary association is not eliminated by the mass at zero or by overlapping response windows.

\input{tables/table_2_match_robustness}

\subsection{Persistence and forecast improvement}

Table~\ref{tab:h3} distinguishes continued price movement from ex-post forecast improvement. Signed flow is positively associated with movement from minute 5 to minute 15 after controlling for the initial response, which is consistent with continuation rather than average reversal. The Brier-improvement coefficients are not statistically distinguishable from zero. H3 therefore receives mixed evidence: whale-associated movement persists over the measured interval, but the analysis does not show that larger whale flow improves alignment with the resolved outcome.

\input{tables/table_3_h3}

\subsection{Time-to-kickoff heterogeneity}

The joint tests in Table~\ref{tab:h4} reject equality of the flow association across timing bands for both whale and non-whale flow. Whale coefficients remain positive in every band, but the pattern is non-monotonic: estimates are larger more than 24 hours before kickoff and during the final hour, and smaller between 6 and 12 hours before kickoff. The pooled final-hour non-whale coefficient is negative, but its sign is not stable in the separately stratified model and is therefore treated as specification-sensitive. H4's timing component is supported, but the estimates do not support a simple claim that the association increases continuously as kickoff approaches.

\input{tables/table_4_h4}

\subsection{Country outright markets}

Table~\ref{tab:country} presents the separately estimated secondary outright analysis. Five-minute total, whale, and non-whale flow coefficients are positive, and the whale coefficient exceeds the non-whale coefficient ($p<0.001$ for the difference). At thirty minutes, total and whale flow remain positive, while the non-whale coefficient is not statistically significant at the 5\% level. The match and outright estimates point in the same qualitative direction, but coefficient magnitudes are not directly comparable because the information environments, controls, clustering structures, and market horizons differ.

\input{tables/table_5_country}

\subsection{Inference and interpretation}

The match-market conclusions are unchanged under market clustering and two-way event--calendar-hour clustering. Country results are also stable under market and UTC-date clustering and on a non-overlapping grid. The country market--date two-way covariance is retained only as a sensitivity check because the variance estimate for one lagged-volume control is non-finite, although all focal flow variances remain finite. Across all analyses, statistical significance is interpreted as evidence of conditional association rather than private information, manipulation, or a market-integrity violation.

'''
    (SECTIONS/"results.tex").write_text(results,encoding="utf-8")
    appendix=r'''\section{Regression Robustness Appendix}
\label{sec:regression-appendix}

Appendix Table~\ref{tab:appendix-zero} reports the zero-outcome, conditional non-zero, and non-overlapping-window checks for match and country markets. Appendix Table~\ref{tab:appendix-clusters} reports alternative covariance estimators. These models supplement rather than replace the frozen primary specifications.

\input{tables/appendix_zero_nonoverlap}

\input{tables/appendix_alternative_clustering}
'''
    (SECTIONS/"appendix_robustness.tex").write_text(appendix,encoding="utf-8")
    print(f"Built 7 tables, Results, and robustness appendix")
    return 0


if __name__=="__main__": raise SystemExit(main())
