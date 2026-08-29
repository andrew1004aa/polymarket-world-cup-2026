# Dissertation empirical release v5

Release status: **analysis complete; empirical outputs frozen**  
Release date: 2026-08-22

## Canonical data statement

Trade-level observations were collected directly from the Polymarket Data API and constitute the sole canonical trade-level dataset. After collection, independently obtained Dune Analytics market-level trade counts were used solely to assess the completeness of the API extraction. No Dune observations were incorporated into or used to modify the canonical dataset. The canonical sample contains 6,725,732 retained Data API trade records across 360 markets.

## Primary empirical definition

- Unit: match-market minute.
- Large trade: same-market past-only P99 of trade USDC value, with ties included.
- Primary threshold: expanding history through the previous complete UTC day, requiring at least 1,000 prior trades.
- Robustness threshold: preceding seven complete UTC days, requiring at least 1,000 prior trades.
- Strict-0 price rule: latest price strictly before the interval; signed flow in `[t,t+1)`; first price at or after the target; both alignment gaps below 60 seconds; no interpolation.
- Inference: event-clustered covariance; Holm adjustment for the lead--lag family.

## Frozen primary results

| Result | Expanding | Rolling seven-day | Release interpretation |
|---|---:|---:|---|
| H2 five-minute large-minus-ordinary contrast | 0.000082 | 0.000081 | Positive, statistically significant, economically small |
| Any-update LPM contrast | 0.006406 | 0.006052 | Positive and statistically significant |
| Market-FE logit large-minus-ordinary contrast | 0.000254 (`p=0.986`) | -0.010396 (`p=0.511`) | Extensive-margin difference not reproduced |
| Lead placebos (-15, -5, -1 minutes) | Holm-adjusted `p=1` | Holm-adjusted `p=1` | No detectable pre-trend |
| Post path (+1, +5, +15 minutes) | Positive, Holm `<0.001` | Positive, Holm `<0.001` | Robust short-horizon temporal ordering |
| +30 minutes | Holm `p=0.0385` | Holm `p=0.0674` | Not robust across definitions |
| Event-by-minute / event-by-five-minute FE | Positive, `p<0.001` | Positive, `p<0.001` | Remains positive under shared event-time controls |

## Canonical v5 inputs

- `regression_inputs/v5/past_only_p99/past_only_p99_primary_sample.csv.gz`
- `regression_inputs/v5/past_only_p99/daily_thresholds.csv.gz`
- `regression_inputs/v5/strict_timing/strict_timing_past_only_p99.csv.gz`
- `regression_inputs/v5/lead_lag/lead_lag_panel.csv.gz`

## Canonical v5 results

- `regression_results/v5/past_only_p99_h2/`
- `regression_results/v5/strict_timing_h2/`
- `regression_results/v5/lead_lag/`
- `regression_results/v5/event_time_fe/`
- `regression_results/v5/binary_fe/`

## Audit and interpretation documents

- `docs/data_audit/v5/ml_threshold_leakage_audit.json`
- `docs/results/v5_strict_timing_summary.md`
- `docs/results/v5_lead_lag_summary.md`
- `docs/results/v5_event_time_and_binary_summary.md`
- `docs/results/v5_ml_leakage_audit_summary.md`
- `docs/research_design/supervisor_revision_v5_specification.md`

## Thesis files updated for v5

- `overleaf/main.tex`
- `overleaf/sections/abstract.tex`
- `overleaf/sections/introduction.tex`
- `overleaf/sections/data.tex`
- `overleaf/sections/research_design.tex`
- `overleaf/sections/results.tex`
- `overleaf/sections/discussion.tex`
- `overleaf/sections/conclusion.tex`
- `overleaf/sections/impact_statement.tex`
- `overleaf/sections/declaration.tex`
- `overleaf/sections/appendix_wallet_methods.tex`
- `overleaf/tables/table_v5_primary.tex`
- `overleaf/tables/table_v5_leadlag.tex`
- `overleaf/tables/table_v5_mechanism.tex`
- `overleaf/tables/table_hypothesis_decisions.tex`

## Interpretation boundary

The supervised machine-learning outcome is thirty-minute full price reversal, not manipulation. Unsupervised scores prioritise unusual observations for contextual review and do not constitute verified misconduct labels. All regression results are conditional associations rather than causal structural price-impact estimates.

## Submission consistency note

The final IFTE0008-method main-text count is 10,376 words, within the
supervisor-approved 9,000--11,000 range. It excludes the title page, signed
declaration, contents and lists, glossary/abbreviations, abstract, equations,
code, tables/figures/graphs, footnotes, references and appendices. The submitted
source intentionally excludes the Declaration. Compile `overleaf/main.tex` in
Overleaf and perform final visual QA without changing the frozen empirical
values or the canonical data statement above. The immutable repository snapshot
is identified by `v1.0.4-submission-final`.
