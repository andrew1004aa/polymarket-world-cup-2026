# V5 Event-time Fixed Effects and Binary-model Results

## Event-by-time fixed effects

The five-minute directional large-minus-ordinary contrast remained positive
after absorbing common information within each match-time cell:

| P99 definition | Event × minute | Event × 5-minute |
|---|---:|---:|
| Expanding history | 0.000102 (p < 0.001) | 0.000074 (p < 0.001) |
| Rolling seven day | 0.000099 (p < 0.001) | 0.000073 (p < 0.001) |

Event-by-minute identification used 227,973 expanding and 191,320 rolling
market-minutes. Event-by-five-minute identification used 271,155 and 224,609.
Single-market cells were excluded because they contain no cross-contract
variation. Depending on the P99 support rule, the identifying samples contained
87 expanding and 78 rolling match events.

The result is not explained entirely by information common to the related
home-win, draw, and away-win contracts within the same match-time cell. It
remains a conditional association rather than a causal estimate.

## Binary price-update robustness

Exact calendar-hour and complementary log-log specifications did not produce a
finite stable maximum-likelihood solution because the outcome is extremely
sparse. Their estimates were rejected by numerical QC and are not reported.

The stable market-fixed-effects logit models converged in 10 iterations
(expanding) and 9 iterations (rolling). Markets without within-market outcome
variation were excluded, leaving 417,363 and 364,854 observations.

Both large and ordinary absolute flow were positively associated with the odds
of any five-minute price update:

- Expanding: large 0.1717; ordinary 0.1715; both p < 0.001.
- Rolling: large 0.1651; ordinary 0.1755; both p < 0.001.

However, the large-minus-ordinary contrasts were not significant:

- Expanding: 0.00025, p = 0.986.
- Rolling: -0.01040, p = 0.511.

The non-linear robustness therefore supports a positive association between
flow magnitude and price-update incidence, but does **not** reproduce the LPM
claim that large flow has a stronger extensive-margin association than ordinary
flow. The differential extensive-margin conclusion must consequently be
presented as model-sensitive. This does not overturn the directional
price-change result, which uses a continuous outcome and survives event-time
fixed effects.

## Reproducibility

- Frozen specification: `docs/research_design/v5_event_time_and_binary_specification.md`
- Event-time runner: `scripts/run_v5_event_time_fe.py`
- Event-time results: `regression_results/v5/event_time_fe/`
- Binary runner: `scripts/run_v5_binary_fe_models.py`
- Stable logit results: `regression_results/v5/binary_fe/`

