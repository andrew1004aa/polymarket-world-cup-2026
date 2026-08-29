# V5 Pre-trade Placebo and Post-trade Lead–Lag Results

The horizon set and evaluation rule were frozen in
`docs/research_design/v5_lead_lag_preregistration.md` before estimation.

## Finding

The large-minus-ordinary flow contrast was effectively zero at all three
pre-trade horizons. None of the -15, -5, or -1 minute placebo estimates was
statistically significant under either the expanding-history or rolling-seven-
day P99 definition. Their Holm-adjusted p-values were all 1.000.

The contrast became positive immediately after the flow window:

| Horizon | Expanding estimate | Expanding Holm p | Rolling estimate | Rolling Holm p |
|---:|---:|---:|---:|---:|
| -15 | -0.0000005 | 1.000 | -0.0000028 | 1.000 |
| -5 | 0.0000009 | 1.000 | 0.0000004 | 1.000 |
| -1 | 0.0000001 | 1.000 | -0.0000007 | 1.000 |
| +1 | 0.0000581 | <0.001 | 0.0000580 | <0.001 |
| +5 | 0.0000815 | <0.001 | 0.0000814 | <0.001 |
| +15 | 0.0001078 | <0.001 | 0.0001052 | <0.001 |
| +30 | 0.0001186 | 0.039 | 0.0001210 | 0.067 |

The +1, +5, and +15 minute results survive Holm adjustment under both past-only
P99 definitions. At +30 minutes, the expanding result passes the 5% threshold
but the rolling result does not. The positive association at the 30-minute
horizon is therefore not treated as robust across definitions.

## Interpretation

The absence of corresponding pre-flow coefficients weakens the specific concern
that the focal association merely reflects a price movement already completed
before the observed flow minute. The positive post-flow path is consistent with
short-horizon price adjustment following relatively large trades. It does not,
however, establish causality: contemporaneous private information or other
unobserved trading signals may still jointly determine order flow and subsequent
prices.

Coefficient magnitudes are price-probability units per one-unit difference in
signed-log net flow. For example, 0.000058 corresponds to 0.0058 percentage
points, so the estimated economic magnitude remains small despite precise
statistical inference.

## Reproducibility

- Panel builder: `scripts/build_v5_lead_lag_panel.py`
- Model runner: `scripts/run_v5_lead_lag.py`
- Panel: `regression_inputs/v5/lead_lag/lead_lag_panel.csv.gz`
- Results: `regression_results/v5/lead_lag/lead_lag_contrasts.csv`
