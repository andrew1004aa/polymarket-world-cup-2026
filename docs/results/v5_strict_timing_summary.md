# V5 Strict Price-Timing Summary

## Timing convention

- Baseline price: latest observed YES price strictly before minute `t`.
- Flow: trades timestamped in `[t, t+1 minute)`.
- Post-flow price: first observed YES price at or after `t+h`.
- Strict-0: baseline and outcome gaps are each below 60 seconds.
- Strict-1: baseline and outcome gaps are each below 120 seconds.
- No interpolation, resampling, or gap filling was used.

## Coverage

The source contains 748,481 market-minute observations from 312 match markets
and 104 events. Strict-0 retained 748,372 observations at the five-minute
horizon; strict-1 retained 748,420. Strict-0 therefore retained 99.985% of the
source observations. The five-minute zero-change rate was 96.238% under both
rules. Above-median total prematch volume identified 156 active markets.

The corrected five-minute change differed from the earlier timing convention in
11,917 rows. This confirms that the timing correction is substantive even
though the aggregate zero-change rate remains similar.

## H2 robustness result

Across expanding-history and rolling-seven-day past-only P99 definitions, and
across strict-0, strict-1, and strict-0 active-market samples, the estimated
large-minus-ordinary flow contrast remained positive.

For the strict-0 full sample:

- directional five-minute price change: 0.000082 (expanding) and 0.000081
  (rolling-seven-day);
- probability of any five-minute update: 0.006406 and 0.006052;
- conditional non-zero price-change magnitude: 0.000210 and 0.000194.

The corresponding event-clustered tests reject equality of the large- and
ordinary-flow coefficients. The estimates represent conditional associations,
not causal effects. The conditional non-zero model is outcome-selected and is
reported as descriptive intensive-margin evidence.

## Reproducibility

- Panel builder: `scripts/build_v5_strict_timing_panel.py`
- Model runner: `scripts/run_v5_strict_timing_h2.py`
- Timing panel: `regression_inputs/v5/strict_timing/strict_timing_past_only_p99.csv.gz`
- Coverage: `regression_inputs/v5/strict_timing/strict_sample_coverage.csv`
- Results: `regression_results/v5/strict_timing_h2/`

