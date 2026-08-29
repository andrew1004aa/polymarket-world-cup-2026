# Supervisor consistency correction v7

Date: 2026-08-29

This revision corrects interpretation and documentation without changing any
frozen empirical result. The signed-log regressor is
`sign(F) * log(1 + abs(F))`, so multiplying a coefficient by `log(10)` is not
an exact tenfold-flow effect. The dissertation instead reports the verified
one-standard-deviation effect from
`regression_results/v6/threshold_sensitivity/standardised_and_common_value_effects.csv`:
0.0224 percentage points for expanding-P99 large flow and 0.0059 percentage
points for ordinary flow.

Inspection of `scripts/run_h3_regressions.py`, the v3 H3 configuration, model
diagnostics, and frozen coefficient files confirmed that primary H3a uses the
unadjusted dependent variable `p[t+15] - p[t+5]`. Signed large- and
ordinary-flow regressors encode direction. The methodology and Table 3 note
were aligned with that implementation. No H3a rerun was required and no
coefficient, p-value, sample size, table value, figure, or frozen model output
was changed.

H1 is now described as a baseline sanity check for the directional-flow
measure, rather than as formally validating the construction. The intended new
submission tag is `v1.0.4-submission-final`; its full version identifiers are
recorded using a two-commit workflow. The substantive content commit is
`1a1e3fbde8541a930416adbd6faa7dde9aa6a1fa`. The tag points to the subsequent
metadata-only commit that inserts this SHA into the dissertation and release
documentation, avoiding the impossible claim that a commit contains its own
identifier.
