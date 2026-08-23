# V5 Pre-trade Placebo and Post-trade Lead–Lag Specification

This specification is frozen before estimation.

## Question

Does the large-minus-ordinary flow association appear only after the focal
market-minute, or is a similar association already present in price changes
completed before that minute?

## Timing

- Flow is measured during `[t,t+1 minute)`.
- The reference price is the latest price strictly before `t`.
- Pre-trade outcomes are cumulative changes ending at the reference price over
  `[-15,0)`, `[-5,0)`, and `[-1,0)` minutes.
- Post-trade outcomes are cumulative changes from the reference price to the
  first price at or after `t+h`, for `h = 1, 5, 15, 30` minutes.
- Both endpoints must satisfy strict-0 alignment: an absolute timing gap below
  60 seconds. No interpolation, resampling, or gap filling is allowed.

## Estimation

Each horizon is estimated separately. The regressors of interest are signed-log
large and ordinary net flow. Controls are the reference price, lagged 30-minute
price change, lagged 30-minute volume, and minutes to kickoff. Market and UTC
calendar-hour fixed effects are absorbed. CRV1 standard errors are clustered by
the 104 match events.

The primary large-trade classification is the market-specific expanding-history
past-only P99. Rolling-seven-day past-only P99 is a definition robustness check.

## Evaluation rule

The focal statistic is the large-minus-ordinary coefficient contrast at every
pre-specified horizon. Negative-horizon estimates are placebo/diagnostic
associations and cannot be effects of the later flow. Positive-horizon estimates
are interpreted as conditional post-flow associations, not causal effects.

Evidence consistent with temporal ordering requires the post-flow contrast to
be stronger and more persistent than the pre-flow contrasts. Significant
negative-horizon estimates instead indicate anticipation, price-chasing, shared
information, or remaining timing confounding.

