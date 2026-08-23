# V5 Event-time Fixed Effects and Binary-model Specification

This specification is frozen before estimation.

## Event-time fixed effects

The directional five-minute outcome is re-estimated using (a) event-by-minute
and (b) event-by-five-minute fixed effects together with market fixed effects.
Identification therefore comes from differences across the related home-win,
draw, and away-win contracts for the same match during the same time cell.
Cells represented by only one market contain no identifying cross-contract
variation and are excluded before estimation. Standard errors remain clustered
by match event.

The strict-0 price sample is used. Expanding-history P99 is primary and rolling-
seven-day P99 is the definition robustness. The comparison statistic remains the
large-minus-ordinary signed-log flow coefficient.

These models absorb common match-specific information arriving within the time
cell. They do not absorb contract-specific information and do not establish
causality.

## Binary price-update models

The dependent variable equals one when the strictly aligned five-minute price
change is non-zero. The existing linear-probability model is the baseline.
Fixed-effects logit and complementary log-log models are estimated on the same
strict-0 sample, with market and UTC calendar-hour fixed effects and the same
controls. CRV1 inference is clustered by match event.

If the exact UTC calendar-hour specification has no finite maximum-likelihood
solution because of complete or quasi-complete separation, it is recorded as a
failed diagnostic and is not interpreted. The pre-defined stable fallback uses
market fixed effects plus separate UTC-date and UTC-hour-of-day fixed effects.
Markets with no within-market outcome variation are excluded because they carry
no finite binary-response slope information.

If separation remains under that fallback, the final estimable robustness uses
market fixed effects only, retaining the complete set of continuous controls and
event-clustered inference. The failed richer non-linear specifications are not
interpreted; common event-time shocks are assessed separately in the linear
event-by-time models above.

The non-linear robustness is considered supportive when the large-flow and
ordinary-flow coefficients retain the same ordering as in the LPM. Coefficients
across different links are not compared numerically as if they shared the same
scale; average marginal effects are reported for economic interpretation.
