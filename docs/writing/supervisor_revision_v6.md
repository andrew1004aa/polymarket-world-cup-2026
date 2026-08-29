# Supervisor revision v6

Generated: 2026-08-25

## Implemented

- Documented the canonical Polymarket Data API collection: 1,000-row pages,
  recursive time-window subdivision, half-open intervals, checkpoints, complete
  API provenance and the independent Dune market-level count-validation role.
  Development-error counts are not part of the dissertation narrative.
- Clarified `side`, the endpoint's default `takerOnly=true` setting, and why
  transaction hash is not treated as a fill-level unique key.
- Added the full GitHub URL using `xurl` for A4-safe wrapping; the submitted
  source cites the fixed `v1.0.4-submission-final` release tag.
- Described CLOB outcomes as recorded prices and added an executed-trade-price
  robustness test that excludes the focal minute from the baseline.
- Reported strict timestamp-gap distributions.
- Added P95, P97.5, P99 and P99.5 past-only thresholds under expanding and
  rolling histories, with signed-log, `asinh`, and winsorised raw-flow models.
- Added standardised and common-$1,000 flow effects.
- Compared the 252/236 included H2 markets with the 60/76 excluded markets and
  restricted the conclusion to sufficiently active markets with adequate prior
  history.
- Reframed H3a as minute-5-to-minute-15 subsequent continuation and added a
  descriptive outcome-selected check conditional on a non-zero initial move.
- Added fixed-effects-logit average marginal effects.
- Added market, date, and two-way market/date covariance for country markets;
  the country analysis remains a secondary extension.
- Replaced capital-deployment language with cumulative gross trading volume.
- Added a 60-second execution-sequence consolidation check for wallet
  continuation.
- Reported both 73 case-row and 35 linked/standalone-unit audit dispositions.

## Principal new results

- All 24 threshold/transformation models produce a positive and significant
  large-minus-ordinary contrast (`p<0.001`).
- Executed-price contrasts remain positive at 60, 300 and 900 seconds under
  both past-only definitions (0.000086–0.000092; all `p<0.001`).
- Conditional-initial-update H3a contrasts are positive but insignificant:
  0.000108 (`p=0.161`) expanding and 0.000085 (`p=0.317`) rolling.
- Country five-minute contrast is 0.000074 with market, date and two-way
  covariance (all `p<0.001`).
- Fixed-effects-logit AME differences are 0.000008 expanding and -0.000349
  rolling, confirming no robust extensive-margin differential.
- Thirty-five audit units comprise 15 event-information, 11 not-anomalous,
  eight unresolved and one insufficient-data unit. No audited case establishes
  manipulation.
- Consolidating transaction actions into 60-second execution sequences reduces
  2,395,771 actions to 1,844,222 sequences. Ordinary-sequence continuation
  remains significant for both high-activity and whale-volume classifications;
  large-sequence same-direction continuation is not robust.

## Word-count check

The final IFTE0008-method count is 10,410 words for the seven main chapters.
The count excludes the title page and signed declaration, contents and lists,
glossary/abbreviations, abstract, equations, code, tables/figures/graphs,
footnotes, references and appendices, and is within the supervisor-approved
9,000--11,000 range.

## Reproducibility outputs

Machine-readable v6 results are under `regression_results/v6/`. The public
release contains the scripts, compact results, data dictionary, collection
report, dependency pins, and SHA-256 manifests for all 360 trade and 360 price
partitions.
