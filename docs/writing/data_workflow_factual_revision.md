# Data-workflow factual revision

Date: 2026-08-27

The dissertation treats the researcher-confirmed workflow as authoritative:
the Polymarket Data API is the sole canonical trade-level source, while Dune
Analytics is used only for independent market-level count validation. Dune
observations are not merged into, substituted for, or used to modify the
canonical sample.

The API returned 6,725,954 in-window rows. Canonical serialisation and SHA-256
equality of complete API objects identified 222 cross-checkpoint repeated
deliveries across 39 markets, leaving 6,725,732 retained records. Independently
obtained Dune Analytics market-level counts were used only to validate
extraction completeness: 350 of 360 counts matched exactly and Dune was one
record higher in each of ten markets.

The exact-response procedure is not transaction-hash deduplication and does not
aggregate inferred economic trades, orders, or executions. All retained records
have unique non-empty transaction hashes, but hashes were not the canonical
deduplication key. Collection requests set `takerOnly=true` explicitly.

Section 3.1 was rewritten to separate canonical data construction from external
count validation, and to distinguish the trade, CLOB price-history, and FIFA
fixture/outcome sources. Section 4.10 was limited to verified repository
contents. The Abstract and Conclusion were aligned with the same terminology.

The current repository contains a README, stage-specific dependency files, a
data dictionary and collection report, principal v5/v6 output files, SHA-256
manifests with 360 trade and 360 price entries, and the complete LaTeX source.
The fixed final submission snapshot is identified by the tag
`v1.0.6-submission-final`.
