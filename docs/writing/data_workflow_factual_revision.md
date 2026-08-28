# Data-workflow factual revision

Date: 2026-08-27

The dissertation treats the researcher-confirmed workflow as authoritative:
the Polymarket Data API is the sole canonical trade-level source, while Dune
Analytics is used only for independent market-level count validation. Dune
observations are not merged into, substituted for, or used to modify the
canonical sample.

The canonical sample contains 6,725,732 Polymarket Data API rows across 360
markets. Independently obtained Dune Analytics market-level counts were used
only to validate extraction completeness. Public dissertation and repository
materials do not report market-specific reconciliation figures.

No additional canonical trade-level deduplication based on transaction hash or
a synthetic trade identifier is part of the final research workflow. Historical
reports or manifests that retain temporary duplicate-development fields or
abandoned extraction details are development artifacts and are not treated as
the dissertation's methodology.

Section 3.1 was rewritten to separate canonical data construction from external
count validation, and to distinguish the trade, CLOB price-history, and FIFA
fixture/outcome sources. Section 4.10 was limited to verified repository
contents. The Abstract and Conclusion were aligned with the same terminology.

The current repository contains a README, stage-specific dependency files, a
data dictionary and collection report, principal v5/v6 output files, SHA-256
manifests with 360 trade and 360 price entries, and the complete LaTeX source.
The fixed final submission snapshot is identified by the tag
`v1.0.0-submission`.
