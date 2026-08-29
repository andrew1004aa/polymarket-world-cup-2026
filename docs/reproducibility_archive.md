# Fixed reproducibility archive contents

The `v1.0.3-submission-final` tag is the fixed repository snapshot corresponding
to the submitted dissertation source. It contains the complete LaTeX source,
collection and analysis scripts, stage-specific dependency files, data
dictionary, collection report, compact principal v5/v6 model outputs, and
SHA-256 manifests for the 360 local trade partitions and 360 local price
partitions.

Large source, regression-ready and model-input files are not redistributed in
Git because of their size. The hash manifests identify the local files used in
the final analysis, but hashes alone do not permit independent replication
without first recollecting the public-source data. The README documents the
collection and construction sequence. No historical debugging artifact is a
required input merely because it remains in development storage.

The canonical trade source is the Polymarket Data API. Dune queries `8139394`
and `8140259` are market-level count-validation sources only; query `8142697`
is an auxiliary 104-match mapping. Prices come from the Polymarket CLOB
`/prices-history` endpoint, and fixtures/events originate in the official FIFA
`world-cup_2026.xlsx` workbook.
