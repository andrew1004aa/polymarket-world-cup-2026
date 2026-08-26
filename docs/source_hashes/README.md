# Source-data hash manifests

The raw and regression-ready trade and price partitions are too large for this
repository. These manifests identify the exact local files used by the submitted
analysis without redistributing them.

- `trade_files_sha256.csv`: 360 canonical market trade partitions, with row
  counts, timestamp bounds, source hashes and regression-ready file hashes.
- `price_files_sha256.csv`: 360 market price partitions, with row counts,
  timestamp bounds and SHA-256 hashes.

After reproducing the public-API collection, compare each generated file with
the corresponding `sha256` value. The manifests contain 360 trade entries and
360 price entries. Static regression-ready tables and their QC status are
documented separately by the construction logs and release results.
