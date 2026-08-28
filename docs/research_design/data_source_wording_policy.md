# Canonical data-source wording policy

Status: approved by the researcher on 2026-08-27.

Trade-level observations used in the empirical analysis were collected
directly from the Polymarket Data API and are the sole canonical trade-level
dataset. Independently obtained Dune Analytics market-level trade counts were
used only to validate extraction completeness. Dune observations were not
merged into, substituted for, or used to modify the canonical Polymarket API
sample.

The final public methodology reports the verified construction and validation
facts: 6,725,954 in-window API rows, 222 cross-checkpoint repeated complete
objects removed, 6,725,732 retained records, and the 350-exact/10-one-record
Dune count reconciliation. It distinguishes exact-response removal from
transaction-hash deduplication and economic-trade aggregation.
