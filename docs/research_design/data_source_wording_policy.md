# Canonical data-source wording policy

Status: approved by the researcher on 2026-08-27.

Trade-level observations used in the empirical analysis were collected
directly from the Polymarket Data API and are the sole canonical trade-level
dataset. Independently obtained Dune Analytics market-level trade counts were
used only to validate extraction completeness. Dune observations were not
merged into, substituted for, or used to modify the canonical Polymarket API
sample.

Public-facing dissertation and GitHub materials do not report the number or
identity of markets with count differences, or the aggregate size of any count
difference. Local raw QC evidence may be retained for audit but is not part of
the public methodology narrative unless the researcher explicitly changes this
policy.
