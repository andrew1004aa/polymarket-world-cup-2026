# Polymarket World Cup 2026

Reproducibility repository for the MSc dissertation **Large Trades and
Short-Horizon Price Adjustment on Polymarket: Evidence from the 2026 FIFA World
Cup**.

This repository contains the code used to collect the data, construct the
analysis samples, estimate the regressions, run the machine-learning and market
integrity exercises, and compile the dissertation. It is intended to document
the research workflow without reproducing every API call step-by-step in the
dissertation itself.

## Study scope and canonical sample

- Observation window: `2026-06-01 00:00:00 UTC` (inclusive) to
  `2026-08-01 00:00:00 UTC` (exclusive).
- Markets: 360 binary Polymarket contracts: 312 match contracts corresponding
  to 104 FIFA World Cup matches, plus 48 country outright-winner contracts.
- Trades: 6,725,954 in-window Data API rows before exact-response checking;
  222 repeated complete API objects removed; 6,725,732 retained Data API trade
  records in the canonical research sample.
- Prices: 17,980,605 raw price-history observations for 720 outcome tokens at
  the highest available one-minute API fidelity, without interpolation,
  resampling, or gap filling.

Trade-level observations were collected directly from the Polymarket Data API
and constitute the sole canonical trade-level dataset. After collection,
independently obtained Dune Analytics market-level trade counts were used solely
to assess the completeness of the API extraction. No Dune observations were
incorporated into or used to modify the canonical dataset.

Exact-response checking serialises each complete API object in canonical key
order and compares its SHA-256 digest. The 222 removed responses were
cross-checkpoint repeated deliveries affecting 39 markets; no same-checkpoint
duplicates were found. This is not transaction-hash deduplication or aggregation
into inferred economic trades, orders, or executions. All 6,725,732 retained
records have unique non-empty transaction hashes. Collection requests now set
`takerOnly=true` explicitly.
On a matched ten-record validation page, the default and explicit-parameter
responses were byte-identical and had the same SHA-256 digest.

Dune query `8139394` supplied validation counts for the 312 match contracts and
query `8140259` for the 48 country outright contracts. The local exports were
downloaded on 31 July 2026 at 13:52 BST. Across 360 markets, 350 counts matched
exactly and Dune was one record higher in each of ten markets: API total
6,725,732; Dune total 6,725,742. Dune contributed no trade-level regression
observations. Query `8142697` was used only for the auxiliary 104-match mapping
based on the official `world-cup_2026.xlsx` fixture source. Price histories come
separately from the Polymarket CLOB `/prices-history` endpoint.

The fixed final submission version is
[`v1.0.4-submission-final`](https://github.com/andrew1004aa/polymarket-world-cup-2026/tree/v1.0.4-submission-final).

## Repository map

| Path | Purpose |
|---|---|
| `scripts/` | Collection, construction, regression, ML, integrity, robustness, and reporting code |
| `scripts/README.md` | Exact script map and recommended execution order |
| `overleaf/` | Complete LaTeX source for the dissertation |
| `regression_results/v5/` | Frozen v5 coefficient files and diagnostics |
| `regression_results/v6/` | Supervisor-requested sensitivity and audit diagnostics |
| `docs/source_hashes/` | SHA-256 manifests for the local canonical trade and price partitions |
| `docs/data_dictionary.md` | Raw and derived variable definitions |
| `docs/data_collection_report.md` | Collection coverage, limitations, and checkpoints |
| `docs/research_design/` | Frozen empirical specifications |
| `docs/results/` | Final result summaries and release manifest |
| `requirements-*.txt` | Stage-specific Python dependencies |

Large raw and derived files are deliberately excluded from Git because of their
size. Running the collection and construction scripts creates the ignored
directories listed in `.gitignore`. The frozen, compact v5 and v6 result
artifacts are included so that reported coefficients can be checked without
downloading the full dataset. Because the large source and regression-ready
files are not distributed here, the repository documents and audits the final
workflow but does not claim one-command independent replication from hashes
alone.

## Environment setup

Python 3.11 or 3.12 is recommended. Use separate environments for collection,
estimation, and machine learning.

```bash
git clone https://github.com/andrew1004aa/polymarket-world-cup-2026.git
cd polymarket-world-cup-2026

python -m venv .venv-collection
source .venv-collection/bin/activate
python -m pip install -r requirements-collection.txt
```

For regression or ML stages, activate a fresh environment and install the
corresponding file:

```bash
python -m pip install -r requirements-regression.txt
# or
python -m pip install -r requirements-ml.txt
```

The supplemental wild-cluster inference environment is pinned separately in
`requirements-inference-robustness.txt` because its NumPy/Numba constraints can
differ from the other environments.

## Reproduction workflow

All commands below are run from the repository root. Scripts resolve paths
relative to that root.

### 1. Discover markets

```bash
python -u scripts/discover_world_cup_markets.py
```

This stores unchanged Gamma API metadata and creates the market inventory. The
final whitelist contains only match winner/draw contracts and country outright
winner contracts within the stated research scope.

### 2. Download trade-level data

```bash
caffeinate -i python -u scripts/fetch_world_cup_data_api.py \
  --start 2026-06-01T00:00:00Z \
  --end 2026-08-01T00:00:00Z
```

On macOS, `caffeinate` prevents sleep and may be omitted elsewhere. The script
uses time-window pagination, writes every successful response to a compressed
checkpoint, and resumes from its progress file after interruption. To rebuild
the CSV from existing checkpoints without calling the API again:

```bash
python -u scripts/fetch_world_cup_data_api.py \
  --start 2026-06-01T00:00:00Z \
  --end 2026-08-01T00:00:00Z \
  --build-only
```

The collector sends `takerOnly=true` explicitly. Canonical construction keeps
the first complete API object and removes only later byte-equivalent complete
objects identified by canonical serialisation and SHA-256 hashing; it does not
deduplicate by transaction hash or aggregate inferred executions.

### 3. Partition trades and build wallet tables

```bash
python -u scripts/partition_trades_by_market.py
python -u scripts/build_wallet_tables.py
```

These commands create one trade file per market and aggregate `wallets.csv`
and `wallet_market.csv`. Wallet labels are not inferred at collection time.

The event mapping additionally requires the official fixture workbook. Save it
as `source/world-cup_2026.xlsx`, install the small Node dependency, and run:

```bash
npm install
node scripts/build_events_table.mjs source/world-cup_2026.xlsx
```

The workbook times are interpreted as Europe/London summer time (BST, UTC+1).

### 4. Download raw token price histories

```bash
caffeinate -i python -u scripts/fetch_price_history.py \
  --start 2026-06-01T00:00:00Z \
  --end 2026-08-01T00:00:00Z \
  --fidelity 1
```

There are 720 token histories (YES and NO for each market). Responses are
checkpointed and resumable. Use `--build-only` to reconstruct `prices.csv` from
completed checkpoints.

### 5. Construct the regression-ready data

The construction stage creates market-level partitions, event mappings,
market-minute panels, trade-size thresholds, behavioural-event samples, and ML
samples. The final v5 sequence is:

```bash
python scripts/build_v5_past_only_p99_sample.py
python scripts/build_v5_strict_timing_panel.py
python scripts/build_v5_lead_lag_panel.py
```

The v5 primary large-trade definition is the same-market, past-only P99 of USDC
trade value. The primary threshold uses expanding history through the previous
complete UTC day and requires at least 1,000 prior trades. A preceding
seven-complete-day definition is retained as robustness. Ties are included.

Earlier construction scripts required to create the inputs consumed by these
three commands are identified, in order, in `scripts/README.md`.

### 6. Estimate the final v5 regressions

```bash
python scripts/run_v5_past_only_p99_h2.py
python scripts/run_v5_strict_timing_h2.py
python scripts/run_v5_lead_lag.py
python scripts/run_v5_event_time_fe.py
python scripts/run_v5_binary_fe_models.py
```

These scripts produce the main large-versus-ordinary flow contrasts, strict
price-timing checks, lead/lag estimates, shared event-time fixed-effects
robustness, and binary update-incidence models. The frozen outputs are under
`regression_results/v5/`.

Supervisor-requested v6 checks are run after the v5 inputs are frozen:

```bash
python scripts/build_v6_supervisor_diagnostics.py
python scripts/build_v6_threshold_sensitivity.py
python scripts/run_v6_threshold_sensitivity.py
python scripts/build_v6_executed_price_robustness.py
python scripts/run_v6_executed_price_robustness.py
python scripts/run_v6_h3a_conditional_continuation.py
python scripts/run_v6_country_large_ordinary_clustering.py
python scripts/run_v6_wallet_60s_sequence_robustness.py
```

These commands produce timestamp-gap distributions, included/excluded-market
comparisons, logit average marginal effects, P95/P97.5/P99/P99.5 and flow-
transformation sensitivities, executed-price checks, conditional-continuation
estimates, country market-and-time covariance checks, a 60-second execution-
sequence robustness test, and the 35-unit audit disposition table.

### 7. Machine learning and integrity screening

The supervised ML outcome is full thirty-minute price reversal. It is a
prediction task, not a manipulation classifier. The MI3 specification uses
information available through approximately minute 5 to predict reversal from
approximately minute 5 to minute 30. Model thresholds are selected inside the
training folds only.

```bash
python scripts/build_v4_rq3_ml_sample.py
python scripts/run_v4_rq3_ml_baselines.py
python scripts/run_v4_rq3_ml_trees.py
python scripts/run_v4_rq3_rf_ablation.py
python scripts/run_v4_rq3_chronological_validation.py
python scripts/bootstrap_v4_rq3_ml_increment.py
python scripts/audit_v5_ml_leakage.py
```

Unsupervised anomaly screening ranks unusual multi-feature trading events for
contextual audit:

```bash
python scripts/run_v4_integrity_anomaly_screen.py
python scripts/build_v4_integrity_case_audit.py
python scripts/finalize_v4_integrity_case_audit.py
```

An anomaly score is not evidence of misconduct. The researcher audit is a
structured case review and not independent validation; no audited case is
claimed to establish manipulation.

### 8. Compile the dissertation

Upload `overleaf/` to Overleaf and compile `main.tex`, or use a local LaTeX
installation. The submitted PDF should be archived together with the tagged
source snapshot cited in the dissertation.

## Price timing

The strict-0 rule uses the latest observed price strictly before the interval,
signed flow during `[t,t+1)`, and the first observed price at or after each
target horizon. Alignment gaps must be below 60 seconds. Prices are never
interpolated. This prevents contemporaneous price information from leaking into
the baseline.

## Reproducibility and interpretation limits

- API checkpoint and progress files make collection resumable.
- Collection responses are preserved before transformation.
- Raw files are not uploaded because they are very large; their public API
  origins, exact dates, and reconstruction commands are documented here.
- Historical order-book snapshots are outside the design; none of the reported
  models requires them.
- Regressions estimate conditional associations, not structural causal price
  impact.
- Supervised ML predicts reversal; unsupervised ML prioritises unusual cases.
  Neither procedure supplies a verified manipulation label.

## Citation and version matching

The submitted dissertation cites the immutable
[`v1.0.4-submission-final`](https://github.com/andrew1004aa/polymarket-world-cup-2026/tree/v1.0.4-submission-final)
snapshot. This tag and the dissertation source refer to the same frozen
version; the PDF compiled in Overleaf should be checked against that source
before submission.
