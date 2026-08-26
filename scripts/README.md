# Script map

This directory contains the code used across the dissertation. Version numbers
refer to development freezes, not different datasets: later specifications can
consume samples produced by earlier construction stages.

## Core collection pipeline

| Order | Script | Function |
|---:|---|---|
| 1 | `discover_world_cup_markets.py` | Discover and whitelist the World Cup markets; preserve Gamma metadata |
| 2 | `fetch_world_cup_data_api.py` | Resumable time-window download of the canonical trade observations |
| 3 | `partition_trades_by_market.py` | Write one trade file per market and a partition manifest |
| 4 | `build_wallet_tables.py` | Create wallet and wallet-market aggregates |
| 5 | `fetch_price_history.py` | Resumable YES/NO token price-history collection at one-minute fidelity |
| 6 | `build_events_table.mjs` | Link 312 match contracts to the 104 official match events |

The on-chain RPC prototype is not part of the canonical collection pipeline.
Trade-level analysis uses the Polymarket Data API; Dune market counts were used
only for an independent coverage cross-check.

## Analysis-data construction

The principal dependency chain is:

1. `build_regression_ready.py`
2. `build_market_minute_panel_v2.py`
3. `build_model_samples.py`
4. `build_within_market_large_trade_thresholds.py`
5. `build_large_trade_market_minute_v3.py`
6. `build_v4_behavioral_event_features.py`
7. `build_v4_h2b_trade_classification.py`
8. `build_v4_h2c_sample.py`
9. `build_v4_rq3_ml_sample.py`
10. `build_v5_past_only_p99_sample.py`
11. `build_v5_strict_timing_panel.py`
12. `build_v5_lead_lag_panel.py`

Supporting builders create country-market, wallet-sequence, dynamic-wallet,
phase, horizon, and anomaly-audit samples. Each script declares its source and
output paths near the top of the file.

## Main v5 estimation

| Script | Output |
|---|---|
| `run_v5_past_only_p99_h2.py` | Past-only P99 large-versus-ordinary H2 estimates |
| `run_v5_strict_timing_h2.py` | Strict price-time alignment estimates |
| `run_v5_lead_lag.py` | Pre-trend and post-trade horizon family with Holm adjustment |
| `run_v5_event_time_fe.py` | Event-by-minute and event-by-five-minute fixed effects |
| `run_v5_binary_fe_models.py` | Update-incidence LPM and fixed-effects binary models |
| `plot_v5_lead_lag.py` | Dissertation lead/lag coefficient figure |

## Behavioural and wallet analyses

The `build_v4_past_only_*`, `run_v4_past_only_*`,
`run_v4_dynamic_wallet_*`, and
`run_v4_continuous_cumulative_wallet_volume.py` families implement the
past-only wallet classifications, 2x2 large-trade/wallet comparisons,
continuation, favourable-opposite action, reversal, and continuous cumulative
wallet-volume robustness checks.

## Machine learning

| Script | Function |
|---|---|
| `build_v4_rq3_ml_sample.py` | Construct the no-look-ahead reversal prediction sample |
| `run_v4_rq3_ml_baselines.py` | Logistic and regularised linear baselines |
| `run_v4_rq3_ml_trees.py` | Random-forest and boosted-tree models |
| `run_v4_rq3_rf_ablation.py` | Feature-block ablation |
| `run_v4_rq3_chronological_validation.py` | Chronological out-of-sample validation |
| `bootstrap_v4_rq3_ml_increment.py` | Bootstrap uncertainty for incremental predictive performance |
| `audit_v5_ml_leakage.py` | Verify feature availability and training-only threshold selection |

## Integrity screening and audit

`run_v4_integrity_anomaly_screen.py` performs unsupervised anomaly screening.
`build_v4_integrity_case_audit.py` and
`finalize_v4_integrity_case_audit.py` construct and freeze the contextual case
audit. These scripts flag observations for review; they do not produce a
ground-truth manipulation label.

## Robustness and reporting

Final supervisor-requested v6 checks are:

| Script | Function |
|---|---|
| `build_v6_supervisor_diagnostics.py` | Timestamp gaps, H2 market inclusion, logit AMEs, and 35-unit audit dispositions |
| `build_v6_threshold_sensitivity.py` | Past-only P95/P97.5/P99/P99.5 flow panels |
| `run_v6_threshold_sensitivity.py` | Signed-log, `asinh`, raw-flow, standardised, and common-value comparisons |
| `build_v6_executed_price_robustness.py` | Executed-trade-price outcome construction |
| `run_v6_executed_price_robustness.py` | 60/300/900-second executed-price models |
| `run_v6_h3a_conditional_continuation.py` | Descriptive H3a check conditional on an observed initial update |
| `run_v6_country_large_ordinary_clustering.py` | Market, date, and two-way country covariance estimates |
| `run_v6_wallet_60s_sequence_robustness.py` | Collapse adjacent actions into 60-second execution sequences |

- `run_v3_streaming_wild_cluster_bootstrap.py`: memory-safe wild-cluster
  bootstrap implementation.
- `run_phase_interaction_v3.py`: phase interaction / difference-in-differences
  style comparisons.
- `run_horizon_robustness.py`, `run_country_robustness.py`, and
  `run_inference_robustness.py`: horizon, market-family, and inference checks.
- `build_overleaf_v3_tables.py`, `build_overleaf_regression_tables.py`, and
  `build_v3_coefficient_plots.py`: publication artifacts.

Scripts labelled `prototype`, workbook-inspection utilities, and internal draft
audits are development aids and are not required to reproduce the submitted
estimates.
