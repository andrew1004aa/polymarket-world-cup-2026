# Data Collection Report

## Project

**Title:** Large Trades and Short-Horizon Price Adjustment on Polymarket: Evidence from the 2026 FIFA World Cup
**Phase:** Data collection only
**Study interval:** `2026-06-01T00:00:00Z <= timestamp < 2026-08-01T00:00:00Z`
**Canonical trade source:** Polymarket Data API collected sample

This report documents collection, preservation, organization and structural quality control. It contains no whale classification, price-impact estimation, return construction, regression, hypothesis test, visualization, interpretation or conclusion.

## Final dataset coverage

| Component | Final coverage |
|---|---:|
| Markets | 360 |
| Individual match markets | 312 |
| Outright winner markets | 48 |
| FIFA match events | 104 |
| Event–market mappings | 312 |
| Data API trade rows | 6,725,732 |
| Market-partitioned trade files | 360 |
| Unique wallets | 335,414 |
| Wallet–market rows | 2,497,766 |
| YES/NO price-history tokens | 720 |
| Price-history request windows | 3,600 |
| Price-history points | 17,980,605 |

## Market universe

Market discovery used public Polymarket Gamma API responses and preserved raw discovery pages. Initial discovery produced 362 candidates: 312 match markets and 50 outright markets. The final confirmed research universe contains 360 markets: 312 match markets and 48 outright markets.

Italy and Peru were excluded during final universe reconciliation because they were not part of the confirmed 48-country outright sample for the fixed study interval. Player props, goals, corners, cards, Asian handicap, over/under, qualification, group betting and unrelated sports were excluded by design.

The final whitelist is stored in `intermediate/market_partitions/market_whitelist.csv`. Final resolution metadata is stored in `intermediate/market_partitions/markets_final.csv`.

### Resolution metadata QC

- Final market rows: 360
- Unique market IDs: 360
- Unique condition IDs: 360
- Match markets: 312
- Outright markets: 48
- Resolved outright winner: Spain
- QC errors: 0
- Markets whose recorded last trade is later than `resolved_on_timestamp`: 138

The last item is retained as a timing-definition warning. No rows were removed on that basis.

## Trade collection

Trade records were collected from the public Polymarket Data API and checkpointed as 22,575 gzip-compressed raw responses under `raw/trades/data_api/`. Requests used resumable time-window subdivision so an interrupted collection could continue without restarting completed work.

The original combined file is `trades.csv`. A compact research copy was partitioned into 360 gzip CSV files under `intermediate/market_partitions/csv/`, one file per market.

### Canonical-source decision

Trade-level observations were collected directly from the Polymarket Data API
and constitute the sole canonical trade-level dataset. After collection,
independently obtained Dune Analytics market-level trade counts were used solely
to assess the completeness of the API extraction. No Dune observations were
incorporated into or used to modify the canonical dataset. The canonical sample
contains 6,725,732 trade observations across 360 markets.

### Trade QC

- Included Data API rows: 6,725,732
- Missing timestamps: 0
- Missing condition IDs: 0
- Missing token IDs: 0
- Missing wallet addresses: 0
- Failed markets: 0
- Saturated minimum request windows: 0
- Rows outside the exact final study interval: 0
- First retained trade: `2026-06-01T00:00:01Z`
- Last retained trade: `2026-07-20T01:21:19Z`

Although the collection window extends to 1 August, no retained sample trade occurred after 20 July because the included markets had closed or resolved.

## Wallet tables

Wallet tables were constructed exclusively from the canonical Data API trades. Addresses were normalized to lowercase. No wallet ownership, institution status or whale classification was inferred.

### Wallet QC

- Wallet rows: 335,414
- Wallet–market rows: 2,497,766
- Sum of wallet trade counts: 6,725,732
- Sum of wallet–market trade counts: 6,725,732
- BUY rows: 5,223,471
- SELL rows: 1,502,261
- Total shares: 5,654,390,220.077361
- Total trade value, defined as `size × price`: 1,533,127,776.53532 USDC
- Invalid wallet addresses in independent CSV scan: 0
- Wallet–market buy/sell reconciliation errors: 0
- QC errors: 0

Aggregated share and USDC fields use SQLite `REAL` values. This storage choice is recorded because it can create immaterial floating-point differences in final decimal places.

## Price history

Historical prices were downloaded from the public Polymarket CLOB `/prices-history` endpoint for both YES and NO tokens. Each of the 360 binary markets contributed two token IDs, for 720 token histories.

The API rejected a single 61-day request at one-minute fidelity, so each token was divided into five request windows of no more than 14 days. Every successful response was saved unchanged inside a gzip checkpoint. The merged long-format CSV contains no resampling, interpolation or gap filling.

### Price-history QC

- Tokens expected: 720
- Request windows expected: 3,600
- Missing checkpoints: 0
- Price rows: 17,980,605
- Malformed points excluded: 0
- Points outside the exact study interval excluded: 0
- Empty request windows: 2,232
- Fidelity: 1 minute
- QC errors: 0

An empty request window means the API returned no price points for that token and period. It is not treated as a failed request and is not filled artificially.

## FIFA event table

The preserved FIFA source file at `raw/events/source/fifa_world_cup_2026.xlsx` is the fixture and result source. Its 104 rows were matched to the existing 312 Polymarket match markets using native Polymarket `event_id`, `game_id`, `event_slug`, team identities and kickoff metadata.

Each event maps to exactly three markets:

- home-team win
- away-team win
- draw

### Event-table QC

- FIFA fixtures: 104
- Polymarket event groups: 104
- Event rows: 104
- Event–market rows: 312
- Unique market IDs: 312
- Unique condition IDs: 312
- Home-win rows: 104
- Away-win rows: 104
- Draw rows: 104
- Unmatched fixtures: 0
- Unused Polymarket event groups: 0
- Role errors: 0
- QC errors: 0

FIFA times are recorded in `Europe/London`. During the sample this is British Summer Time, UTC+1. FIFA match 92, Mexico versus England, began 60 minutes late because of weather. The event table therefore preserves:

- scheduled kickoff: `2026-07-06T00:00:00Z`
- actual kickoff: `2026-07-06T01:00:00Z`
- delay: 60 minutes
- reason: weather

This is a documented event, not a data-quality error.

## Historical order book availability

The documented public CLOB API provides current order books and real-time WebSocket updates, but no documented endpoint was identified that provides retrospective historical order-book snapshots for the completed study period. The project did not run an order-book recorder during the tournament. Consequently, historical best bid, best ask, midpoint, spread, depth and order-book imbalance cannot be reconstructed from the collected public API files. The assessment and official documentation URLs are recorded in `docs/orderbook_data_availability.md`.

Available transaction prices and one-minute CLOB price history are retained, but they must not be described as historical bid–ask quotes or pre-trade midpoints. This limitation affects quote-based measures such as quoted spread, effective spread, realized spread and historical depth. It does not alter the completeness of the canonical trade sample.

## Failed requests, retries and rate limiting

- Gamma discovery unresolved failures: 0
- Data API unresolved failures: 0
- Data API saturated minimum windows: 0
- Price-history missing checkpoints: 0
- Price-history unresolved failures: 0
- Final market partition failures: 0

The Data API log contains 28 transient failures (12 connection errors and 16 read timeouts), all recovered. The price-history log contains 10 transient or development-request failures, including the rejected initial 61-day request; all 3,600 final request windows completed successfully. Neither log contains an HTTP 429 response. A compact summary is stored in `api_log.json`, while the complete request-level records remain in the JSONL logs.

Transient request attempts and development tests are retained in API logs. No unresolved rate-limit failure remains in the final collection. Every completed download stage has a progress or checkpoint file.

## Reproducibility and checkpoints

Key reproducibility artifacts include:

- `raw/markets/gamma_discovery/`: raw Gamma discovery pages
- `raw/trades/data_api/`: raw Data API trade checkpoints
- `raw/prices/checkpoints/`: raw CLOB price-history checkpoints
- `data_api_progress.json`: Data API collection progress
- `raw/prices/progress.json`: price-history progress
- `intermediate/market_partitions/progress.json`: market partition progress
- `intermediate/market_partitions/market_manifest.csv`: per-market checksums and row counts
- `logs/data_api_log.jsonl`: Data API request log
- `logs/price_history_api_log.jsonl`: price-history request log
- `api_log.json`: compact API-log summary and pointers to the complete JSONL logs
- `download_log.md`: human-readable collection and retry record
- `raw/events/source/fifa_world_cup_2026.xlsx`: preserved FIFA source file

## Final collection status

The following collection components are complete:

- market discovery and final universe
- canonical trade collection
- market-level trade partitions
- resolution metadata
- wallet and wallet–market tables
- YES and NO price history
- FIFA event and event–market mapping
- auxiliary Dune/Data API comparison, closed without altering the canonical sample
- historical order-book availability assessment

The researcher completed and signed the documented manual review on 2026-08-02 with decision `PASS` and approval `YES`. A deterministic regression-ready data layer was subsequently created under `regression_ready/`; its construction did not include regression estimation, statistical tests, whale classification or interpretation.
