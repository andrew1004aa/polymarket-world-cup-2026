# Data Dictionary

## Conventions

- All Unix timestamps are seconds since 1970-01-01 UTC unless stated otherwise.
- ISO timestamps ending in `Z` are UTC.
- Wallet addresses and condition IDs are normalized to lowercase where constructed tables require normalization.
- `market_id` is Polymarket's market identifier.
- `condition_id` is the 32-byte conditional-token condition identifier.
- `token_id` or `asset` is the decimal CLOB outcome-token identifier.
- Prices range from 0 to 1 and are denominated in USDC per share.
- `total_volume_usdc` in constructed wallet tables is `Σ(size × price)`.
- No field in the collected data classifies a whale, institution or wallet owner.

## `intermediate/market_partitions/markets_final.csv`

**Unit:** one market  
**Rows:** 360

| Field | Type | Definition |
|---|---|---|
| market_id | string identifier | Polymarket market ID. |
| condition_id | hex string | Conditional-token condition ID. |
| market_type | categorical | `match` or `outright`. |
| market_subtype | categorical | Match winner, match draw or outright market subtype. |
| country | string, nullable | Country represented by an outright market. |
| question | string | Market question. |
| slug | string | Market URL slug. |
| event_slug | string | Parent-event slug. |
| market_start_time | timestamp | Market metadata start/creation time from the final metadata source. |
| market_end_time | timestamp | Market metadata end time. This is not necessarily resolution time. |
| country_market_versions | integer, nullable | Number of country-market versions represented in the Dune aggregate. |
| resolution_status | categorical | Recorded resolution state. |
| resolution_result | categorical | Recorded yes/no result label. |
| is_resolved | boolean/integer | Resolution indicator. |
| resolved_outcome | categorical | Winning outcome label, normally `Yes` or `No`. |
| country_won_world_cup | boolean/integer, nullable | Indicator for the final outright winner. |
| winning_outcome_index | integer | Winning outcome index. |
| maximum_settlement_value | numeric | Maximum observed settlement value. |
| total_settlement_value | numeric | Total settlement value recorded by the metadata query. |
| resolved_on_timestamp | timestamp | Recorded on-chain resolution timestamp. |
| yes_outcome_won | boolean/integer, nullable | Whether YES won. |
| outcome_index_0_won | boolean/integer, nullable | Whether outcome index 0 won. |
| outcome_token_count | integer | Number of outcome tokens. |
| settled_token_count | integer | Number of settled tokens. |
| trade_count | integer | Auxiliary Dune aggregate count; not the canonical trade sample count. |
| transaction_count | integer | Auxiliary Dune aggregate transaction count. |
| volume_usdc | numeric | Auxiliary aggregate USDC volume. |
| volume_shares | numeric | Auxiliary aggregate share volume. |
| all_match_markets_volume_usdc | numeric, nullable | Aggregate across all included match markets; repeated across match rows. |
| total_traded_usdc | numeric, nullable | Legacy auxiliary aggregate field. Do not treat as market-level volume without checking its source definition. |
| sample_total_volume_usdc | numeric | Auxiliary sample-wide volume field. |
| first_trade_time | timestamp | Auxiliary aggregate first-trade timestamp. |
| last_trade_time | timestamp | Auxiliary aggregate last-trade timestamp. |
| dune_query_id | string identifier | Dune query used for auxiliary market/resolution metadata. |

## `intermediate/market_partitions/csv/{market_id}.csv.gz`

**Unit:** one canonical Data API trade record  
**Files:** 360  
**Rows across files:** 6,725,732

| Field | Type | Definition |
|---|---|---|
| timestamp | integer | Trade timestamp in Unix seconds. |
| transactionHash | hex string | Polygon transaction hash returned by the Data API. |
| proxyWallet | hex string | User proxy-wallet address returned by the Data API. It is not an inferred owner identity. |
| side | categorical | API trade side, normally `BUY` or `SELL`. |
| outcome | categorical | Outcome token label, normally `Yes` or `No`. |
| asset | decimal string | CLOB outcome-token ID. Must be treated as text because it exceeds ordinary integer precision. |
| conditionId | hex string | Conditional-token condition ID as returned by the API. |
| market_id | string identifier | Polymarket market ID added during partition construction. |
| market_type | categorical | `match` or `outright`. |
| title | string | Market title/question returned with the trade. |
| price | decimal | Transaction price in USDC per share. |
| size | decimal | Number of shares in the trade record. |

The canonical trade value is not stored as a separate column in the partition file. When required later it is defined as `size × price`.

## `trades.csv`

**Unit:** one collected Data API record before compact market partitioning  
**Rows:** 6,725,732 retained unique rows

| Field | Type | Definition |
|---|---|---|
| proxyWallet | hex string | Proxy-wallet address. |
| side | categorical | API side. |
| asset | decimal string | Outcome-token ID. |
| conditionId | hex string | Condition ID. |
| size | decimal | Shares. |
| price | decimal | Transaction price. |
| timestamp | integer | Unix timestamp. |
| title | string | Market question/title. |
| slug | string | Market slug returned on the trade. |
| icon | string/URL | Market icon field returned by the API. |
| eventSlug | string | Parent-event slug returned by the API. |
| outcome | categorical | Outcome label. |
| outcomeIndex | integer | Outcome index. |
| name | string, nullable | Public profile name returned by the API. |
| pseudonym | string, nullable | Public profile pseudonym. |
| bio | string, nullable | Public profile biography. |
| profileImage | string/URL, nullable | Profile image field. |
| profileImageOptimized | string/URL, nullable | Optimized profile image field. |
| transactionHash | hex string | Transaction hash. |
| market_id | string identifier | Joined Polymarket market ID. |
| market_type | categorical | Joined market type. |
| window_start | integer | Start of the collection request window. |
| window_end_exclusive | integer | Exclusive end of the collection request window. |
| raw_json | JSON string | Original API row serialized for preservation. |

## `intermediate/wallets/wallets.csv`

**Unit:** one normalized wallet address  
**Rows:** 335,414

| Field | Type | Definition |
|---|---|---|
| wallet_address | hex string | Lowercase proxy-wallet address. |
| number_of_trades | integer | Number of canonical trade records associated with the wallet. |
| markets_traded | integer | Number of distinct markets traded. |
| total_shares | decimal | Sum of trade `size`. |
| total_volume_usdc | decimal | Sum of `size × price`. |
| first_trade_timestamp | integer | Earliest canonical trade timestamp. |
| first_trade_utc | timestamp | Earliest trade in ISO UTC. |
| last_trade_timestamp | integer | Latest canonical trade timestamp. |
| last_trade_utc | timestamp | Latest trade in ISO UTC. |

## `intermediate/wallets/wallet_market.csv`

**Unit:** one wallet–market pair  
**Rows:** 2,497,766

| Field | Type | Definition |
|---|---|---|
| wallet_address | hex string | Lowercase proxy-wallet address. |
| market_id | string identifier | Polymarket market ID. |
| condition_id | hex string | Condition ID. |
| market_type | categorical | `match` or `outright`. |
| trade_count | integer | Canonical trade records for this wallet–market pair. |
| buy_count | integer | Records whose Data API side is BUY. |
| sell_count | integer | Records whose Data API side is SELL. |
| total_shares | decimal | Sum of shares for the pair. |
| total_volume_usdc | decimal | Sum of `size × price` for the pair. |
| first_trade_timestamp | integer | Earliest pair-level trade timestamp. |
| first_trade_utc | timestamp | Earliest pair-level trade in ISO UTC. |
| last_trade_timestamp | integer | Latest pair-level trade timestamp. |
| last_trade_utc | timestamp | Latest pair-level trade in ISO UTC. |

## `raw/prices/prices.csv`

**Unit:** one API price-history point for one outcome token  
**Rows:** 17,980,605

| Field | Type | Definition |
|---|---|---|
| market_id | string identifier | Polymarket market ID. |
| condition_id | hex string | Condition ID. |
| market_type | categorical | `match` or `outright`. |
| question | string | Market question. |
| outcome | categorical | `YES` or `NO`; identifies the token history requested. |
| token_id | decimal string | Outcome-token ID. Treat as text. |
| timestamp | integer | Price-point Unix timestamp. |
| timestamp_utc | timestamp | Price-point ISO UTC timestamp. |
| price | decimal | Price returned by `/prices-history`. |
| requested_start_timestamp | integer | Start of the overall requested interval. |
| requested_end_timestamp_exclusive | integer | Exclusive end of the overall interval. |
| fidelity_minutes | integer | Requested API fidelity in minutes; value is 1. |
| request_window_start | integer | Start of the 14-day-or-shorter API request checkpoint. |
| request_window_end | integer | End of the API request checkpoint. |
| raw_point_json | JSON string | Original `{t,p}` price point serialized for preservation. |

Price-history points are not historical bid–ask quotes or historical order-book snapshots. No missing minute is filled.

## `raw/events/events.csv`

**Unit:** one FIFA match event  
**Rows:** 104

| Field | Type | Definition |
|---|---|---|
| event_id | integer identifier | Project event ID, equal to FIFA match sequence. |
| fifa_match_id | integer identifier | Match sequence in the FIFA source file. |
| polymarket_event_id | string identifier | Polymarket parent-event ID. |
| polymarket_game_id | string identifier | Polymarket game ID. |
| event_slug | string | Polymarket event slug. |
| stage | categorical | Group stage, Round of 32, Round of 16, quarter-final, semi-final, third-place match or final. |
| match_date_bst | date string | Actual match date in Europe/London time. |
| match_time_bst | time string | Actual match time in Europe/London time. |
| actual_kickoff_bst | timestamp | Actual FIFA kickoff with `+01:00` BST offset. |
| actual_kickoff_utc | timestamp | Actual FIFA kickoff converted to UTC. |
| scheduled_kickoff_utc | timestamp | Scheduled kickoff, represented by Polymarket market-end metadata. |
| polymarket_market_end_utc | timestamp | Original Polymarket market-end metadata, retained explicitly. |
| kickoff_metadata_difference_seconds | integer | `polymarket_market_end_utc − actual_kickoff_utc`, in seconds. |
| kickoff_delay_minutes | integer | Non-negative documented delay in minutes. Match 92 equals 60; other rows equal 0. |
| kickoff_delay_reason | string, nullable | Documented delay reason. Match 92 is `weather`. |
| timezone | string | IANA timezone `Europe/London`. |
| home_team | string | FIFA home-team label. |
| away_team | string | FIFA away-team label. |
| match_name | string | `home_team vs. away_team`. |
| result | string | FIFA result exactly as supplied, including penalty notation where present. |
| market_count | integer | Number of linked markets; always 3. |
| home_win_market_id | string identifier | Home-win market ID. |
| home_win_condition_id | hex string | Home-win condition ID. |
| home_win_question | string | Home-win question. |
| away_win_market_id | string identifier | Away-win market ID. |
| away_win_condition_id | hex string | Away-win condition ID. |
| away_win_question | string | Away-win question. |
| draw_market_id | string identifier | Draw market ID. |
| draw_condition_id | hex string | Draw condition ID. |
| draw_question | string | Draw question. |
| fixture_source | string | Fixture-source description. |
| market_source | string | Market-mapping source description. |

## `raw/events/event_market_mapping.csv`

**Unit:** one event–market role  
**Rows:** 312

| Field | Type | Definition |
|---|---|---|
| event_id | integer identifier | Project/FIFA event ID. |
| fifa_match_id | integer identifier | FIFA match sequence. |
| polymarket_event_id | string identifier | Polymarket parent-event ID. |
| polymarket_game_id | string identifier | Polymarket game ID. |
| event_slug | string | Polymarket event slug. |
| actual_kickoff_utc | timestamp | Actual FIFA kickoff in UTC. |
| home_team | string | FIFA home-team label. |
| away_team | string | FIFA away-team label. |
| market_role | categorical | `home_win`, `away_win` or `draw`. |
| market_id | string identifier | Linked market ID. |
| condition_id | hex string | Linked condition ID. |
| market_subtype | categorical | Match winner or match draw. |
| question | string | Market question. |
| resolution_status | categorical | Recorded resolution status. |
| resolved_outcome | categorical | Recorded winning outcome. |
| yes_outcome_won | boolean/integer | Whether YES won. |
| resolved_on_timestamp | timestamp | Recorded resolution timestamp. |

## Manifests, progress and logs

### `intermediate/market_partitions/market_manifest.csv`

One row per market partition. Contains IDs, title, row count, first/last timestamp, missing-field counts, duplicate count, source checkpoint count, output path and SHA-256 checksum.

### Progress files

- `data_api_progress.json`: completed Data API market/time-window downloads, failures and retries.
- `raw/prices/progress.json`: completed token/window price downloads, failures and retry counts.
- `intermediate/market_partitions/progress.json`: completed partition outputs and failures.

### Request logs

- `logs/data_api_log.jsonl`: one log row per Data API request attempt.
- `logs/price_history_api_log.jsonl`: one log row per CLOB price-history request attempt.

These logs and progress files are collection metadata, not regression observations.

## Explicitly unavailable fields

The final collection does not contain historical:

- best bid
- best ask
- bid–ask midpoint
- quoted spread
- order-book depth
- depth imbalance
- full order-book snapshots

Current-order-book endpoints and real-time WebSockets cannot retrospectively reconstruct these fields for the completed tournament.
