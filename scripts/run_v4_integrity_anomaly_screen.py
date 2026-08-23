#!/usr/bin/env python3
"""Rank unusual P99 large-trade episodes without claiming manipulation."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "analysis_ready/v4/behavioral_events/behavioral_p99_events_all.csv"
OUT = ROOT / "integrity_results/v4/anomaly_screen"
SEED = 20260814


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def q_inclusive(s: pd.Series, q: float) -> float:
    return float(s.dropna().quantile(q))


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    eps = 1e-12
    p99 = df["p99_threshold_usdc"].clip(lower=eps)
    feature = pd.DataFrame(index=df.index)
    feature["log_trade_to_p99"] = np.log1p(df["initiating_trade_value_usdc"] / p99)
    feature["minute_wallet_hhi"] = df["initiating_minute_wallet_hhi"]
    feature["minute_abs_flow_imbalance"] = df["initiating_minute_absolute_flow_imbalance"]
    feature["log_minute_value_to_p99"] = np.log1p(df["initiating_minute_gross_value_usdc"] / p99)
    feature["log_minute_wallets"] = np.log1p(df["initiating_minute_distinct_wallets"])
    feature["follower_imbalance_5m"] = df["follower_imbalance_5m"]
    follower_value = df["same_direction_value_5m"] + df["opposite_direction_value_5m"]
    feature["log_follower_value_to_p99_5m"] = np.log1p(follower_value / p99)
    feature["observed_opposite_trade_60m"] = df["initiating_wallet_opposite_trade_within_60m"]
    feature["log_opposite_value_to_p99_60m"] = np.log1p(df["initiating_wallet_opposite_value_60m"] / p99)
    feature["baseline_price_extremeness"] = (df["baseline_yes_price"] - 0.5).abs()
    feature["abs_recent_price_change_15m"] = df["recent_price_change_15m"].abs()
    base_cols = list(feature.columns)
    for c in base_cols:
        if feature[c].isna().any():
            feature[f"{c}__missing"] = feature[c].isna().astype(np.int8)
    return feature, base_cols


def robust_scale_impute(x: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    z = x.copy().astype(float)
    stats: dict[str, dict[str, float]] = {}
    keep: list[str] = []
    for c in z.columns:
        median = float(z[c].median()) if z[c].notna().any() else 0.0
        z[c] = z[c].fillna(median)
        q1, q3 = z[c].quantile([0.25, 0.75]).tolist()
        iqr = float(q3 - q1)
        stats[c] = {"median": median, "q1": float(q1), "q3": float(q3), "iqr": iqr}
        if iqr > 0:
            z[c] = (z[c] - median) / iqr
            keep.append(c)
        elif z[c].nunique(dropna=False) > 1:
            scale = float(z[c].std(ddof=0))
            if scale > 0:
                z[c] = (z[c] - median) / scale
                stats[c]["fallback_std"] = scale
                keep.append(c)
    return z[keep], {"columns_retained": keep, "scaling": stats}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SOURCE, low_memory=False)
    df["market_family"] = np.where(df["event_id"].notna(), "match", "country_outright")
    df["trade_to_p99"] = df["initiating_trade_value_usdc"] / df["p99_threshold_usdc"]
    df["follower_gross_value_5m"] = df["same_direction_value_5m"] + df["opposite_direction_value_5m"]
    df["opposite_to_initiating_value_60m"] = (
        df["initiating_wallet_opposite_value_60m"] / df["initiating_trade_value_usdc"]
    )
    feature, feature_base_cols = build_features(df)

    ranked_parts = []
    configs: dict[str, dict] = {}
    thresholds: list[dict] = []
    for family, idx in df.groupby("market_family", sort=True).groups.items():
        idx = pd.Index(idx)
        sub = df.loc[idx]
        size_cut = q_inclusive(sub["trade_to_p99"], 0.99)
        thresholds.append({"market_family": family, "rule": "extreme_relative_size", "threshold": size_cut})

        rule = pd.DataFrame(index=idx)
        rule["rule_extreme_relative_size"] = sub["trade_to_p99"].ge(size_cut)
        rule["rule_dominant_wallet_minute"] = sub["initiating_minute_wallet_hhi"].ge(0.90)
        rule["rule_unidirectional_minute"] = sub["initiating_minute_absolute_flow_imbalance"].ge(0.90)
        rule["rule_aligned_following_5m"] = (
            sub["follower_imbalance_5m"].ge(0.90)
            & sub["follower_gross_value_5m"].ge(sub["p99_threshold_usdc"])
        )
        rule["rule_observed_opposite_60m"] = (
            sub["initiating_wallet_opposite_trade_within_60m"].eq(1)
            & sub["opposite_to_initiating_value_60m"].ge(0.50)
        )
        rule = rule.astype(np.int8)
        rule["transparent_rule_score"] = rule.sum(axis=1)

        z, preprocessing = robust_scale_impute(feature.loc[idx])
        model = IsolationForest(
            n_estimators=500,
            max_samples=min(4096, len(sub)),
            contamination="auto",
            random_state=SEED,
            n_jobs=1,
        )
        model.fit(z)
        # sklearn score_samples is larger for normal observations; reverse it.
        anomaly = pd.Series(-model.score_samples(z), index=idx, name="iforest_anomaly_score")
        pct_rank = anomaly.rank(method="max", pct=True)
        if_top = pct_rank.ge(0.99).astype(np.int8).rename("iforest_top_1pct")
        family_rank = anomaly.rank(method="min", ascending=False).astype(int).rename("iforest_family_rank")
        ranked_parts.append(pd.concat([rule, anomaly, pct_rank.rename("iforest_percentile_rank"), if_top, family_rank], axis=1))
        configs[family] = {
            "n_rows": int(len(sub)),
            "n_markets": int(sub["market_id"].nunique()),
            "n_events_nonmissing": int(sub["event_id"].nunique()),
            "model": {
                "n_estimators": 500,
                "max_samples": min(4096, len(sub)),
                "contamination": "auto",
                "random_state": SEED,
                "n_jobs": 1,
            },
            "preprocessing": preprocessing,
        }

    scores = pd.concat(ranked_parts).sort_index()
    ranked = pd.concat([df, scores], axis=1)
    rule_cols = [c for c in scores.columns if c.startswith("rule_")]
    ranked["rule_high_priority"] = ranked["transparent_rule_score"].ge(4).astype(np.int8)
    ranked["method_agreement_high_priority"] = (
        ranked["rule_high_priority"].eq(1) & ranked["iforest_top_1pct"].eq(1)
    ).astype(np.int8)
    ranked["initiating_minute_utc"] = pd.to_datetime(
        ranked["initiating_time_utc"], utc=True
    ).dt.floor("min").dt.strftime("%Y-%m-%dT%H:%M:00Z")
    ranked["episode_cluster_id"] = (
        ranked["market_id"].astype(str) + ":" + ranked["initiating_minute_utc"]
    )
    ranked["episode_cluster_large_trade_count"] = ranked.groupby(
        "episode_cluster_id"
    )["initiating_trade_record_id"].transform("size")

    # Outcomes are annotations only and were excluded from model features.
    for h in (15, 30, 60):
        d = ranked[f"directional_price_change_{h}m"]
        init = ranked["directional_price_change_5m"]
        ranked[f"full_reversal_{h}m"] = ((init > 0) & (d < 0)).astype("Int8")
        ranked.loc[init.isna() | d.isna(), f"full_reversal_{h}m"] = pd.NA
        ranked[f"partial_reversal_{h}m"] = ((init > 0) & (d < init)).astype("Int8")
        ranked.loc[init.isna() | d.isna(), f"partial_reversal_{h}m"] = pd.NA

    ranked_path = OUT / "ranked_large_trade_episodes.csv.gz"
    ranked.to_csv(ranked_path, index=False, compression="gzip")
    top = ranked.sort_values(["market_family", "iforest_anomaly_score"], ascending=[True, False])
    top = top.groupby("market_family", sort=False).head(100)
    top.to_csv(OUT / "top_100_cases_per_family.csv", index=False)

    # A burst may contain several P99 trades in one market-minute. Manual case
    # review uses one row per burst to avoid treating those trades as independent
    # cases. The representative row is the largest Isolation Forest score.
    representatives = ranked.loc[
        ranked.groupby("episode_cluster_id")["iforest_anomaly_score"].idxmax()
    ].copy()
    cluster_agg = ranked.groupby("episode_cluster_id", as_index=False).agg(
        cluster_total_large_trade_value_usdc=("initiating_trade_value_usdc", "sum"),
        cluster_max_trade_value_usdc=("initiating_trade_value_usdc", "max"),
        cluster_max_rule_score=("transparent_rule_score", "max"),
        cluster_any_rule_high_priority=("rule_high_priority", "max"),
        cluster_any_iforest_top_1pct=("iforest_top_1pct", "max"),
        cluster_any_method_agreement=("method_agreement_high_priority", "max"),
    )
    clusters = representatives.merge(cluster_agg, on="episode_cluster_id", how="left")
    clusters["iforest_cluster_family_rank"] = clusters.groupby("market_family")[
        "iforest_anomaly_score"
    ].rank(method="min", ascending=False).astype(int)
    clusters.to_csv(OUT / "ranked_market_minute_episode_clusters.csv.gz", index=False, compression="gzip")
    cluster_top = clusters.sort_values(
        ["market_family", "iforest_anomaly_score"], ascending=[True, False]
    ).groupby("market_family", sort=False).head(50)
    cluster_top.to_csv(OUT / "top_50_episode_clusters_per_family.csv", index=False)
    pd.DataFrame(thresholds).to_csv(OUT / "transparent_rule_thresholds.csv", index=False)

    summary_rows = []
    for family, sub in ranked.groupby("market_family", sort=True):
        summary_rows.append({
            "market_family": family,
            "episodes": len(sub),
            "markets": sub["market_id"].nunique(),
            "transparent_score_ge_4": int(sub["rule_high_priority"].sum()),
            "iforest_top_1pct": int(sub["iforest_top_1pct"].sum()),
            "both_methods": int(sub["method_agreement_high_priority"].sum()),
            "rule_iforest_jaccard": float(
                sub["method_agreement_high_priority"].sum()
                / max(1, ((sub["rule_high_priority"] == 1) | (sub["iforest_top_1pct"] == 1)).sum())
            ),
            "observed_opposite_60m": int(sub["initiating_wallet_opposite_trade_within_60m"].eq(1).sum()),
            "valid_directional_price_5m": int(sub["directional_price_change_5m"].notna().sum()),
            "market_minute_episode_clusters": int(sub["episode_cluster_id"].nunique()),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "anomaly_screen_summary.csv", index=False)

    missing = pd.DataFrame({
        "variable": feature_base_cols,
        "missing_count": [int(feature[c].isna().sum()) for c in feature_base_cols],
        "missing_fraction": [float(feature[c].isna().mean()) for c in feature_base_cols],
    })
    missing.to_csv(OUT / "feature_missingness.csv", index=False)

    metadata = {
        "purpose": "Unsupervised ranking for manual review; not manipulation classification",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(SOURCE),
        "rows": int(len(ranked)),
        "markets": int(ranked["market_id"].nunique()),
        "seed": SEED,
        "outcomes_excluded_from_model": [
            c for c in ranked.columns if c.startswith("price_change_")
            or c.startswith("directional_price_change_")
            or "reversal" in c
        ],
        "rule_columns": rule_cols,
        "family_models": configs,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    with (OUT / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    hashes = []
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name != "output_sha256.csv":
            hashes.append({"file": p.name, "sha256": sha256(p), "bytes": p.stat().st_size})
    pd.DataFrame(hashes).to_csv(OUT / "output_sha256.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {ranked_path} ({len(ranked):,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
