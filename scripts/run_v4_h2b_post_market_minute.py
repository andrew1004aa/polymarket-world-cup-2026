#!/usr/bin/env python3
"""Aggregate H2b post-kickoff trades to equal-weight market-minute cells and estimate LPMs."""

from __future__ import annotations

import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "model_samples/v4/h2b/h2b_5m_binary.csv.gz"
SAMPLE = ROOT / "model_samples/v4/h2b/h2b_match_post_5m_market_minute.csv.gz"
OUT = ROOT / "regression_results/v4/h2b"
USECOLS = [
    "market_id", "event_id", "market_type", "timestamp", "phase",
    "trade_value_usdc", "is_p99_large", "yes_price_t",
    "recent_price_change", "momentum",
]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_sample():
    SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    partial = []
    source_rows = post_rows = 0
    for chunk in pd.read_csv(SOURCE, usecols=USECOLS, chunksize=250_000):
        source_rows += len(chunk)
        chunk = chunk[(chunk.market_type == "match") & (chunk.phase == "post")].copy()
        if chunk.empty:
            continue
        post_rows += len(chunk)
        chunk["minute_start_timestamp"] = (chunk.timestamp.astype("int64") // 60) * 60
        chunk["log_trade_value"] = np.log1p(chunk.trade_value_usdc)
        chunk["abs_recent_price_change"] = chunk.recent_price_change.abs()
        grouped = chunk.groupby(
            ["market_id", "event_id", "minute_start_timestamp", "is_p99_large"],
            sort=False, observed=True,
        ).agg(
            momentum_sum=("momentum", "sum"),
            trade_count=("momentum", "size"),
            log_trade_value_sum=("log_trade_value", "sum"),
            abs_recent_price_change_sum=("abs_recent_price_change", "sum"),
            yes_price_sum=("yes_price_t", "sum"),
        ).reset_index()
        partial.append(grouped)
        print(f"Read {source_rows:,} rows; retained {post_rows:,} post-kickoff rows", flush=True)

    cells = pd.concat(partial, ignore_index=True)
    keys = ["market_id", "event_id", "minute_start_timestamp", "is_p99_large"]
    cells = cells.groupby(keys, sort=True, observed=True).agg(
        momentum_sum=("momentum_sum", "sum"),
        trade_count=("trade_count", "sum"),
        log_trade_value_sum=("log_trade_value_sum", "sum"),
        abs_recent_price_change_sum=("abs_recent_price_change_sum", "sum"),
        yes_price_sum=("yes_price_sum", "sum"),
    ).reset_index()
    cells["momentum_share"] = cells.momentum_sum / cells.trade_count
    cells["mean_log_trade_value"] = cells.log_trade_value_sum / cells.trade_count
    cells["mean_abs_recent_price_change"] = cells.abs_recent_price_change_sum / cells.trade_count
    cells["mean_yes_price_t"] = cells.yes_price_sum / cells.trade_count
    cells["log_trade_count"] = np.log1p(cells.trade_count)
    cells["calendar_hour_utc"] = pd.to_datetime(
        cells.minute_start_timestamp, unit="s", utc=True
    ).dt.floor("h").astype(str)
    pair_count = cells.groupby(["market_id", "minute_start_timestamp"], observed=True)[
        "is_p99_large"
    ].transform("nunique")
    cells["paired_minute"] = (pair_count == 2).astype("int8")
    if int(cells.trade_count.sum()) != post_rows:
        raise RuntimeError("Aggregation reconciliation failed: cell trade counts do not match post-kickoff rows")
    if cells.duplicated(["market_id", "minute_start_timestamp", "is_p99_large"]).any():
        raise RuntimeError("Duplicate market-minute-by-trade-class cells detected")
    paired_check = cells.loc[cells.paired_minute == 1].groupby(
        ["market_id", "minute_start_timestamp"], observed=True
    ).size()
    if not paired_check.eq(2).all():
        raise RuntimeError("Paired-minute QC failed")
    keep = [
        "market_id", "event_id", "minute_start_timestamp", "calendar_hour_utc",
        "is_p99_large", "momentum_share", "trade_count", "log_trade_count",
        "mean_log_trade_value", "mean_abs_recent_price_change", "mean_yes_price_t",
        "paired_minute",
    ]
    temporary = SAMPLE.with_suffix(SAMPLE.suffix + ".tmp")
    cells[keep].to_csv(temporary, index=False, compression="gzip")
    temporary.replace(SAMPLE)
    summary = {
        "generated_at": now(),
        "status": "built_qc_passed",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(SOURCE),
        "source_binary_rows": source_rows,
        "post_kickoff_trade_rows": post_rows,
        "aggregated_trade_count": int(cells.trade_count.sum()),
        "market_minute_class_cells": len(cells),
        "unique_market_minutes": int(cells[["market_id", "minute_start_timestamp"]].drop_duplicates().shape[0]),
        "paired_cells": int(cells.paired_minute.sum()),
        "paired_market_minutes": int(cells.loc[cells.paired_minute == 1, ["market_id", "minute_start_timestamp"]].drop_duplicates().shape[0]),
        "markets": int(cells.market_id.nunique()),
        "events": int(cells.event_id.nunique()),
        "duplicate_cells": int(cells.duplicated(["market_id", "minute_start_timestamp", "is_p99_large"]).sum()),
        "output": str(SAMPLE.relative_to(ROOT)),
        "output_sha256": sha256(SAMPLE),
        "weighting": "Each market-minute-by-trade-class cell receives equal regression weight.",
    }
    (SAMPLE.parent / "h2b_match_post_5m_market_minute_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return cells[keep], summary


def absorb_two_way(values, group_a, group_b, tolerance=1e-10, max_iterations=10_000):
    """Alternating projections for two additive fixed effects."""
    residual = np.asarray(values, dtype=float).copy()
    codes = [pd.factorize(group_a, sort=False)[0], pd.factorize(group_b, sort=False)[0]]
    for iteration in range(1, max_iterations + 1):
        previous = residual.copy()
        for code in codes:
            counts = np.bincount(code)
            for column in range(residual.shape[1]):
                means = np.bincount(code, weights=residual[:, column]) / counts
                residual[:, column] -= means[code]
        if np.max(np.abs(residual - previous)) < tolerance:
            return residual, iteration
    raise RuntimeError("Two-way fixed-effect absorption did not converge")


def fwl_cluster_lpm(data, terms):
    matrix = data[["momentum_share", *terms]].to_numpy(dtype=float)
    residualized, iterations = absorb_two_way(
        matrix, data.market_id.to_numpy(), data.calendar_hour_utc.to_numpy()
    )
    y = residualized[:, 0]
    x = residualized[:, 1:]
    beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if rank != x.shape[1]:
        raise RuntimeError(f"Rank-deficient residualized design: rank={rank}, k={x.shape[1]}")
    error = y - x @ beta
    bread = np.linalg.inv(x.T @ x)
    cluster_codes, cluster_labels = pd.factorize(data.event_id, sort=False)
    scores = np.zeros((len(cluster_labels), x.shape[1]))
    np.add.at(scores, cluster_codes, x * error[:, None])
    meat = scores.T @ scores
    n, k, groups = len(data), x.shape[1], len(cluster_labels)
    correction = (groups / (groups - 1)) * ((n - 1) / (n - k))
    covariance = correction * bread @ meat @ bread
    standard_error = np.sqrt(np.diag(covariance))
    t_statistic = beta / standard_error
    p_value = 2 * stats.t.sf(np.abs(t_statistic), df=groups - 1)
    critical = stats.t.ppf(0.975, df=groups - 1)
    coefficients = pd.DataFrame({
        "term": terms, "estimate": beta, "std_error": standard_error,
        "t_statistic": t_statistic, "p_value": p_value,
        "conf_low": beta - critical * standard_error,
        "conf_high": beta + critical * standard_error,
    })
    sst = float(y @ y)
    within_r2 = float(1 - (error @ error) / sst) if sst else math.nan
    return coefficients, within_r2, iterations


def estimate(data, name, sample_note):
    target = OUT / name
    target.mkdir(parents=True, exist_ok=True)
    data = data.copy()
    terms = ["is_p99_large", "mean_log_trade_value",
             "mean_abs_recent_price_change", "mean_yes_price_t", "log_trade_count"]
    formula = (
        "momentum_share ~ is_p99_large + mean_log_trade_value + "
        "mean_abs_recent_price_change + mean_yes_price_t + log_trade_count | "
        "market_id + calendar_hour_utc"
    )
    config = {
        "version": "v4", "hypothesis": "H2b", "sample": name, "horizon": "5m",
        "unit": "market-minute-by-trade-class cell", "status": "running",
        "source": str(SAMPLE.relative_to(ROOT)), "source_sha256": sha256(SAMPLE),
        "rows": len(data), "formula": formula, "cluster": "event_id",
        "regression_weighting": "equal cell weights", "sample_note": sample_note,
        "interpretation": "Conditional association with within-cell momentum share; not a causal or bias-identification model.",
    }
    (target / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Estimating {name} on {len(data):,} equal-weight cells", flush=True)
    coefficients, within_r2, absorption_iterations = fwl_cluster_lpm(data, terms)
    coefficients.to_csv(target / "coefficients.csv", index=False)
    config.update({
        "status": "complete_qc_passed", "completed_at": now(),
        "diagnostics": {
            "observations": int(len(data)), "markets": int(data.market_id.nunique()),
            "clusters": int(data.event_id.nunique()),
            "calendar_hours": int(data.calendar_hour_utc.nunique()),
            "unweighted_mean_cell_momentum_share": float(data.momentum_share.mean()),
            "large_cell_share": float(data.is_p99_large.mean()),
            "within_r_squared": within_r2,
            "absorption_iterations": absorption_iterations,
            "software": {"python": platform.python_version(), "pandas": pd.__version__,
                         "numpy": np.__version__, "scipy": scipy.__version__},
        },
    })
    (target / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return config, coefficients


def main():
    data, build_summary = build_sample()
    all_config, all_coef = estimate(
        data, "match_post_5m_market_minute",
        "All observed post-kickoff market-minute-by-class cells.",
    )
    paired = data[data.paired_minute == 1].copy()
    paired_config, paired_coef = estimate(
        paired, "match_post_5m_paired_market_minute",
        "Only market-minutes containing both P99-large and ordinary trades.",
    )
    result = {
        "generated_at": now(), "status": "complete_qc_passed", "hypothesis": "H2b",
        "replacement_for": "deferred 1,978,216-row trade-level post-kickoff model",
        "build": build_summary,
        "models": [],
    }
    for label, config, coefficients in [
        ("all_cells", all_config, all_coef), ("paired_minutes", paired_config, paired_coef)
    ]:
        row = coefficients.loc[coefficients.term == "is_p99_large"].iloc[0]
        result["models"].append({
            "model": label, "observations": config["diagnostics"]["observations"],
            "clusters": config["diagnostics"]["clusters"],
            "estimate": float(row.estimate), "std_error": float(row.std_error),
            "p_value": float(row.p_value), "conf_low": float(row.conf_low),
            "conf_high": float(row.conf_high),
        })
    (OUT / "h2b_post_market_minute_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
