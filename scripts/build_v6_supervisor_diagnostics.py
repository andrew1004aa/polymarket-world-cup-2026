#!/usr/bin/env python3
"""Build supervisor-requested non-model diagnostics for dissertation v6."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STRICT = ROOT / "regression_inputs/v5/strict_timing/strict_timing_past_only_p99.csv.gz"
MARKETS = ROOT / "regression_ready/tables/markets.csv"
BINARY_COEF = ROOT / "regression_results/v5/binary_fe/coefficients.csv"
BINARY_CONTRAST = ROOT / "regression_results/v5/binary_fe/contrasts.csv"
AUDIT = ROOT / "integrity_results/v4/case_audit/case_audit_signed.csv"
OUT = ROOT / "regression_results/v6/supervisor_diagnostics"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def quantile_rows(data: pd.DataFrame) -> pd.DataFrame:
    samples = {
        "all_aligned_rows": np.ones(len(data), dtype=bool),
        "expanding_eligible": data.expanding_eligible.eq(1),
        "rolling7_eligible": data.rolling7_eligible.eq(1),
        "expanding_strict0": data.expanding_eligible.eq(1) & data.v5_strict0_5m.eq(1),
        "rolling7_strict0": data.rolling7_eligible.eq(1) & data.v5_strict0_5m.eq(1),
    }
    gap_columns = {
        "baseline_before_t": "v5_baseline_gap_seconds",
        "outcome_after_t_plus_1": "v5_post_1m_gap_seconds",
        "outcome_after_t_plus_5": "v5_post_5m_gap_seconds",
        "outcome_after_t_plus_15": "v5_post_15m_gap_seconds",
    }
    rows = []
    for sample, mask in samples.items():
        part = data.loc[mask]
        for label, column in gap_columns.items():
            values = part[column].dropna().astype(float)
            qs = values.quantile([0, .25, .5, .75, .9, .95, .99, 1])
            rows.append({
                "sample": sample, "gap": label, "rows": len(values),
                "mean_seconds": values.mean(), "p0": qs.loc[0], "p25": qs.loc[.25],
                "p50": qs.loc[.5], "p75": qs.loc[.75], "p90": qs.loc[.9],
                "p95": qs.loc[.95], "p99": qs.loc[.99], "maximum": qs.loc[1],
                "share_below_60_seconds": values.lt(60).mean(),
                "share_at_most_60_seconds": values.le(60).mean(),
            })
    return pd.DataFrame(rows)


def smd(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce").dropna(); b = pd.to_numeric(b, errors="coerce").dropna()
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else np.nan


def market_selection(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    market = pd.read_csv(MARKETS, dtype={"market_id": str})
    market = market[market.market_type.eq("match")].copy()
    fields = ["market_id", "market_subtype", "trade_count", "volume_usdc", "first_trade_time",
              "last_trade_time", "yes_outcome_won"]
    market = market[fields]
    characteristics = data.groupby("market_id", as_index=False).agg(
        event_id=("event_id", "first"), stage=("stage", "first"), market_role=("market_role", "first"),
        median_baseline_price=("v5_baseline_price", "median"),
        mean_baseline_price=("v5_baseline_price", "mean"),
        sample_minutes=("minute_start_timestamp", "size"),
        sample_gross_volume_usdc=("gross_volume_usdc", "sum"),
        median_minutes_to_kickoff=("minutes_to_actual_kickoff", "median"),
        expanding_included=("expanding_eligible", "max"),
        rolling7_included=("rolling7_eligible", "max"),
    )
    characteristics["market_id"] = characteristics.market_id.astype(str)
    frame = market.merge(characteristics, on="market_id", how="left", validate="one_to_one")
    numeric = ["trade_count", "volume_usdc", "median_baseline_price", "mean_baseline_price",
               "sample_minutes", "sample_gross_volume_usdc", "median_minutes_to_kickoff"]
    summaries = []; differences = []
    for spec in ("expanding", "rolling7"):
        flag = f"{spec}_included"
        for group, included in (("included", 1), ("excluded", 0)):
            part = frame[frame[flag].eq(included)]
            for variable in numeric:
                values = pd.to_numeric(part[variable], errors="coerce")
                summaries.append({"specification": spec, "group": group, "variable": variable,
                                  "markets": len(part), "mean": values.mean(), "median": values.median(),
                                  "p25": values.quantile(.25), "p75": values.quantile(.75)})
        inc = frame[frame[flag].eq(1)]; exc = frame[frame[flag].eq(0)]
        for variable in numeric:
            differences.append({"specification": spec, "variable": variable,
                                "included_markets": len(inc), "excluded_markets": len(exc),
                                "standardised_mean_difference": smd(inc[variable], exc[variable])})
    categorical = []
    for spec in ("expanding", "rolling7"):
        flag = f"{spec}_included"
        for variable in ("stage", "market_role", "market_subtype", "yes_outcome_won"):
            counts = frame.groupby([flag, variable], dropna=False).size().rename("markets").reset_index()
            counts["share_within_group"] = counts.groupby(flag).markets.transform(lambda x: x / x.sum())
            counts.insert(0, "specification", spec); counts.insert(1, "variable", variable)
            counts = counts.rename(columns={flag: "included", variable: "category"})
            categorical.append(counts)
    return frame, pd.DataFrame(summaries).merge(pd.DataFrame(differences), on=["specification", "variable"], how="left"), pd.concat(categorical, ignore_index=True)


def average_marginal_effects() -> pd.DataFrame:
    coefficients = pd.read_csv(BINARY_COEF)
    contrasts = pd.read_csv(BINARY_CONTRAST)
    rows = []
    for contrast in contrasts.itertuples():
        derivative = contrast.average_marginal_effect_difference / contrast.estimate_link_scale
        part = coefficients[coefficients.model.eq(contrast.model)]
        focal = part[part.term.str.contains("log_abs_large|log_abs_ordinary", regex=True)]
        for row in focal.itertuples():
            rows.append({"model": row.model, "effect": row.term,
                         "average_marginal_effect": row.estimate * derivative,
                         "approx_clustered_se": row.std_error * derivative,
                         "link_scale_estimate": row.estimate,
                         "average_link_derivative": derivative})
        rows.append({"model": contrast.model, "effect": "large_minus_ordinary",
                     "average_marginal_effect": contrast.average_marginal_effect_difference,
                     "approx_clustered_se": contrast.std_error * derivative,
                     "link_scale_estimate": contrast.estimate_link_scale,
                     "average_link_derivative": derivative})
    return pd.DataFrame(rows)


def audit_units() -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["audit_case_id", "related_case_group", "manual_disposition", "manual_confidence",
            "manual_evidence_note", "market_id", "event_id", "question", "public_source_url"]
    data = pd.read_csv(AUDIT, usecols=cols)
    data["audit_unit_id"] = data.related_case_group.fillna("").astype(str).str.strip()
    data.loc[data.audit_unit_id.eq(""), "audit_unit_id"] = data.loc[data.audit_unit_id.eq(""), "audit_case_id"]
    records = []
    for unit, part in data.groupby("audit_unit_id", sort=True):
        dispositions = sorted(part.manual_disposition.dropna().unique())
        # Conservative unit-level adjudication: any unresolved row keeps the
        # linked unit unresolved; otherwise event-information explanations take
        # precedence over the weaker not-anomalous disposition.
        if "unusual_but_unresolved" in dispositions:
            final = "unusual_but_unresolved"
        elif "event_information" in dispositions:
            final = "event_information"
        elif "not_anomalous_after_context" in dispositions:
            final = "not_anomalous_after_context"
        else:
            final = "insufficient_data"
        records.append({
            "audit_unit_id": unit, "case_rows": len(part),
            "audit_case_ids": ";".join(part.audit_case_id),
            "row_dispositions": ";".join(dispositions), "unit_disposition": final,
            "markets": part.market_id.nunique(), "events": part.event_id.nunique(dropna=True),
            "questions": " | ".join(part.question.dropna().astype(str).unique()),
            "unit_review_rule": "unresolved > event_information > not_anomalous_after_context > insufficient_data",
        })
    units = pd.DataFrame(records)
    summary = units.groupby("unit_disposition").agg(units=("audit_unit_id", "size"),
                                                     case_rows=("case_rows", "sum")).reset_index()
    summary["share_of_35_units"] = summary.units / len(units)
    return units, summary


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["market_id", "event_id", "stage", "market_role", "minute_start_timestamp",
            "minutes_to_actual_kickoff", "gross_volume_usdc", "v5_baseline_price",
            "expanding_eligible", "rolling7_eligible", "v5_strict0_5m",
            "v5_baseline_gap_seconds", "v5_post_1m_gap_seconds", "v5_post_5m_gap_seconds",
            "v5_post_15m_gap_seconds"]
    data = pd.read_csv(STRICT, usecols=cols, dtype={"market_id": str})
    gaps = quantile_rows(data); gaps.to_csv(OUT / "price_timestamp_gap_distributions.csv", index=False)
    market, comparison, categorical = market_selection(data)
    market.to_csv(OUT / "market_inclusion_characteristics.csv", index=False)
    comparison.to_csv(OUT / "market_inclusion_numeric_comparison.csv", index=False)
    categorical.to_csv(OUT / "market_inclusion_categorical_comparison.csv", index=False)
    ame = average_marginal_effects(); ame.to_csv(OUT / "logit_average_marginal_effects.csv", index=False)
    units, unit_summary = audit_units(); units.to_csv(OUT / "audit_35_units.csv", index=False)
    unit_summary.to_csv(OUT / "audit_35_unit_dispositions.csv", index=False)
    summary = {"generated_at": now(), "status": "complete_qc_passed", "strict_rows": len(data),
               "match_markets": int(data.market_id.nunique()),
               "expanding_included_markets": int(market.expanding_included.sum()),
               "rolling7_included_markets": int(market.rolling7_included.sum()),
               "audit_rows": int(units.case_rows.sum()), "audit_units": len(units),
               "unresolved_audit_units": int((units.unit_disposition == "unusual_but_unresolved").sum()),
               "notes": ["No model was estimated by this script.",
                         "AME standard errors use the fitted average link derivative as a delta-method approximation."]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

