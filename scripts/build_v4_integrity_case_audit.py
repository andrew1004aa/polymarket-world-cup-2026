#!/usr/bin/env python3
"""Build the frozen V4 integrity case-audit sample (standard library only)."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "integrity_results/v4/anomaly_screen"
CLUSTERS = BASE / "top_50_episode_clusters_per_family.csv"
REGRESSION_SAMPLE = ROOT / "regression_inputs/v3/primary_prematch_5m_complete_case.csv.gz"
MARKETS = ROOT / "raw/markets/markets.csv"
OUT = ROOT / "integrity_results/v4/case_audit"


def rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fnum(value, default=math.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def dt(value):
    if not value:
        return None
    value = value.strip().replace(" UTC", "+00:00")
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(value)
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def minute_string(value: str) -> str:
    parsed = dt(value)
    return parsed.strftime("%Y-%m-%dT%H:%M:00Z") if parsed else ""


def select_sample(clusters):
    selected = defaultdict(set)
    by_family = defaultdict(list)
    for i, row in enumerate(clusters):
        by_family[row["market_family"]].append(i)
        if row.get("cluster_any_method_agreement") == "1":
            selected[i].add("both_methods")
    for family, indices in by_family.items():
        top25 = sorted(indices, key=lambda i: fnum(clusters[i]["iforest_anomaly_score"], -math.inf), reverse=True)[:25]
        for i in top25:
            selected[i].add("iforest_top25_cluster")
        remaining = [i for i in indices if i not in selected]
        additions = sorted(
            remaining,
            key=lambda i: (
                fnum(clusters[i]["cluster_max_rule_score"], -math.inf),
                fnum(clusters[i]["iforest_anomaly_score"], -math.inf),
            ),
            reverse=True,
        )[:10]
        for i in additions:
            selected[i].add("rule_priority_addition")
    return selected


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Reading cluster rankings...", flush=True)
    clusters = list(rows(CLUSTERS))
    selected = select_sample(clusters)
    audit = []
    for i, reasons in selected.items():
        row = dict(clusters[i])
        row["selection_reason"] = ";".join(sorted(reasons))
        audit.append(row)
    print(f"Selected {len(audit)} unique clusters from {len(clusters):,}.", flush=True)

    hash_records = Counter()
    hash_clusters = defaultdict(set)
    hash_wallets = defaultdict(set)
    print("Checking transaction hashes within the frozen 100-cluster review frame...", flush=True)
    for row in clusters:
        tx = row.get("initiating_transaction_hash", "")
        hash_records[tx] += 1
        hash_clusters[tx].add(row.get("episode_cluster_id", ""))
        hash_wallets[tx].add(row.get("initiating_wallet", ""))

    target_match_markets = {r["market_id"] for r in audit if r["market_family"] == "match"}
    market_events = {}
    print("Recovering match timing metadata from the frozen V3 regression sample...", flush=True)
    for r in rows(REGRESSION_SAMPLE):
        market_id = r.get("market_id", "")
        if market_id in target_match_markets and market_id not in market_events:
            market_events[market_id] = {
                k: r.get(k, "") for k in (
                    "event_id", "fifa_match_id", "stage", "actual_kickoff_utc",
                    "home_team", "away_team", "market_role", "resolved_on_timestamp",
                    "yes_outcome_won",
                )
            }
        if len(market_events) == len(target_match_markets):
            break
    markets = {r["market_id"]: r for r in rows(MARKETS)}

    priority_map = {"both_methods": 0, "iforest_top25_cluster": 1, "rule_priority_addition": 2}
    warnings = Counter()
    for row in audit:
        tx = row.get("initiating_transaction_hash", "")
        row["transaction_hash_representative_count_in_review_frame"] = hash_records[tx]
        row["transaction_hash_cluster_count"] = len(hash_clusters[tx])
        row["transaction_hash_observed_wallet_count"] = len(hash_wallets[tx])
        row["transaction_hash_shared_across_clusters"] = int(len(hash_clusters[tx]) > 1)

        event = market_events.get(row["market_id"], {})
        market = markets.get(row["market_id"], {})
        for k in (
            "fifa_match_id", "stage", "actual_kickoff_utc", "home_team", "away_team",
            "result", "market_role", "resolved_outcome", "yes_outcome_won",
            "resolved_on_timestamp",
        ):
            row[k] = event.get(k, "")
        for k in ("condition_id", "market_type", "start_date", "end_date", "closed"):
            row[k] = market.get(k, "")

        t = dt(row.get("initiating_time_utc", ""))
        kickoff = dt(row.get("actual_kickoff_utc", ""))
        resolved = dt(row.get("resolved_on_timestamp", ""))
        market_end = dt(row.get("end_date", ""))
        row["minutes_from_actual_kickoff"] = "" if not (t and kickoff) else (t - kickoff).total_seconds() / 60
        row["minutes_from_market_end_metadata"] = "" if not (t and market_end) else (t - market_end).total_seconds() / 60
        row["minutes_from_resolution"] = "" if not (t and resolved) else (t - resolved).total_seconds() / 60
        if row["market_family"] == "match" and t and kickoff:
            if t < kickoff:
                phase = "pre_kickoff"
            elif resolved and t >= resolved:
                phase = "post_resolution"
            else:
                phase = "post_kickoff_pre_resolution"
        elif row["market_family"] == "country_outright" and t and market_end:
            phase = "pre_metadata_end" if t < market_end else "post_metadata_end"
        else:
            phase = "unknown"
        row["timing_phase"] = phase

        price_fields = [f"yes_price_{h}m" for h in (5, 15, 30, 60)]
        available = sum(bool(row.get(k, "")) for k in price_fields)
        row["future_price_points_available"] = available
        row["all_future_prices_missing"] = int(available == 0)
        lateness = [fnum(row.get(f"price_lateness_seconds_{h}m")) for h in (5, 15, 30, 60)]
        lateness = [x for x in lateness if not math.isnan(x)]
        row["max_future_price_lateness_seconds"] = max(lateness) if lateness else ""
        if not row.get("baseline_yes_price", ""):
            warning = "missing_baseline_price"
        elif available == 0:
            warning = "missing_all_future_prices"
        elif phase == "post_resolution":
            warning = "trade_at_or_after_resolution"
        elif lateness and max(lateness) > 60:
            warning = "future_price_lateness_over_60s"
        else:
            warning = "none"
        row["automated_data_warning"] = warning
        warnings[warning] += 1
        row["manual_disposition"] = "pending_review"
        row["manual_confidence"] = ""
        row["manual_evidence_note"] = ""
        row["public_source_url"] = ""
        row["reviewer"] = ""
        row["review_date"] = ""
        row["review_priority"] = min(priority_map[r] for r in row["selection_reason"].split(";"))

    audit.sort(key=lambda r: (int(r["review_priority"]), r["market_family"], -fnum(r["iforest_anomaly_score"], -math.inf)))
    for n, row in enumerate(audit, 1):
        row["audit_case_id"] = f"V4-CASE-{n:03d}"
    preferred = ["audit_case_id", "review_priority", "selection_reason"]
    fields = preferred + [k for k in audit[0] if k not in preferred]
    with (OUT / "case_audit.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(audit)

    codebook = [
        ("pending_review", "Not yet manually assessed"),
        ("data_artifact", "Demonstrable data or alignment issue explains the pattern"),
        ("event_information", "Contemporaneous match/resolution information plausibly explains it"),
        ("unusual_but_unresolved", "Unusual after checks; intent cannot be inferred"),
        ("insufficient_data", "Available public data cannot support assessment"),
        ("not_anomalous_after_context", "Ordinary context explains the multivariate ranking"),
    ]
    with (OUT / "disposition_codebook.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["code", "meaning"]); w.writerows(codebook)

    sample_counts = Counter((r["market_family"], r["selection_reason"]) for r in audit)
    with (OUT / "case_audit_sample_summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["market_family", "selection_reason", "cases"])
        for (family, reason), count in sorted(sample_counts.items()): w.writerow([family, reason, count])
    metadata = {
        "rows": len(audit), "unique_clusters": len({r["episode_cluster_id"] for r in audit}),
        "unique_markets": len({r["market_id"] for r in audit}),
        "source_cluster_sha256": sha256(CLUSTERS),
        "timing_metadata_source": str(REGRESSION_SAMPLE.relative_to(ROOT)),
        "timing_markets_requested": len(target_match_markets),
        "timing_markets_recovered": len(market_events),
        "selection": "within frozen top-50 cluster queue per family: union(method-agreement clusters, top 25 IF clusters/family, 10 rule-priority additions/family)",
        "manual_fields_initialised_blank": True,
        "interpretation_boundary": "No manipulation label permitted",
        "automated_warning_counts": dict(warnings),
    }
    with (OUT / "case_audit_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(json.dumps(metadata, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
