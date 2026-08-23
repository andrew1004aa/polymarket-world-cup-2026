#!/usr/bin/env python3
"""Summarize all 73 provisional integrity reviews."""
import csv, json
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"integrity_results/v4/case_audit/case_audit.csv"
OUT=ROOT/"integrity_results/v4/case_audit"
with SOURCE.open(newline="",encoding="utf-8") as f:d=list(csv.DictReader(f))
if len(d)!=73 or any(not r["provisional_disposition"] for r in d):raise RuntimeError("Expected 73 completed provisional reviews")
counts=Counter((r["market_family"],r["provisional_disposition"]) for r in d)
with (OUT/"provisional_disposition_summary.csv").open("w",newline="",encoding="utf-8") as f:
 w=csv.writer(f);w.writerow(["market_family","provisional_disposition","cases"])
 for (family,decision),n in sorted(counts.items()):w.writerow([family,decision,n])
u=[r for r in d if r["provisional_disposition"]=="unusual_but_unresolved"]
fields=["audit_case_id","market_family","market_id","question","initiating_time_utc","timing_phase","related_case_group","provisional_confidence","initiating_trade_value_usdc","cluster_total_large_trade_value_usdc","trade_to_p99","initiating_minute_wallet_hhi","directional_price_change_5m","directional_price_change_30m","directional_price_change_60m","provisional_evidence_note"]
with (OUT/"unusual_unresolved_candidates.csv").open("w",newline="",encoding="utf-8") as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r.get(k,"") for k in fields} for r in u)
units={r["related_case_group"] or r["audit_case_id"] for r in d}
unresolved_units={r["related_case_group"] or r["audit_case_id"] for r in u}
summary={"cases":len(d),"provisional_complete":len(d),"manual_signed":sum(r["manual_disposition"]!="pending_review" for r in d),"dispositions":dict(Counter(r["provisional_disposition"] for r in d)),"confidence":dict(Counter(r["provisional_confidence"] for r in d)),"linked_or_standalone_units":len(units),"unusual_unresolved_cases":len(u),"unusual_unresolved_units":len(unresolved_units),"requires_researcher_confirmation":sum(r["requires_researcher_confirmation"]=="1" for r in d)}
with (OUT/"provisional_review_summary.json").open("w",encoding="utf-8") as f:json.dump(summary,f,indent=2)
print(json.dumps(summary,indent=2))
