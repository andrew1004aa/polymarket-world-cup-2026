#!/usr/bin/env python3
"""Create a signed audit copy and final integrity conclusion table."""
import csv, json, hashlib
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"integrity_results/v4/case_audit/case_audit.csv"
SIGNED=ROOT/"integrity_results/v4/case_audit/case_audit_signed.csv"
SUMMARY=ROOT/"integrity_results/v4/case_audit/signed_review_summary.json"
CONCLUSIONS=ROOT/"integrity_results/v4/case_audit/market_integrity_conclusion_table.csv"
REVIEWER="CHIHYU LIANG"
REVIEW_DATE="2026-08-15"

def digest(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()

with SOURCE.open(newline="",encoding="utf-8") as f:data=list(csv.DictReader(f));fields=list(data[0])
if len(data)!=73 or any(not r["provisional_disposition"] for r in data):raise RuntimeError("Provisional review is incomplete")
for r in data:
 r["manual_disposition"]=r["provisional_disposition"]
 r["manual_confidence"]=r["provisional_confidence"]
 r["manual_evidence_note"]=r["provisional_evidence_note"]
 r["reviewer"]=REVIEWER
 r["review_date"]=REVIEW_DATE
 r["requires_researcher_confirmation"]="0"
with SIGNED.open("w",newline="",encoding="utf-8") as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)

counts=Counter(r["manual_disposition"] for r in data)
units={r["related_case_group"] or r["audit_case_id"] for r in data}
unresolved={r["related_case_group"] or r["audit_case_id"] for r in data if r["manual_disposition"]=="unusual_but_unresolved"}
summary={
 "status":"SIGNED","reviewer":REVIEWER,"review_date":REVIEW_DATE,
 "acceptance_basis":"User instructed continuation after receiving the complete provisional review and explicit signing next-step notice.",
 "source_file":str(SOURCE.relative_to(ROOT)),"source_sha256":digest(SOURCE),
 "signed_file":str(SIGNED.relative_to(ROOT)),"signed_sha256":digest(SIGNED),
 "cases":len(data),"dispositions":dict(counts),"linked_or_standalone_units":len(units),
 "unusual_unresolved_units":len(unresolved),
 "interpretation_boundary":"No case establishes manipulation, maker identity, ownership, or intent.",
}
with SUMMARY.open("w",encoding="utf-8") as f:json.dump(summary,f,indent=2)

rows=[
 {"component":"H3a","method":"Market and calendar-hour FE; event-clustered SE; two-part robustness","sample":"748,481 match pre-kickoff market-minutes","main_result":"HHI predicts a higher probability of any 5-minute price update (beta 0.040424, SE 0.003687, p<0.001); conditional nonzero magnitude is small and not robust across concentration proxies.","decision":"Conditionally supported","permitted_interpretation":"Concentration is associated mainly with whether prices update, not reliably with larger jumps."},
 {"component":"H3b","method":"Two-way FWL fixed effects; event-clustered CRV1","sample":"19,256 positive initial directional-move episodes","main_result":"Concentration-by-directional-pressure coefficients are not significant for partial or full reversal at 15, 30, or 60 minutes; all Holm-adjusted p-values equal 1.","decision":"Not supported","permitted_interpretation":"No systematic evidence that concentrated directional trading predicts reversal."},
 {"component":"H3c","method":"Grouped nested CV and chronological match-event holdout; Logistic, Elastic Net, HGB, Random Forest","sample":"5,843 match pre-kickoff P99 events; 748 reversals","main_result":"Expanded random forest reaches PR-AUC 0.264, F1 0.336 and ROC-AUC 0.720 in grouped CV; chronological expanded RF PR-AUC 0.337, F1 0.423 and ROC-AUC 0.791. Cluster-bootstrap discrimination increments are positive.","decision":"Supported","permitted_interpretation":"Expanded behavioral/concentration features contain incremental out-of-sample reversal information; prediction is not causation."},
 {"component":"Unsupervised anomaly screen","method":"Transparent five-rule score plus within-family Isolation Forest","sample":"67,845 P99 episodes; 44,232 market-minute clusters","main_result":"104 match and 38 outright episodes score at least 4; Isolation Forest top 1% contains 534 match and 145 outright episodes; method overlap is 20 episodes.","decision":"Completed","permitted_interpretation":"Ranks unusual multivariate episodes for review; scores are not manipulation probabilities."},
 {"component":"Contextual case audit","method":"Pre-specified top-cluster review with neutral disposition codes","sample":"73 cases representing 35 linked or standalone units","main_result":"38 event-information cases, 18 not anomalous after context, 12 unusual but unresolved, and 5 insufficient-data cases; unresolved rows collapse to 8 units.","decision":"Completed and signed","permitted_interpretation":"Most top anomalies reflect live/terminal information; eight units merit cautious qualitative discussion, none proves misconduct."},
 {"component":"Market-making-like case","method":"Cross-market observed-wallet sequence comparison","sample":"Australia and Bosnia outright Cases 060-061","main_result":"Events occur 45 seconds apart, show near-equal opposite-direction value, near-zero imbalance and zero price movement.","decision":"Illustrative evidence only","permitted_interpretation":"Consistent with systematic two-sided or inventory-adjustment behavior; cannot authenticate maker identity or intent."},
]
with CONCLUSIONS.open("w",newline="",encoding="utf-8") as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(summary,indent=2))
