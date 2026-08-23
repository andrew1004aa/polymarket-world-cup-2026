#!/usr/bin/env python3
"""Append reproducible, provisional reviews for V4-CASE-001--009."""

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "integrity_results/v4/case_audit/case_audit.csv"

REVIEWS = {
    "V4-CASE-001": (
        "unusual_but_unresolved", "medium", "",
        "Egypt outright: one USDC 199,594 sell-equivalent P99 event at YES 0.0045, 225.7 times the market P99; HHI 0.971 and aligned follower flow. The directional price response persists through 60 minutes with no reversal. Concentration is unusual, but terminal probability, one-sided wallet data, and absent country-resolution timing prevent an intent inference.",
    ),
    "V4-CASE-002": (
        "unusual_but_unresolved", "medium", "",
        "Portugal outright: one USDC 153,810 sell-equivalent event at YES 0.0715, 72.3 times P99; HHI 0.993. The observed wallet later trades in the opposite direction with 55.3% of initiating value, but the directional price response remains positive through 60 minutes rather than reversing. This is an unusual observed-wallet sequence, not evidence of inventory, profit, ownership, or manipulation.",
    ),
    "V4-CASE-003": (
        "not_anomalous_after_context", "high", "terminal_outcome_context",
        "Spain outright: cluster occurs 45.4 minutes after metadata end with YES at 0.9995. Two P99 records total USDC 3.300 million, but price is unchanged at 5 and 15 minutes and later prices are unavailable. The extreme size is consistent with terminal-market or settlement-related activity; no price distortion is observed. Exact redemption mechanics are unavailable in the Data API.",
    ),
    "V4-CASE-004": (
        "event_information", "high", "canada_morocco_match",
        "Morocco win market: five P99 trades total USDC 1.512 million 2.5 minutes before actual kickoff. The five-minute interval crosses kickoff; the directional move changes from +0.010 at 5 minutes to -0.070, -0.140, and -0.190 at 15, 30, and 60 minutes. The reversal occurs during live play, so match information is a direct competing explanation. Both-method anomaly status remains descriptive.",
    ),
    "V4-CASE-005": (
        "unusual_but_unresolved", "medium", "canada_morocco_match",
        "Canada win market: one USDC 299,999 sell-equivalent event 75.6 minutes before kickoff, 72.1 times P99; minute HHI 0.998. The observed wallet's opposite-direction value within 60 minutes is 3.27 times initiating value, while the Canada price gradually moves in the initiating sell direction. The sequence is unusual but does not show a price reversal and cannot reveal inventory or intent.",
    ),
    "V4-CASE-006": (
        "not_anomalous_after_context", "medium", "",
        "United States win market: two P99 records total USDC 924,283, 224 minutes before kickoff. Despite extreme size and HHI 0.986, the directional price change is zero through 30 minutes and only +0.010 at 60 minutes. With no short-run distortion or reversal, the multivariate anomaly ranking alone does not establish an integrity concern.",
    ),
    "V4-CASE-007": (
        "unusual_but_unresolved", "medium", "canada_morocco_match",
        "Canada win market: four P99 records total USDC 407,487, 66.6 minutes before kickoff; HHI 0.912 and rule score 5. The observed wallet's opposite-direction value is 4.02 times initiating value. Price is unchanged through 30 minutes and declines 0.010 by 60 minutes in the initiating sell direction. Unusual sequence, but no reversal or ownership/intent evidence.",
    ),
    "V4-CASE-008": (
        "unusual_but_unresolved", "medium", "canada_morocco_match",
        "Canada win market: one USDC 290,000 sell-equivalent event 52.9 minutes before kickoff; HHI 0.987. Observed opposite-direction value within 60 minutes is 4.98 times initiating value. Price is flat at 5 and 15 minutes, then moves -0.010 and -0.030 at 30 and 60 minutes, consistent with rather than reversing the initiating direction. Intent remains unidentified.",
    ),
    "V4-CASE-009": (
        "event_information", "high", "france_senegal_terminal",
        "France win market: two P99 records total USDC 1.247 million at YES 0.9995, 130.2 minutes after kickoff and about 20 minutes before recorded resolution. France is the winning outcome in the frozen metadata. Price remains unchanged at 5 and 15 minutes and later points are unavailable. Terminal match/result information plausibly explains the flow.",
    ),
}


def main():
    with PATH.open(newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f)); fields = list(records[0])
    new_fields = [
        "provisional_disposition", "provisional_confidence", "related_case_group",
        "provisional_evidence_note", "requires_researcher_confirmation",
    ]
    for field in new_fields:
        if field not in fields: fields.append(field)
    found = set()
    for row in records:
        case = row["audit_case_id"]
        if case in REVIEWS:
            disposition, confidence, group, note = REVIEWS[case]
            row.update({
                "provisional_disposition": disposition,
                "provisional_confidence": confidence,
                "related_case_group": group,
                "provisional_evidence_note": note,
                "requires_researcher_confirmation": "1",
            })
            found.add(case)
        else:
            for field in new_fields: row.setdefault(field, "")
    if found != set(REVIEWS):
        raise RuntimeError(f"Missing cases: {sorted(set(REVIEWS) - found)}")
    temporary = PATH.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(records)
    temporary.replace(PATH)
    print(f"Added provisional reviews for {len(found)} cases; manual dispositions remain unchanged.")


if __name__ == "__main__":
    main()
