#!/usr/bin/env python3
"""Append provisional reviews for V4-CASE-010--025 without signing them."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "integrity_results/v4/case_audit/case_audit.csv"

R = {
"V4-CASE-010": ("insufficient_data","high","france_norway_2026-07-10_missing_prices","France outright cluster has no baseline or 5/15/30/60-minute prices. Trading concentration is measurable, but price distortion or reversal cannot be assessed."),
"V4-CASE-011": ("event_information","high","final_terminal_2026-07-19","Spain outright cluster occurs 123.5 minutes before metadata end at YES 0.9285. Price reaches approximately 0.9995 within five minutes and remains there. The observed wallet later trades oppositely, but terminal tournament information is the direct price explanation."),
"V4-CASE-012": ("insufficient_data","high","france_norway_2026-07-10_missing_prices","France outright cluster has no baseline or future price path. Its extreme concentration can be ranked, but market impact cannot be evaluated."),
"V4-CASE-013": ("insufficient_data","high","france_norway_2026-07-10_missing_prices","France outright cluster is adjacent to Cases 010, 012, and 014 and has no baseline or future price observations. It is part of one missing-price episode rather than an independent price-distortion case."),
"V4-CASE-014": ("insufficient_data","high","france_norway_2026-07-10_missing_prices","France outright cluster has aligned follower flow but no baseline or future price observations. The effect on price is unidentifiable."),
"V4-CASE-015": ("insufficient_data","high","france_norway_2026-07-10_missing_prices","Norway outright cluster occurs in the same broad July 10 missing-price period as the France clusters. No price path is available, preventing an integrity assessment."),
"V4-CASE-016": ("event_information","medium","knockout_information_2026-07-14_15","France outright sell-equivalent cluster starts at YES 0.054 and the directional response grows to 0.0535 by 60 minutes, nearly exhausting the YES value. Tournament-result information is a plausible dominant explanation; exact contemporaneous news is not independently sourced here."),
"V4-CASE-017": ("event_information","high","final_terminal_2026-07-19","Argentina outright sell-equivalent cluster occurs 153.6 minutes before metadata end. The YES price initially moves slightly against the trade, then falls by roughly 0.276 at 30 minutes and 0.342 at 60 minutes. The timing and terminal movement indicate tournament-outcome information rather than an isolated manipulation inference."),
"V4-CASE-018": ("event_information","medium","knockout_information_2026-07-14_15","England outright sell-equivalent cluster begins at YES 0.056 and loses about 0.0555 in the initiating direction by 15 minutes. Terminal knockout information plausibly explains the movement; the opposite-wallet proxy alone does not establish intent."),
"V4-CASE-019": ("event_information","high","final_terminal_2026-07-19","Spain outright cluster occurs 130.2 minutes before metadata end at YES 0.908 and reaches approximately 0.9995 by 15 minutes. This is a terminal tournament-information pattern despite an observed opposite trade."),
"V4-CASE-020": ("event_information","medium","knockout_information_2026-07-14_15","Spain outright buy-equivalent event starts near YES 0.509 and rises progressively through 60 minutes. The price path is persistent, not reversing, and is consistent with contemporaneous tournament information; exact public event timing remains uncited."),
"V4-CASE-021": ("unusual_but_unresolved","medium","morocco_outright_2026-06-30","Morocco outright sell-equivalent cluster contains six P99 trades and strongly aligned follower flow. The directional response peaks by 15 minutes and returns approximately to baseline by 60 minutes. The temporary movement is unusual, but exact match-news timing and intent are unavailable."),
"V4-CASE-022": ("event_information","high","knockout_information_2026-07-14_15","France outright cluster contains about USDC 1.642 million, 612.8 times the market P99 for its representative trade, at YES 0.0185. YES falls approximately 0.018 to its terminal floor and stays there. Tournament elimination information is the plausible explanation."),
"V4-CASE-023": ("event_information","medium","knockout_information_2026-07-14_15","Argentina outright cluster totals about USDC 1.170 million. A sell-equivalent move initially lowers YES but reverses strongly by 30 and 60 minutes. The sharp within-hour reversal is consistent with changing live tournament information; without order book and exact event news it cannot establish manipulation."),
"V4-CASE-024": ("event_information","high","final_terminal_2026-07-19","Spain outright cluster occurs 122.4 minutes before metadata end at YES 0.939 and reaches approximately 0.9995 within five minutes. It is part of the same terminal outcome episode as Cases 011 and 019."),
"V4-CASE-025": ("not_anomalous_after_context","high","terminal_near_zero_market","Japan outright representative trade is 701.4 times P99 but occurs at YES 0.0005 and produces zero price change through 60 minutes. Extreme relative size at the terminal price floor, without observed price distortion, is not an integrity signal by itself."),
}

def main():
    with PATH.open(newline="",encoding="utf-8") as f:
        data=list(csv.DictReader(f)); fields=list(data[0])
    for field in ["provisional_disposition","provisional_confidence","related_case_group","provisional_evidence_note","requires_researcher_confirmation"]:
        if field not in fields: fields.append(field)
    found=set()
    for row in data:
        if row["audit_case_id"] in R:
            disposition,confidence,group,note=R[row["audit_case_id"]]
            row.update(provisional_disposition=disposition,provisional_confidence=confidence,
                       related_case_group=group,provisional_evidence_note=note,
                       requires_researcher_confirmation="1")
            found.add(row["audit_case_id"])
    if found != set(R): raise RuntimeError(f"Missing cases: {sorted(set(R)-found)}")
    tmp=PATH.with_suffix(".csv.tmp")
    with tmp.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)
    tmp.replace(PATH)
    print(f"Added provisional reviews for {len(found)} cases; signed manual fields unchanged.")

if __name__=="__main__": main()
