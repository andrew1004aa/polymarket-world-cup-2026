#!/usr/bin/env python3
"""Append provisional reviews for V4-CASE-051--073 without signing them."""
import csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/"integrity_results/v4/case_audit/case_audit.csv"
R={
"V4-CASE-051":("event_information","high","canada_morocco_inplay","Morocco buy-equivalent cluster occurs 22.9 minutes after kickoff. Price initially falls, then rises 0.280 by 60 minutes; Morocco is the recorded winner. The reversal is consistent with changing live match information."),
"V4-CASE-052":("event_information","high","germany_ivory_coast_postmatch","Germany buy-equivalent cluster occurs 118 minutes after kickoff at YES 0.955 and reaches approximately 0.9995. Germany is the recorded winner; terminal result information explains the move."),
"V4-CASE-053":("not_anomalous_after_context","high","argentina_egypt_postmatch_terminal","Argentina cluster occurs 125 minutes after kickoff at YES 0.9995 and produces no price change. It is terminal winning-market flow from the same event as Cases 035, 036, 040, and 041."),
"V4-CASE-054":("event_information","high","final_terminal_2026-07-19","Spain outright cluster occurs near tournament end. Price initially falls slightly, then rises 0.278 at 30 minutes and 0.338 at 60 minutes. It belongs to the final outcome episode."),
"V4-CASE-055":("unusual_but_unresolved","medium","morocco_outright_2026-06-30","Morocco outright cluster is adjacent to Case 021 and contains 14 P99 trades totaling USDC 324,978. A sell-direction response peaks at 15 minutes then returns to baseline by 30--60 minutes. Exact information timing and intent remain unavailable."),
"V4-CASE-056":("event_information","high","final_terminal_2026-07-19","Spain outright price rises from 0.6015 to approximately 0.9995 within 30 minutes near tournament end. Terminal outcome information is the direct explanation."),
"V4-CASE-057":("event_information","high","final_terminal_2026-07-19","Spain outright cluster at YES 0.9395 is briefly negative at five minutes then reaches approximately 0.9995. It is part of the linked final outcome episode."),
"V4-CASE-058":("event_information","high","england_terminal_2026-07-15","England sell-equivalent cluster begins at YES 0.1765 and falls roughly 0.176 to the terminal floor within 15 minutes. Elimination information explains the persistent movement."),
"V4-CASE-059":("event_information","high","england_terminal_2026-07-15","England sell-equivalent event at YES 0.007 falls to approximately 0.0005 and remains there. It is terminal elimination flow."),
"V4-CASE-060":("unusual_but_unresolved","medium","cross_market_two_sided_2026-06-26","Australia outright shows two same-minute P99 records, one observed wallet, near-zero minute imbalance, an opposite-direction value ratio of 0.999, and no price movement. Together with Bosnia Case 061, this is consistent with cross-market two-sided or inventory-adjustment behaviour, but maker status and intent are not observable."),
"V4-CASE-061":("unusual_but_unresolved","medium","cross_market_two_sided_2026-06-26","Bosnia outright occurs 45 seconds after Australia Case 060 with the same structural pattern: two P99 records, one observed wallet, near-zero imbalance, opposite-value ratio 0.998, and zero price movement. It is a useful market-making-like case, not proof of maker identity or manipulation."),
"V4-CASE-062":("event_information","medium","norway_terminal_2026-07-11","Norway sell-equivalent cluster starts at YES 0.0195 and falls about 0.019 to the floor by 60 minutes. Terminal tournament information is the plausible explanation; exact public event timing is not independently sourced."),
"V4-CASE-063":("event_information","high","knockout_information_2026-07-14_15","France sell-equivalent cluster starts at YES 0.1765 and falls approximately 0.176 to the terminal floor. It belongs to the linked France elimination repricing episode."),
"V4-CASE-064":("not_anomalous_after_context","high","portugal_spain_postmatch_terminal","Portugal cluster occurs 126 minutes after kickoff at the 0.0005 floor and produces no movement. It is the same post-match terminal burst as Cases 034 and 037--039."),
"V4-CASE-065":("not_anomalous_after_context","medium","japan_sweden_early_prematch","Japan event occurs about 42 hours before kickoff, is only 1.19 times P99 and about USDC 5,081, and produces no price movement. One-wallet concentration in an otherwise empty minute does not create an integrity concern."),
"V4-CASE-066":("not_anomalous_after_context","medium","egypt_iran_early_prematch","Draw-market event occurs about 121 hours before kickoff and is only USDC 3,347, 1.22 times P99. Despite an observed opposite trade, the small size and modest 0.03 price drift do not support a material integrity concern."),
"V4-CASE-067":("unusual_but_unresolved","medium","belgium_senegal_prematch","Belgium sell-equivalent cluster totals USDC 308,697, 53.6 minutes before kickoff, with HHI 0.807 and strongly opposite follower flow. Price moves only 0.01--0.02 against the trade and partly returns. The concentration is unusual but no manipulation mechanism or intent is observed."),
"V4-CASE-068":("not_anomalous_after_context","high","germany_curacao_postmatch_terminal","Curaçao win market occurs 219 minutes after kickoff at the 0.0005 floor, with no movement; Curaçao is the losing outcome. Terminal context explains the ranking."),
"V4-CASE-069":("event_information","high","jordan_algeria_kickoff","Algeria buy-equivalent cluster occurs 19.3 minutes before kickoff. Price rises 0.020 initially but reverses by 0.300 during the next hour of live play; Algeria ultimately wins. Crossing kickoff makes live information the direct competing explanation."),
"V4-CASE-070":("not_anomalous_after_context","medium","korea_czechia_early_prematch","Korea event occurs about 45 hours before kickoff, is almost exactly the P99 threshold at USDC 2,000, and produces no price movement. Sparse-minute concentration alone explains its rule score."),
"V4-CASE-071":("event_information","high","panama_croatia_inplay","Croatia buy-equivalent cluster occurs 74.7 minutes after kickoff and price rises persistently; Croatia is the recorded winner. Live match information explains the movement."),
"V4-CASE-072":("not_anomalous_after_context","high","ghana_panama_postmatch_terminal","Ghana cluster occurs 125 minutes after kickoff at YES 0.9995, Ghana is the winner, and price does not move. This is terminal winning-market flow."),
"V4-CASE-073":("not_anomalous_after_context","high","germany_ivory_coast_postmatch","Germany sell-equivalent cluster occurs 121 minutes after kickoff at YES 0.9995 and produces no movement. Recorded winning outcome and terminal price explain the flow."),
}
def main():
 with PATH.open(newline="",encoding="utf-8") as f:data=list(csv.DictReader(f));fields=list(data[0])
 found=set()
 for row in data:
  if row["audit_case_id"] in R:
   disposition,confidence,group,note=R[row["audit_case_id"]]
   row.update(provisional_disposition=disposition,provisional_confidence=confidence,related_case_group=group,provisional_evidence_note=note,requires_researcher_confirmation="1");found.add(row["audit_case_id"])
 if found!=set(R):raise RuntimeError(f"Missing cases: {sorted(set(R)-found)}")
 tmp=PATH.with_suffix(".csv.tmp")
 with tmp.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)
 tmp.replace(PATH);print(f"Added provisional reviews for {len(found)} cases; signed manual fields unchanged.")
if __name__=="__main__":main()
