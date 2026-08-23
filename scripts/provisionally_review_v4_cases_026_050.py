#!/usr/bin/env python3
"""Append provisional reviews for V4-CASE-026--050 without signing them."""
import csv
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/"integrity_results/v4/case_audit/case_audit.csv"
R={
"V4-CASE-026":("event_information","medium","knockout_information_2026-07-14_15","Spain outright buy-equivalent event near YES 0.5165 is followed by a persistent rise to +0.056 at 60 minutes. The low HHI and low minute imbalance weaken a dominant-wallet interpretation; tournament repricing is more plausible."),
"V4-CASE-027":("unusual_but_unresolved","medium","argentina_outright_2026-07-07","Argentina outright event is USDC 644,529, 183.6 times P99, with HHI 0.979. Price rises slightly then returns just below baseline by 60 minutes. The temporary response is unusual, but exact news timing and intent are unavailable."),
"V4-CASE-028":("event_information","high","final_terminal_2026-07-19","Argentina outright buy-equivalent cluster occurs near tournament end at YES 0.0965, but YES falls approximately 0.096 to the terminal floor within 15 minutes. Terminal outcome information dominates the interpretation."),
"V4-CASE-029":("not_anomalous_after_context","high","england_terminal_2026-07-15","England outright is already at YES 0.0015 and moves to the 0.0005 floor. The terminal price and negligible absolute movement explain the ranking after context."),
"V4-CASE-030":("event_information","high","knockout_information_2026-07-14_15","France outright cluster contains 13 P99 trades and YES falls from 0.0505 to approximately 0.0005 by 60 minutes. Knockout-result information is a direct explanation despite observed opposite-direction activity."),
"V4-CASE-031":("event_information","high","knockout_information_2026-07-14_15","France buy-equivalent event at YES 0.019 is followed by a fall to approximately 0.0005. The move against the initiating direction is terminal tournament repricing, not standalone manipulation evidence."),
"V4-CASE-032":("event_information","high","final_terminal_2026-07-19","Spain outright cluster occurs near tournament end and moves from YES 0.9165 to approximately 0.9995. It belongs to the same terminal outcome episode as Cases 011, 019, and 024."),
"V4-CASE-033":("event_information","high","england_terminal_2026-07-15","England buy-equivalent cluster at YES 0.018 is followed by a fall to approximately 0.0005. The outcome-driven terminal movement explains the reversal."),
"V4-CASE-034":("not_anomalous_after_context","high","portugal_spain_postmatch_terminal","Portugal loss market cluster occurs 123 minutes after kickoff at YES 0.0015 and moves only to the 0.0005 floor. It is one post-match terminal-flow burst."),
"V4-CASE-035":("event_information","high","argentina_egypt_postmatch_terminal","Argentina win market sell-equivalent cluster occurs 117 minutes after kickoff at YES 0.9055. Price rises to approximately 0.9995 against the trade as the winning result becomes terminal."),
"V4-CASE-036":("event_information","high","argentina_egypt_postmatch_terminal","Same Argentina--Egypt post-match minute sequence as Cases 035, 040, and 041. Price moves to the winning terminal value against the initiating sell direction."),
"V4-CASE-037":("not_anomalous_after_context","high","portugal_spain_postmatch_terminal","Portugal loss market terminal cluster at YES 0.0015, 124 minutes after kickoff. Extreme nominal volume produces only a 0.001 move to the price floor."),
"V4-CASE-038":("not_anomalous_after_context","high","portugal_spain_postmatch_terminal","Portugal loss market terminal cluster at YES 0.0015, part of the same post-match burst. Later prices are unavailable after the floor move."),
"V4-CASE-039":("not_anomalous_after_context","high","portugal_spain_postmatch_terminal","Portugal loss market terminal cluster at YES 0.0015, part of the same 21:03--21:07 UTC burst; it is not an independent integrity case."),
"V4-CASE-040":("event_information","high","argentina_egypt_postmatch_terminal","Argentina buy-equivalent cluster occurs 118.6 minutes after kickoff and moves from YES 0.907 to approximately 0.9995. Winning-result information explains the terminal repricing."),
"V4-CASE-041":("event_information","high","argentina_egypt_postmatch_terminal","Argentina buy-equivalent cluster occurs 120.7 minutes after kickoff and moves from YES 0.9325 to approximately 0.9995. It is part of the same post-match terminal sequence."),
"V4-CASE-042":("unusual_but_unresolved","medium","mexico_prematch_2026-06-24","Mexico win market has two P99 records totaling USDC 310,040 about 20.2 hours before kickoff; HHI 0.999. Price rises 0.010 and remains there through 60 minutes. The concentration is unusual, but no reversal, order-book evidence, or intent information exists."),
"V4-CASE-043":("event_information","high","germany_paraguay_inplay","Draw-market sell-equivalent cluster occurs 65 minutes after kickoff. Price is initially unchanged, then rises about 0.120 at 30 minutes and 0.655 at 60 minutes against the trade, consistent with decisive live match information; draw is the recorded winning outcome."),
"V4-CASE-044":("event_information","high","brazil_japan_postmatch_terminal","Brazil win market cluster occurs 120 minutes after kickoff at YES 0.965 and reaches approximately 0.9995. The recorded winning outcome and terminal timing explain the move."),
"V4-CASE-045":("event_information","high","panama_croatia_inplay","Croatia sell-equivalent cluster occurs 70 minutes after kickoff, but YES rises 0.310--0.390 against the trade; Croatia is the recorded winning outcome. Live match information is the direct explanation."),
"V4-CASE-046":("not_anomalous_after_context","high","belgium_egypt_postmatch_terminal","Belgium loss market trade occurs 392 minutes after kickoff at the 0.0005 floor and causes no price movement. Terminal settlement context explains the ranking."),
"V4-CASE-047":("event_information","high","france_senegal_late_match","France buy-equivalent cluster occurs 90 minutes after kickoff. YES rises from 0.795 to approximately 0.9995 and France is the recorded winner; late-match information explains the movement."),
"V4-CASE-048":("not_anomalous_after_context","high","spain_saudi_postmatch_terminal","Draw market occurs 136 minutes after kickoff at the 0.0005 floor, with no price movement; draw is not the winning outcome. Terminal context explains the extreme concentration metric."),
"V4-CASE-049":("event_information","high","usa_paraguay_inplay","United States buy-equivalent cluster occurs 2.3 minutes after kickoff. Price initially falls 0.020 then rises 0.260, 0.400, and 0.512 by 15, 30, and 60 minutes; USA is the recorded winner. This is a live-information path."),
"V4-CASE-050":("event_information","high","england_argentina_postmatch_terminal","Argentina win-market cluster occurs 114 minutes after semi-final kickoff at YES 0.88125 and rises toward the terminal winning value. Recorded outcome and timing explain the repricing."),
}

def main():
 with PATH.open(newline="",encoding="utf-8") as f:data=list(csv.DictReader(f));fields=list(data[0])
 found=set()
 for row in data:
  if row["audit_case_id"] in R:
   disposition,confidence,group,note=R[row["audit_case_id"]]
   row.update(provisional_disposition=disposition,provisional_confidence=confidence,related_case_group=group,provisional_evidence_note=note,requires_researcher_confirmation="1")
   found.add(row["audit_case_id"])
 if found!=set(R):raise RuntimeError(f"Missing cases: {sorted(set(R)-found)}")
 tmp=PATH.with_suffix(".csv.tmp")
 with tmp.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)
 tmp.replace(PATH);print(f"Added provisional reviews for {len(found)} cases; signed manual fields unchanged.")
if __name__=="__main__":main()
