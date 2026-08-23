#!/usr/bin/env python3
"""Create the reproducible H2c coefficient summary and research report."""
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "regression_results/v4/h2c/match_pre"
DOC = ROOT / "docs/results/v4_h2c_results.md"


def main():
    paths = {
        "joint": RESULT / "coefficients.csv",
        "imbalance_only": RESULT / "imbalance/coefficients.csv",
        "signed_log_only": RESULT / "signed_log/coefficients.csv",
    }
    frames = []
    for specification, path in paths.items():
        frame = pd.read_csv(path)
        frame = frame[frame["term"].str.startswith("follower_")].copy()
        frame.insert(0, "specification", specification)
        frames.append(frame)
    summary = pd.concat(frames, ignore_index=True)
    summary.to_csv(RESULT / "h2c_follower_coefficients.csv", index=False)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "Stronger other-wallet following after a P99 trade predicts subsequent price reversal.",
        "primary_sample": "match markets, pre-kickoff, P99 events with positive initial 5-minute directional move",
        "specifications": list(paths),
        "models": int(summary.shape[0]),
        "p_below_0_05": int((summary.p_value < 0.05).sum()),
        "minimum_p_value": float(summary.p_value.min()),
        "conclusion": "H2c is not supported in the primary pre-match sample.",
        "limitations": [
            "The first-five-minute follower-flow measures overlap temporally with the initial price move.",
            "Associations are conditional and do not establish causality.",
            "A null result is not evidence that manipulation or overreaction never occurs.",
        ],
    }
    (RESULT / "h2c_summary.json").write_text(json.dumps(payload, indent=2) + "\n")

    rows = summary.copy()
    rows["estimate"] = rows.estimate.map(lambda x: f"{x:.6f}")
    rows["std_error"] = rows.std_error.map(lambda x: f"{x:.6f}")
    rows["p_value"] = rows.p_value.map(lambda x: f"{x:.4f}")
    table = rows[["specification", "model", "term", "estimate", "std_error", "p_value"]].to_markdown(index=False)
    text = f"""# V4 H2c Results: Following and Subsequent Reversal

## Hypothesis

H2c proposes that stronger other-wallet following after a P99 large trade predicts a subsequent price reversal.

## Primary design

The primary sample contains pre-kickoff match-market P99 events with a positive initial five-minute directional price move. Partial and full reversals are evaluated after 15, 30, and 60 minutes. Linear probability models include market and UTC calendar-hour fixed effects and standard errors clustered by match event. Controls include the initial price move, initiating-trade value, initiating-minute wallet concentration, and baseline YES price.

## Results

{table}

Across all {payload['models']} reported follower coefficients, none is statistically significant at the 5% level. The smallest p-value is {payload['minimum_p_value']:.4f}. Separating follower imbalance and signed-log net flow therefore does not overturn the null results from the joint specifications.

## Decision

**H2c is not supported in the primary pre-match sample.** The results do not show that stronger following after a large trade systematically predicts later reversal. This contrasts with H2a, which finds evidence of short-run following in match markets: following may occur without producing detectable subsequent overreaction.

## Interpretation limits

This is a conditional association, not a causal test. The follower-flow variables and initial price move both use the first five minutes after the initiating trade. The null result does not prove that manipulation or overreaction is absent; it means the specified measures do not detect a systematic relationship in this sample.
"""
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(text)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
