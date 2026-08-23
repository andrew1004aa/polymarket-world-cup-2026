#!/usr/bin/env python3
"""Consolidate completed H2b trade- and market-minute results."""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "regression_results/v4/h2b"
SPECS = [
    ("match_pre", "5m", BASE / "match_pre_5m"),
    ("match_pre", "15m", BASE / "match_pre_15m"),
    ("outright", "5m", BASE / "outright_5m"),
    ("outright", "15m", BASE / "outright_15m"),
    ("match_post_market_minute", "5m", BASE / "match_post_5m_market_minute"),
    ("match_post_paired_minute", "5m", BASE / "match_post_5m_paired_market_minute"),
]

rows = []
for sample, horizon, directory in SPECS:
    config = json.loads((directory / "config.json").read_text())
    with (directory / "coefficients.csv").open(newline="") as stream:
        coefficient = next(row for row in csv.DictReader(stream) if row["term"] == "is_p99_large")
    diagnostics = config["diagnostics"]
    rows.append({
        "sample": sample, "horizon": horizon, "unit": config.get("unit", "trade"),
        "observations": diagnostics["observations"], "clusters": diagnostics["clusters"],
        "momentum_share": diagnostics.get("momentum_share", diagnostics.get("unweighted_mean_cell_momentum_share")),
        "estimate": float(coefficient["estimate"]), "std_error": float(coefficient["std_error"]),
        "p_value": float(coefficient["p_value"]), "conf_low": float(coefficient["conf_low"]),
        "conf_high": float(coefficient["conf_high"]),
    })

deferred = BASE / "match_post_5m/config.json"
if deferred.exists():
    config = json.loads(deferred.read_text())
    config.update({
        "status": "superseded_without_estimation",
        "replacement_completed_at": datetime.now(timezone.utc).isoformat(),
        "replacement_primary": "regression_results/v4/h2b/match_post_5m_market_minute",
        "replacement_robustness": "regression_results/v4/h2b/match_post_5m_paired_market_minute",
        "reason": "The 1,978,216-row trade-level design was replaced by equal-weight market-minute-by-trade-class cells so dense minutes do not dominate.",
    })
    deferred.write_text(json.dumps(config, indent=2) + "\n")

with (BASE / "h2b_large_trade_summary.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

payload = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "status": "complete_including_postkickoff_market_minute_replacement",
    "hypothesis": "H2b", "models": rows,
    "interpretation": "Conditional associations with momentum versus contrarian classification; not proof of a behavioural bias or causality.",
}
(BASE / "h2b_summary.json").write_text(json.dumps(payload, indent=2) + "\n")

lines = [
    "# V4 H2b: Momentum versus contrarian trade classification", "",
    "The binary trade-level samples condition on an observable non-zero recent price change. Momentum equals one when the YES-equivalent trade direction matches the recent price direction; contrarian equals zero. Models include market and UTC calendar-hour fixed effects and control for trade value, absolute recent price change and current YES probability.", "",
    "| Sample | Horizon | Unit | N | Clusters | Large-trade probability difference | SE | p-value | 95% CI |",
    "|---|---:|---|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['sample']} | {row['horizon']} | {row['unit']} | {row['observations']:,} | "
        f"{row['clusters']} | {100*row['estimate']:.2f} pp | {100*row['std_error']:.2f} pp | "
        f"{row['p_value']:.3g} | [{100*row['conf_low']:.2f}, {100*row['conf_high']:.2f}] pp |"
    )
lines += [
    "", "P99 large trades are more likely than ordinary trades to align with recent price movements in the primary prematch sample and in the separate outright extension. This is consistent with differential momentum-style execution, but it does not by itself identify extrapolative beliefs, hot-hand bias, information, or causality.", "",
    "The 1,978,216-row trade-level post-kickoff specification was stopped before estimation and replaced by equal-weight market-minute-by-trade-class cells. The all-cell replacement contains 43,508 cells and the stricter paired-minute robustness sample contains 24,154 cells. Both retain market and UTC calendar-hour fixed effects and event-clustered inference. The positive large-trade difference remains significant in both designs, although its magnitude is smaller in the paired-minute comparison.",
]
(ROOT / "docs/results/v4_h2b_results.md").write_text("\n".join(lines) + "\n")
print(json.dumps(payload, indent=2))
