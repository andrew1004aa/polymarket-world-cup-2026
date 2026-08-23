#!/usr/bin/env python3
"""Build the two v3 coefficient plots for the Results chapter.

Graph 1: large- vs ordinary-flow coefficients at the common 1/5/15-minute
horizons (source: robustness_results/v3/horizons/horizon_coefficients.csv).

Graph 2: large- vs ordinary-flow coefficients across the five pre-kickoff
time-to-kickoff bands (source: robustness_results/v3/h4/h4_timing_marginal_effects.csv).

Both source files are frozen v3 regression outputs and are only read here,
never modified. Error bars are the 95% confidence intervals already reported
in the source CSVs (t-distribution, 103 degrees of freedom, event clustering).
No new statistical test is computed in this script.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DATA = ROOT / "regression_results/v3/descriptive"
OUT_FIG = ROOT / "overleaf/figures"

# Okabe-Ito colorblind-safe palette; marker/linestyle also vary so the two
# series remain distinguishable on a black-and-white printout.
LARGE_STYLE = dict(color="#0072B2", marker="o", linestyle="-", label="P99 large-trade flow")
ORDINARY_STYLE = dict(color="#E69F00", marker="s", linestyle="--", label="Ordinary-trade flow")
DODGE = 0.08

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})


def errorbar_series(ax, x, df, style, label):
    ax.errorbar(
        x, df["estimate"], yerr=[df["estimate"] - df["conf_low"], df["conf_high"] - df["estimate"]],
        fmt=style["marker"], color=style["color"], linestyle=style["linestyle"],
        capsize=4, markersize=7, linewidth=1.4, elinewidth=1.4, label=label,
    )


def build_graph_1() -> pd.DataFrame:
    h = pd.read_csv(ROOT / "robustness_results/v3/horizons/horizon_coefficients.csv")
    horizons = [1, 5, 15]
    model_map = {1: "H1_SPLIT", 5: "H5_SPLIT", 15: "H15_SPLIT"}
    rows = []
    for hz in horizons:
        m = model_map[hz]
        for term, flow_type in [
            ("p99_signed_log_large_net_flow", "large"),
            ("p99_signed_log_ordinary_net_flow", "ordinary"),
        ]:
            r = h[(h.model == m) & (h.term == term)].iloc[0]
            rows.append({
                "horizon_minutes": hz, "flow_type": flow_type,
                "estimate": r.estimate, "std_error": r.std_error,
                "conf_low": r.conf_low, "conf_high": r.conf_high, "p_value": r.p_value,
            })
    data = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=300)
    x = pd.Series(range(len(horizons)))
    large = data[data.flow_type == "large"].set_index("horizon_minutes").loc[horizons]
    ordinary = data[data.flow_type == "ordinary"].set_index("horizon_minutes").loc[horizons]
    errorbar_series(ax, x - DODGE, large, LARGE_STYLE, LARGE_STYLE["label"])
    errorbar_series(ax, x + DODGE, ordinary, ORDINARY_STYLE, ORDINARY_STYLE["label"])
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{hz} minute{'s' if hz != 1 else ''}" for hz in horizons])
    ax.set_xlim(-0.5, len(horizons) - 0.5)
    ax.set_xlabel("Response horizon")
    ax.set_ylabel("Coefficient on signed-log net flow\n(YES-equivalent probability points)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_1_horizon_large_vs_ordinary.png")
    fig.savefig(OUT_FIG / "fig_1_horizon_large_vs_ordinary.pdf")
    plt.close(fig)
    return data


def build_graph_2() -> pd.DataFrame:
    m = pd.read_csv(ROOT / "robustness_results/v3/h4/h4_timing_marginal_effects.csv")
    bands = ["gt_24h", "12_to_24h", "6_to_12h", "1_to_6h", "last_60m"]
    band_labels = {
        "gt_24h": ">24 hours", "12_to_24h": "12–24 hours", "6_to_12h": "6–12 hours",
        "1_to_6h": "1–6 hours", "last_60m": "Final 60 minutes",
    }
    data = m[m.band.isin(bands) & m.flow_type.isin(["large", "ordinary"])].copy()
    data["band"] = pd.Categorical(data["band"], categories=bands, ordered=True)
    data = data.sort_values(["flow_type", "band"])

    fig, ax = plt.subplots(figsize=(7.0, 4.4), dpi=300)
    x = pd.Series(range(len(bands)))
    large = data[data.flow_type == "large"].set_index("band").loc[bands]
    ordinary = data[data.flow_type == "ordinary"].set_index("band").loc[bands]
    errorbar_series(ax, x - DODGE, large, LARGE_STYLE, LARGE_STYLE["label"])
    errorbar_series(ax, x + DODGE, ordinary, ORDINARY_STYLE, ORDINARY_STYLE["label"])
    ax.set_xticks(list(x))
    ax.set_xticklabels([band_labels[b] for b in bands])
    ax.set_xlim(-0.5, len(bands) - 0.5)
    ax.set_xlabel("Time to kickoff (approaching kickoff →)")
    ax.set_ylabel("Coefficient on signed-log net flow\n(YES-equivalent probability points)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig_2_h4_timing_large_vs_ordinary.png")
    fig.savefig(OUT_FIG / "fig_2_h4_timing_large_vs_ordinary.pdf")
    plt.close(fig)
    return data


def main() -> int:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_DATA.mkdir(parents=True, exist_ok=True)

    g1 = build_graph_1()
    g2 = build_graph_2()
    g1.to_csv(OUT_DATA / "graph_1_horizon_large_vs_ordinary.csv", index=False)
    g2.to_csv(OUT_DATA / "graph_2_h4_timing_large_vs_ordinary.csv", index=False)

    run_log = {
        "version": "v3",
        "step": "coefficient_plots",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": [
            "robustness_results/v3/horizons/horizon_coefficients.csv",
            "robustness_results/v3/h4/h4_timing_marginal_effects.csv",
        ],
        "error_bars": "95% CI, t-distribution, df=103, event-clustered CRV1 standard errors (already computed in source CSVs; no new test run here)",
        "palette": "Okabe-Ito colorblind-safe (large=#0072B2 circle solid, ordinary=#E69F00 square dashed)",
        "outputs": [
            "overleaf/figures/fig_1_horizon_large_vs_ordinary.png",
            "overleaf/figures/fig_1_horizon_large_vs_ordinary.pdf",
            "overleaf/figures/fig_2_h4_timing_large_vs_ordinary.png",
            "overleaf/figures/fig_2_h4_timing_large_vs_ordinary.pdf",
            "regression_results/v3/descriptive/graph_1_horizon_large_vs_ordinary.csv",
            "regression_results/v3/descriptive/graph_2_h4_timing_large_vs_ordinary.csv",
        ],
    }
    (OUT_DATA / "plots_run_log.json").write_text(json.dumps(run_log, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"graph_1_rows": len(g1), "graph_2_rows": len(g2)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
