#!/usr/bin/env python3
"""Build summary statistics and a correlation matrix for the v3 primary
pre-kickoff match sample. Reads the frozen regression_inputs/v3 complete-case
file only; writes new versioned outputs and never modifies frozen inputs.

Outputs:
  regression_results/v3/descriptive/summary_statistics.csv
  regression_results/v3/descriptive/correlation_matrix.csv
  regression_results/v3/descriptive/run_log.json
  regression_results/v3/descriptive/README.md
  overleaf/tables/table_summary_statistics.tex
  overleaf/tables/table_correlation_matrix.tex

This is a purely descriptive step: no hypothesis test, p-value, or
significance marker is attached to any statistic below.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "regression_inputs/v3/primary_prematch_5m_complete_case.csv.gz"
OUT = ROOT / "regression_results/v3/descriptive"
TABLES = ROOT / "overleaf/tables"

# Variables used in the primary v3 match-market models (H1/H2), in the order
# they should appear in the descriptive tables.
VARIABLES = [
    ("delta_yes_price", "5-minute $\\Delta$ YES-equivalent price"),
    ("signed_log_total_net_flow", "Signed-log total net flow"),
    ("p99_signed_log_large_net_flow", "Signed-log P99 large-trade net flow"),
    ("p99_signed_log_ordinary_net_flow", "Signed-log ordinary-trade net flow"),
    ("yes_price_t", "Baseline YES-equivalent price"),
    ("lagged_30m_price_change", "Lagged 30-minute price change"),
    ("log_lagged_30m_volume", "Log lagged 30-minute volume"),
    ("log_minutes_to_kickoff", "Log minutes to kickoff"),
]
SHORT_LABELS = [
    "$\\Delta$ price", "Total flow", "Large flow", "Ordinary flow",
    "Baseline price", "Lag 30m $\\Delta$", "Log lag vol.", "Log min. to kickoff",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fmt(x: float, decimals: int = 6) -> str:
    return f"{x:.{decimals}f}"


def build_summary_table(stats: pd.DataFrame) -> list[str]:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Summary statistics: primary pre-kickoff match-market sample}",
        r"\label{tab:summary-statistics}",
        r"\scriptsize",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\hline",
        r"Variable & N & Mean & SD & Min & P25 & Median & P75 & Max \\",
        r"\hline",
    ]
    for (_, label), row in zip(VARIABLES, stats.itertuples(index=False)):
        lines.append(
            f"{label} & {int(row.n):,} & {fmt(row.mean)} & {fmt(row.std)} & "
            f"{fmt(row.min)} & {fmt(row.p25)} & {fmt(row.median)} & "
            f"{fmt(row.p75)} & {fmt(row.max)} " + r"\\"
        )
    lines += [
        r"\hline",
        r"\multicolumn{9}{p{0.96\textwidth}}{\footnotesize Notes: Statistics are "
        r"computed on the 748,481-observation primary pre-kickoff match-market "
        r"sample (312 match markets nested in 104 events, five-minute horizon). "
        r"Flow variables are signed-log transformed: "
        r"$\mathrm{sign}(\mathrm{flow})\times\log(1+|\mathrm{flow}|)$. Large "
        r"trades are individual trades at or above their market-specific P99 "
        r"threshold; ordinary trades are all remaining trades. These are "
        r"descriptive statistics only and involve no hypothesis test.}",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return lines


def build_correlation_table(corr: pd.DataFrame) -> list[str]:
    ncols = len(SHORT_LABELS)
    col_spec = "l" + "r" * ncols
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Pairwise Pearson correlations: primary pre-kickoff match-market sample}",
        r"\label{tab:correlation-matrix}",
        r"\scriptsize",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\hline",
        " & " + " & ".join(SHORT_LABELS) + r" \\",
        r"\hline",
    ]
    codes = [code for code, _ in VARIABLES]
    for code, label in zip(codes, SHORT_LABELS):
        vals = " & ".join(fmt(corr.loc[code, other], 3) for other in codes)
        lines.append(f"{label} & {vals} " + r"\\")
    lines += [
        r"\hline",
        rf"\multicolumn{{{ncols+1}}}{{p{{0.96\textwidth}}}}{{\footnotesize Notes: "
        r"Pairwise Pearson correlation coefficients computed on the 748,481-observation "
        r"primary pre-kickoff match-market sample. Correlations are descriptive only "
        r"and are not given a causal interpretation.}",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return lines


def main() -> int:
    input_hash = sha256(INPUT)
    usecols = [code for code, _ in VARIABLES]
    df = pd.read_csv(INPUT, usecols=usecols)

    missing = {c: int(df[c].isna().sum()) for c in usecols}
    nonfinite = {c: int((~np.isfinite(df[c])).sum() - df[c].isna().sum()) for c in usecols}

    rows = []
    for code, label in VARIABLES:
        s = df[code]
        rows.append({
            "variable": code,
            "label": label,
            "n": int(s.notna().sum()),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "p25": float(s.quantile(0.25)),
            "median": float(s.quantile(0.50)),
            "p75": float(s.quantile(0.75)),
            "max": float(s.max()),
        })
    stats = pd.DataFrame(rows)

    corr = df[usecols].corr(method="pearson")

    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    stats.to_csv(OUT / "summary_statistics.csv", index=False)
    corr.to_csv(OUT / "correlation_matrix.csv")

    (TABLES / "table_summary_statistics.tex").write_text(
        "\n".join(build_summary_table(stats)) + "\n", encoding="utf-8"
    )
    (TABLES / "table_correlation_matrix.tex").write_text(
        "\n".join(build_correlation_table(corr)) + "\n", encoding="utf-8"
    )

    run_log = {
        "version": "v3",
        "step": "descriptive_statistics",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": str(INPUT.relative_to(ROOT)),
        "input_sha256": input_hash,
        "rows": int(len(df)),
        "variables": usecols,
        "missing_values": missing,
        "nonfinite_values": nonfinite,
        "software": {"python": "3.13", "pandas": pd.__version__, "numpy": np.__version__},
        "outputs": [
            "regression_results/v3/descriptive/summary_statistics.csv",
            "regression_results/v3/descriptive/correlation_matrix.csv",
            "overleaf/tables/table_summary_statistics.tex",
            "overleaf/tables/table_correlation_matrix.tex",
        ],
    }
    (OUT / "run_log.json").write_text(json.dumps(run_log, indent=2) + "\n", encoding="utf-8")

    readme = f"""# V3 descriptive statistics (primary pre-kickoff match sample)

Generated {run_log['generated_at']} from the frozen input
`regression_inputs/v3/primary_prematch_5m_complete_case.csv.gz`
(sha256 `{input_hash}`). {len(df):,} rows, no missing or non-finite values in
any of the eight model variables.

`summary_statistics.csv` reports count, mean, standard deviation, minimum,
P25, median, P75 and maximum for the dependent variable and the seven
regressors used in the primary H1/H2 match-market models.

`correlation_matrix.csv` reports the 8x8 pairwise Pearson correlation matrix
for the same variables. Both files are purely descriptive: no hypothesis test,
p-value, or causal claim is attached to any entry.

Regenerate with `python3 scripts/build_v3_descriptive_stats.py` after
activating `.venv-regression`. Do not edit the CSV or `.tex` outputs by hand.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    print(json.dumps({"rows": len(df), "variables": len(usecols), "missing_total": sum(missing.values())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
