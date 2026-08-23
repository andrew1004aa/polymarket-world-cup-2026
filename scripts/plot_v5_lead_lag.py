#!/usr/bin/env python3
"""Create the frozen v5 lead-lag figure from stored coefficient contrasts."""

import csv
from pathlib import Path

from reportlab.lib.colors import Color, HexColor, black, white
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "regression_results/v5/lead_lag/lead_lag_contrasts.csv"
OUTPUT = ROOT / "overleaf/figures/fig_v5_lead_lag.pdf"

HORIZONS = [-15, -5, -1, 1, 5, 15, 30]
SERIES = [
    ("expanding", "Expanding history", HexColor("#0072B2"), "circle", -5),
    ("rolling7", "Rolling seven day", HexColor("#D55E00"), "square", 5),
]


def load_rows():
    rows = {}
    with INPUT.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (row["specification"], int(row["signed_horizon_minutes"]))
            rows[key] = {
                "estimate": float(row["estimate"]) * 100,
                "low": float(row["conf_low"]) * 100,
                "high": float(row["conf_high"]) * 100,
                "holm": float(row["holm_p_value_within_spec"]),
            }
    return rows


def marker(pdf, x, y, shape, color, filled=True):
    pdf.setStrokeColor(color)
    pdf.setFillColor(color if filled else white)
    if shape == "circle":
        pdf.circle(x, y, 3.4, stroke=1, fill=1)
    else:
        pdf.rect(x - 3.2, y - 3.2, 6.4, 6.4, stroke=1, fill=1)


def main():
    rows = load_rows()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    width, height = 540, 330
    left, right, bottom, top = 72, 22, 55, 42
    plot_w, plot_h = width - left - right, height - bottom - top
    x_inset = 11
    xs = [
        left + x_inset + i * (plot_w - 2 * x_inset) / (len(HORIZONS) - 1)
        for i in range(len(HORIZONS))
    ]

    all_low = [rows[(spec, h)]["low"] for spec, *_ in SERIES for h in HORIZONS]
    all_high = [rows[(spec, h)]["high"] for spec, *_ in SERIES for h in HORIZONS]
    y_min = min(-0.0015, min(all_low) - 0.001)
    y_max = max(all_high) + 0.001

    def y_map(value):
        return bottom + (value - y_min) / (y_max - y_min) * plot_h

    pdf = canvas.Canvas(str(OUTPUT), pagesize=(width, height))
    pdf.setTitle("Past-only lead-lag large-minus-ordinary coefficient path")

    # Light pre-/post-trade backgrounds.
    split = (xs[2] + xs[3]) / 2
    pdf.setFillColor(Color(0.92, 0.94, 0.97, alpha=1))
    pdf.rect(left, bottom, split - left, plot_h, stroke=0, fill=1)
    pdf.setFillColor(Color(0.98, 0.95, 0.90, alpha=1))
    pdf.rect(split, bottom, left + plot_w - split, plot_h, stroke=0, fill=1)

    # Horizontal grid and percentage-point scale.
    ticks = [-0.001, 0.000, 0.005, 0.010, 0.015, 0.020]
    for tick in ticks:
        if y_min <= tick <= y_max:
            y = y_map(tick)
            pdf.setStrokeColor(HexColor("#D0D0D0"))
            pdf.setLineWidth(0.45)
            pdf.line(left, y, left + plot_w, y)
            pdf.setFillColor(black)
            pdf.setFont("Helvetica", 8)
            pdf.drawRightString(left - 8, y - 2.8, f"{tick:.3f}")

    # Axes and trade-time divider.
    pdf.setStrokeColor(black)
    pdf.setLineWidth(0.8)
    pdf.rect(left, bottom, plot_w, plot_h, stroke=1, fill=0)
    pdf.setDash(3, 2)
    pdf.line(split, bottom, split, bottom + plot_h)
    pdf.setDash()
    pdf.setLineWidth(1.0)
    pdf.line(left, y_map(0), left + plot_w, y_map(0))

    # Series, confidence intervals, and Holm-significance marker fill.
    for spec, label, color, shape, offset in SERIES:
        points = []
        for x, horizon in zip(xs, HORIZONS):
            item = rows[(spec, horizon)]
            px = x + offset
            py = y_map(item["estimate"])
            points.append((px, py))
            pdf.setStrokeColor(color)
            pdf.setLineWidth(1.0)
            pdf.line(px, y_map(item["low"]), px, y_map(item["high"]))
            pdf.line(px - 3.2, y_map(item["low"]), px + 3.2, y_map(item["low"]))
            pdf.line(px - 3.2, y_map(item["high"]), px + 3.2, y_map(item["high"]))
            marker(pdf, px, py, shape, color, filled=item["holm"] < 0.05)
        pdf.setStrokeColor(color)
        pdf.setLineWidth(1.4)
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            pdf.line(x1, y1, x2, y2)

    # X labels.
    pdf.setFont("Helvetica", 8.5)
    pdf.setFillColor(black)
    for x, horizon in zip(xs, HORIZONS):
        label = f"{horizon:+d}"
        pdf.drawCentredString(x, bottom - 17, label)
    pdf.setFont("Helvetica-Bold", 8.5)
    pdf.drawCentredString((left + split) / 2, top + plot_h + 5, "Pre-trade placebo")
    pdf.drawCentredString((split + left + plot_w) / 2, top + plot_h + 5, "Post-trade response")

    # Axis titles.
    pdf.setFont("Helvetica", 9)
    pdf.drawCentredString(left + plot_w / 2, 18, "Minutes relative to initiating trade")
    pdf.saveState()
    pdf.translate(18, bottom + plot_h / 2)
    pdf.rotate(90)
    pdf.drawCentredString(0, 0, "Large-minus-ordinary coefficient (percentage points)")
    pdf.restoreState()

    # Compact legend.
    legend_y = height - 18
    legend_x = left + 88
    for spec, label, color, shape, _ in SERIES:
        marker(pdf, legend_x, legend_y, shape, color, filled=True)
        pdf.setFillColor(black)
        pdf.setFont("Helvetica", 8.5)
        pdf.drawString(legend_x + 8, legend_y - 3, label)
        legend_x += 148
    pdf.setFont("Helvetica", 7.5)
    pdf.drawRightString(width - right, legend_y - 3, "Filled marker: Holm-adjusted p < 0.05")

    pdf.showPage()
    pdf.save()
    print(OUTPUT)


if __name__ == "__main__":
    main()
