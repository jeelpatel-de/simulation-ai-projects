#!/usr/bin/env python3
"""
ABAQUS Simulation Post-Processing Script
=========================================
Reads FEA results from CSV, analyzes von Mises stress distribution,
and generates a PDF report with plots and summary statistics.

Requirements: pip install pandas matplotlib numpy reportlab
Usage:        python abaqus_postprocessor.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
)
from io import BytesIO
import os
import sys
from datetime import datetime


# ─── Configuration ──────────────────────────────────────────────────
CSV_FILE = "test_results.csv"
PDF_OUTPUT = "simulation_report.pdf"
TOP_N = 10  # Number of highest-stress nodes to display


def load_data(filepath):
    """Load and validate the ABAQUS CSV export."""
    if not os.path.isfile(filepath):
        print(f"ERROR: File '{filepath}' not found.")
        print("Place test_results.csv in the same directory as this script.")
        sys.exit(1)

    df = pd.read_csv(filepath)

    required_cols = {"Node", "S11", "S22", "S33", "Mises", "U1", "U2", "U3"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        sys.exit(1)

    df["Node"] = df["Node"].astype(int)
    print(f"Loaded {len(df)} nodes from '{filepath}'.")
    return df


def analyze_stress(df):
    """Compute summary statistics for von Mises stress."""
    mises = df["Mises"]
    stats = {
        "Minimum [MPa]": mises.min(),
        "Maximum [MPa]": mises.max(),
        "Mean [MPa]": mises.mean(),
        "Median [MPa]": mises.median(),
        "Std. Deviation [MPa]": mises.std(),
    }

    max_idx = mises.idxmax()
    max_node = int(df.loc[max_idx, "Node"])
    max_val = mises.loc[max_idx]

    print("\n" + "=" * 55)
    print("  VON MISES STRESS SUMMARY")
    print("=" * 55)
    for key, val in stats.items():
        print(f"  {key:<28s} {val:>10.2f}")
    print("-" * 55)
    print(f"  Critical node:               Node {max_node}")
    print(f"  Peak von Mises stress:       {max_val:.2f} MPa")
    print("=" * 55 + "\n")

    return stats, max_node, max_val


def plot_top_stress_bar(df, top_n=10):
    """Bar chart of the top-N highest von Mises stress nodes."""
    top = df.nlargest(top_n, "Mises").sort_values("Mises", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_colors = plt.cm.Reds(np.linspace(0.35, 0.95, len(top)))
    bars = ax.barh(
        top["Node"].astype(str),
        top["Mises"],
        color=bar_colors,
        edgecolor="darkred",
        linewidth=0.6,
    )

    # Annotate values on bars
    for bar, val in zip(bars, top["Mises"]):
        ax.text(
            bar.get_width() + 2,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlabel("Von Mises Stress [MPa]", fontsize=11)
    ax.set_ylabel("Node ID", fontsize=11)
    ax.set_title(f"Top {top_n} Nodes by Von Mises Stress", fontsize=13, fontweight="bold")
    ax.set_xlim(right=top["Mises"].max() * 1.15)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_stress_distribution(df):
    """Histogram of the von Mises stress distribution across all nodes."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(
        df["Mises"],
        bins=12,
        color="steelblue",
        edgecolor="white",
        linewidth=0.8,
        alpha=0.85,
    )
    ax.axvline(
        df["Mises"].mean(),
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f'Mean = {df["Mises"].mean():.1f} MPa',
    )
    ax.set_xlabel("Von Mises Stress [MPa]", fontsize=11)
    ax.set_ylabel("Number of Nodes", fontsize=11)
    ax.set_title("Von Mises Stress Distribution", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_displacement_magnitude(df):
    """Scatter plot of displacement magnitude vs. von Mises stress."""
    df = df.copy()
    df["U_mag"] = np.sqrt(df["U1"] ** 2 + df["U2"] ** 2 + df["U3"] ** 2)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sc = ax.scatter(
        df["Mises"],
        df["U_mag"] * 1000,  # convert to mm for readability
        c=df["Mises"],
        cmap="hot_r",
        edgecolors="grey",
        linewidths=0.5,
        s=70,
        alpha=0.85,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Von Mises [MPa]", fontsize=10)

    ax.set_xlabel("Von Mises Stress [MPa]", fontsize=11)
    ax.set_ylabel("Displacement Magnitude [mm]", fontsize=11)
    ax.set_title("Displacement vs. Stress Correlation", fontsize=13, fontweight="bold")
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def fig_to_image(fig, dpi=180):
    """Convert a matplotlib figure to a ReportLab Image flowable."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = Image(buf)
    # Scale to fit A4 width with margins
    max_w = 170 * mm
    aspect = img.imageWidth / img.imageHeight
    img.drawWidth = max_w
    img.drawHeight = max_w / aspect
    return img


def build_pdf(df, stats, max_node, max_val):
    """Assemble the full PDF report using ReportLab."""
    doc = SimpleDocTemplate(
        PDF_OUTPUT,
        pagesize=A4,
        topMargin=25 * mm,
        bottomMargin=20 * mm,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="SmallBody",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
    ))

    story = []

    # ── Title page ──────────────────────────────────────────────────
    story.append(Spacer(1, 40 * mm))
    story.append(Paragraph(
        "ABAQUS FEA Simulation Report", styles["Title"]
    ))
    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d  %H:%M')}",
        styles["Normal"],
    ))
    story.append(Paragraph(
        f"Data source: {CSV_FILE}  |  Nodes analysed: {len(df)}",
        styles["Normal"],
    ))
    story.append(Spacer(1, 12 * mm))
    story.append(Paragraph(
        f"<b>Critical node: {max_node}</b> &mdash; "
        f"Peak von Mises stress: <b>{max_val:.2f} MPa</b>",
        styles["Normal"],
    ))
    story.append(PageBreak())

    # ── Summary statistics table ────────────────────────────────────
    story.append(Paragraph("1. Stress Summary Statistics", styles["Heading2"]))
    story.append(Spacer(1, 4 * mm))

    table_data = [["Statistic", "Value [MPa]"]]
    for key, val in stats.items():
        table_data.append([key, f"{val:.2f}"])

    tbl = Table(table_data, colWidths=[100 * mm, 60 * mm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME",  (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",  (0, 0), (-1, 0), 10),
        ("FONTSIZE",  (0, 1), (-1, -1), 9),
        ("ALIGN",     (1, 0), (1, -1), "RIGHT"),
        ("GRID",      (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 10 * mm))

    # ── Top-N bar chart ─────────────────────────────────────────────
    story.append(Paragraph(
        f"2. Top {TOP_N} Nodes by Von Mises Stress", styles["Heading2"]
    ))
    story.append(Spacer(1, 4 * mm))
    fig_bar = plot_top_stress_bar(df, TOP_N)
    story.append(fig_to_image(fig_bar))
    story.append(PageBreak())

    # ── Histogram ───────────────────────────────────────────────────
    story.append(Paragraph("3. Stress Distribution", styles["Heading2"]))
    story.append(Spacer(1, 4 * mm))
    fig_hist = plot_stress_distribution(df)
    story.append(fig_to_image(fig_hist))
    story.append(Spacer(1, 10 * mm))

    # ── Displacement scatter ────────────────────────────────────────
    story.append(Paragraph(
        "4. Displacement vs. Stress Correlation", styles["Heading2"]
    ))
    story.append(Spacer(1, 4 * mm))
    fig_scatter = plot_displacement_magnitude(df)
    story.append(fig_to_image(fig_scatter))
    story.append(PageBreak())

    # ── Full data table ─────────────────────────────────────────────
    story.append(Paragraph("5. Complete Node Data", styles["Heading2"]))
    story.append(Spacer(1, 4 * mm))

    header = list(df.columns)
    data_rows = [header]
    for _, row in df.iterrows():
        data_rows.append([
            str(int(row["Node"])),
            f'{row["S11"]:.1f}',
            f'{row["S22"]:.1f}',
            f'{row["S33"]:.1f}',
            f'{row["Mises"]:.1f}',
            f'{row["U1"]:.4f}',
            f'{row["U2"]:.4f}',
            f'{row["U3"]:.4f}',
        ])

    col_w = [16 * mm] + [22 * mm] * 7
    data_tbl = Table(data_rows, colWidths=col_w, repeatRows=1)
    data_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 8),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("GRID",       (0, 0), (-1, -1), 0.4, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(data_tbl)

    # ── Build ───────────────────────────────────────────────────────
    doc.build(story)
    print(f"PDF report saved to: {PDF_OUTPUT}")


# ─── Main ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data(CSV_FILE)
    stats, max_node, max_val = analyze_stress(df)
    build_pdf(df, stats, max_node, max_val)
