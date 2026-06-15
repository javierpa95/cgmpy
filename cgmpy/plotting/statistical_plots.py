"""
Module for statistical glucose data plots.

This module contains functions to generate statistical charts:
- Distribution histograms
- Time in range charts
- Correlation charts
- Statistical distribution analysis
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ._constants import (
    GLUCOSE_AXIS_MAX,
    GLUCOSE_AXIS_MIN,
    GLUCOSE_HIST_MAX,
    TARGET_HIGH,
    TARGET_LOW,
)
from ._utils import resolve_axes


def histogram(data: pd.DataFrame, bin_width: int = 10, ax: Axes | None = None) -> Figure:
    """Builds the glucose histogram with fixed intervals.

    Args:
        data: Glucose DataFrame with ``time`` and ``glucose`` columns.
        bin_width: Width of each interval in mg/dL (default 10).
        ax: Optional axis to draw into. If ``None`` a new figure is created.

    Returns:
        The matplotlib ``Figure`` containing the plot.
    """
    bins = range(GLUCOSE_AXIS_MIN, GLUCOSE_HIST_MAX + bin_width, bin_width)

    fig, ax, created = resolve_axes(ax, figsize=(12, 8))

    ax.hist(data["glucose"], bins=bins, edgecolor="black", alpha=0.7)

    ax.axvspan(GLUCOSE_AXIS_MIN, TARGET_LOW, color="#ffcccb", alpha=0.3, label="Hypoglycemia")
    ax.axvspan(TARGET_LOW, TARGET_HIGH, color="#90ee90", alpha=0.3, label="Target range")
    ax.axvspan(TARGET_HIGH, GLUCOSE_AXIS_MAX, color="#ffcccb", alpha=0.3, label="Hyperglycemia")

    ax.set_xlabel("Glucose Level (mg/dL)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(f"Glucose Histogram ({bin_width} mg/dL bins)", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if created:
        fig.tight_layout()
    return fig


def plot_time_in_range(
    data: pd.DataFrame,
    pregnancy: bool = False,
    tir_pregnancy: float = 0,
    tbr63: float = 0,
    tar140: float = 0,
    tir: float = 0,
    tbr70: float = 0,
    tbr55: float = 0,
    tar180: float = 0,
    tar250: float = 0,
) -> Figure:
    """Generates a pie chart of time in range.

    Args:
        data: Glucose DataFrame with ``time`` and ``glucose`` columns.
        pregnancy: If True, uses pregnancy-specific ranges.
        tir_pregnancy: TIR for pregnancy (63-140 mg/dL).
        tbr63: TBR below 63 mg/dL.
        tar140: TAR above 140 mg/dL.
        tir: Standard TIR (70-180 mg/dL).
        tbr70: TBR Level 1 (55-70 mg/dL).
        tbr55: TBR Level 2 (< 55 mg/dL).
        tar180: TAR Level 1 (180-250 mg/dL).
        tar250: TAR Level 2 (> 250 mg/dL).
    """
    if pregnancy:
        labels = [
            "TIR Pregnancy\n(63-140 mg/dL)",
            "TBR\n(< 63 mg/dL)",
            "TAR\n(> 140 mg/dL)",
        ]
        sizes = [tir_pregnancy, tbr63, tar140]
        colors = ["#90ee90", "#ffcccb", "#ffa500"]
        title = "Time in Range - Pregnancy"
    else:
        labels = [
            "TIR\n(70-180 mg/dL)",
            "TBR Level 1\n(55-70 mg/dL)",
            "TBR Level 2\n(< 55 mg/dL)",
            "TAR Level 1\n(180-250 mg/dL)",
            "TAR Level 2\n(> 250 mg/dL)",
        ]
        sizes = [tir, tbr70, tbr55, tar180, tar250]
        colors = ["#90ee90", "#ffeb9c", "#ffcccb", "#ffa500", "#ff6666"]
        title = "Time in Range - Standard"

    non_zero_data = [
        (label, size, color)
        for label, size, color in zip(labels, sizes, colors, strict=False)
        if size > 0
    ]
    if non_zero_data:
        labels, sizes, colors = map(list, zip(*non_zero_data, strict=False))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    wedges, texts, autotexts = ax1.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 10},
    )

    ax1.set_title(title, fontsize=14, fontweight="bold")

    y_pos = np.arange(len(labels))
    bars = ax2.barh(y_pos, sizes, color=colors, alpha=0.7)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=10)
    ax2.set_xlabel("Percentage (%)", fontsize=12)
    ax2.set_title("Detailed Distribution", fontsize=14, fontweight="bold")

    for _i, (bar, size) in enumerate(zip(bars, sizes, strict=False)):
        ax2.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{size:.1f}%",
            ha="left",
            va="center",
            fontsize=10,
        )

    ax2.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    return fig


def plot_distribution_comparison(
    data: pd.DataFrame,
    target_ranges: list[tuple] | None = None,
    tir_val: float = 0,
    tbr_val: float = 0,
    tar_val: float = 0,
    gmi_val: float = 0,
) -> Figure:
    """Compares the current distribution with target ranges.

    Args:
        data: Glucose DataFrame with ``time`` and ``glucose`` columns.
        target_ranges: List of tuples (min, max, label, color) to compare.
        tir_val: Time in Range value.
        tbr_val: Time Below Range value (< 70 mg/dL).
        tar_val: Time Above Range value (> 180 mg/dL).
        gmi_val: Glucose Management Index value.
    """
    if target_ranges is None:
        target_ranges = [
            (70, 180, "Target Range", "#90ee90"),
            (0, 70, "Hypoglycemia", "#ffcccb"),
            (180, 400, "Hyperglycemia", "#ffa500"),
        ]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    ax1.hist(
        data["glucose"],
        bins=50,
        density=True,
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
    )

    for min_val, max_val, label, color in target_ranges:
        ax1.axvspan(min_val, max_val, alpha=0.3, color=color, label=label)

    ax1.set_xlabel("Glucose (mg/dL)")
    ax1.set_ylabel("Density")
    ax1.set_title("Glucose Distribution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.boxplot(
        data["glucose"],
        vert=True,
        patch_artist=True,
        boxprops={"facecolor": "lightblue"},
    )
    ax2.set_ylabel("Glucose (mg/dL)")
    ax2.set_title("Glucose Box Plot")
    ax2.grid(True, alpha=0.3)

    from scipy import stats

    stats.probplot(data["glucose"], dist="norm", plot=ax3)
    ax3.set_title("Q-Q Plot (Normality)")
    ax3.grid(True, alpha=0.3)

    ax4.axis("off")
    stats_text = _generate_statistics_text(data["glucose"], tir_val, tbr_val, tar_val, gmi_val)
    ax4.text(
        0.1,
        0.9,
        stats_text,
        transform=ax4.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox={"boxstyle": "round", "facecolor": "lightgray", "alpha": 0.8},
    )

    fig.tight_layout()
    return fig


def plot_correlation_matrix(
    data: pd.DataFrame,
    time_segments: list[str] | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Builds a correlation matrix between different time segments.

    Args:
        data: Glucose DataFrame with ``time`` and ``glucose`` columns.
        time_segments: List of time segments to analyze.
        ax: Optional axis to draw into. If ``None`` a new figure is created.

    Returns:
        The matplotlib ``Figure`` containing the plot.
    """
    if time_segments is None:
        time_segments = ["00:00-06:00", "06:00-12:00", "12:00-18:00", "18:00-24:00"]

    data_copy = data.copy()
    data_copy["hour"] = data_copy["time"].dt.hour
    data_copy["date"] = data_copy["time"].dt.date

    segment_data = {}

    for segment in time_segments:
        start_hour, end_hour = segment.split("-")
        start_h = int(start_hour.split(":")[0])
        end_h = int(end_hour.split(":")[0])

        if end_h == 0:
            end_h = 24

        if start_h < end_h:
            mask = (data_copy["hour"] >= start_h) & (data_copy["hour"] < end_h)
        else:
            mask = (data_copy["hour"] >= start_h) | (data_copy["hour"] < end_h)

        segment_glucose = data_copy[mask].groupby("date")["glucose"].mean()
        segment_data[segment] = segment_glucose

    correlation_df = pd.DataFrame(segment_data)
    correlation_matrix = correlation_df.corr()

    fig, ax, created = resolve_axes(ax, figsize=(10, 8))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        center=0,
        square=True,
        fmt=".3f",
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )

    ax.set_title(
        "Correlation Matrix between Time Segments",
        fontsize=14,
        fontweight="bold",
    )
    if created:
        fig.tight_layout()
    return fig


def _generate_statistics_text(
    glucose: pd.Series,
    tir_val: float,
    tbr_val: float,
    tar_val: float,
    gmi_val: float,
) -> str:
    """Generates text with summary statistics.

    Args:
        glucose: Glucose values series.
        tir_val: Time in Range percentage.
        tbr_val: Time Below Range percentage.
        tar_val: Time Above Range percentage.
        gmi_val: Glucose Management Index value.

    Returns:
        Formatted statistics string.
    """
    stats_text = f"""DESCRIPTIVE STATISTICS

Mean:           {glucose.mean():.1f} mg/dL
Median:         {glucose.median():.1f} mg/dL
Std Dev:        {glucose.std():.1f} mg/dL
CV:             {(glucose.std() / glucose.mean() * 100):.1f}%

Percentiles:
P5:             {glucose.quantile(0.05):.1f} mg/dL
P25:            {glucose.quantile(0.25):.1f} mg/dL
P75:            {glucose.quantile(0.75):.1f} mg/dL
P95:            {glucose.quantile(0.95):.1f} mg/dL

Time in Range:
TIR (70-180):   {tir_val:.1f}%
TBR (<70):      {tbr_val:.1f}%
TAR (>180):     {tar_val:.1f}%

GMI:            {gmi_val:.1f}%
"""
    return stats_text
