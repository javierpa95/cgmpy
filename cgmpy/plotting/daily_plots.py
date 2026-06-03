"""
Module for daily glucose data plots.

This module contains functions to generate charts related to daily patterns:
- Specific day plots
- Overlapping multiple days
- Boxplots by day of week
- Daily variation analysis
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ._utils import add_glucose_zones

logger = logging.getLogger(__name__)


def day_graph(data: pd.DataFrame, date: str | None = None):
    """Generates and displays the glucose chart for a specific day.

    Args:
        data: Glucose DataFrame with ``time`` and ``glucose`` columns.
        date: Optional date in 'YYYY-MM-DD' format.
              If not provided, the first day of the DataFrame is used.
    """
    if date is None:
        date = data["time"].dt.date.min()
    else:
        date = pd.to_datetime(date).date()

    day_data = data[data["time"].dt.date == date].copy()

    if day_data.empty:
        logger.info("No data for date %s", date)
        return

    day_data["hours"] = day_data["time"].dt.hour + day_data["time"].dt.minute / 60.0

    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.1)

    fig, ax = plt.subplots(figsize=(16, 9))

    add_glucose_zones(ax)

    ax.plot(
        day_data["hours"],
        day_data["glucose"],
        label="Glucose",
        color="#3366CC",
        linewidth=2,
        marker="o",
        markersize=4,
    )

    _add_reference_lines(ax)

    _configure_daily_plot(ax, f"Glucose Levels - {date}")

    plt.tight_layout()
    plt.show()


def plot_overlapping_days(data: pd.DataFrame):
    """Generates a chart with the glucose profiles of multiple overlapping days.

    Args:
        data: Glucose DataFrame with ``time`` and ``glucose`` columns.
    """
    data_copy = data.copy()
    data_copy["time_decimal"] = data_copy["time"].dt.hour + data_copy["time"].dt.minute / 60.0
    data_copy["date"] = data_copy["time"].dt.date

    plt.figure(figsize=(12, 8))

    mean_profile = (
        data_copy.groupby("time_decimal")["glucose"]
        .mean()
        .rolling(window=15, center=True, min_periods=1)
        .mean()
    )

    dates = data_copy["date"].unique()
    for d in dates:
        day_data = data_copy[data_copy["date"] == d]
        plt.plot(
            day_data["time_decimal"],
            day_data["glucose"],
            color="gray",
            alpha=0.2,
            linewidth=1,
        )

    plt.plot(
        mean_profile.index,
        mean_profile.values,
        color="black",
        linewidth=2,
        label="Mean profile",
    )

    _configure_overlapping_plot()

    plt.tight_layout()
    plt.show()


def plot_week_boxplots(data: pd.DataFrame):
    """Generates a boxplot chart to visualize the glucose distribution
    by day of the week.

    Args:
        data: Glucose DataFrame with ``time`` and ``glucose`` columns.
    """
    data_copy = data.copy()
    data_copy["weekday"] = data_copy["time"].dt.day_name()
    data_copy["date"] = data_copy["time"].dt.date

    day_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    unique_days = data_copy.groupby("weekday")["date"].nunique()

    labels = [f"{day}\n(n={unique_days.get(day, 0)} days)" for day in day_order]

    plt.figure(figsize=(12, 8))

    plt.axhspan(0, 70, color="#ffcccb", alpha=0.2, label="Hypoglycemia")
    plt.axhspan(70, 180, color="#90ee90", alpha=0.2, label="Target range")
    plt.axhspan(180, 400, color="#ffcccb", alpha=0.2, label="Hyperglycemia")

    sns.boxplot(
        x="weekday",
        y="glucose",
        data=data_copy,
        order=day_order,
        whis=1.5,
        medianprops={"color": "red", "linewidth": 1.5},
        flierprops={"marker": "o", "markerfacecolor": "gray", "markersize": 4},
    )

    plt.axhline(y=70, color="red", linestyle="--", linewidth=1)
    plt.axhline(y=180, color="red", linestyle="--", linewidth=1)

    plt.title("Glucose Distribution by Day of Week", fontsize=14, pad=20)
    plt.xlabel("Day of Week", fontsize=12)
    plt.ylabel("Glucose Level (mg/dL)", fontsize=12)

    plt.xticks(range(len(day_order)), labels, rotation=45, ha="right")
    plt.ylim(0, 400)

    plt.legend(title="Ranges", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.show()


def plot_daily_variations(data: pd.DataFrame):
    """Generates a chart that shows the average daily variations
    with confidence bands.

    Args:
        data: Glucose DataFrame with ``time`` and ``glucose`` columns.
    """
    data_copy = data.copy()
    data_copy["time_decimal"] = data_copy["time"].dt.hour + data_copy["time"].dt.minute / 60.0

    hourly_stats = (
        data_copy.groupby("time_decimal")["glucose"]
        .agg(
            [
                "mean",
                "std",
                "count",
                lambda x: np.percentile(x, 25),
                lambda x: np.percentile(x, 75),
            ]
        )
        .reset_index()
    )

    hourly_stats.columns = ["time_decimal", "mean", "std", "count", "p25", "p75"]

    window_size = 15
    for col in ["mean", "std", "p25", "p75"]:
        hourly_stats[col] = (
            hourly_stats[col].rolling(window=window_size, center=True, min_periods=1).mean()
        )

    fig, ax = plt.subplots(figsize=(14, 8))

    add_glucose_zones(ax)

    ax.plot(
        hourly_stats["time_decimal"],
        hourly_stats["mean"],
        color="blue",
        linewidth=2,
        label="Mean",
    )

    ax.fill_between(
        hourly_stats["time_decimal"],
        hourly_stats["mean"] - hourly_stats["std"],
        hourly_stats["mean"] + hourly_stats["std"],
        alpha=0.3,
        color="blue",
        label="\u00b1 1 SD",
    )

    ax.fill_between(
        hourly_stats["time_decimal"],
        hourly_stats["p25"],
        hourly_stats["p75"],
        alpha=0.2,
        color="green",
        label="Interquartile range",
    )

    ax.set_xlabel("Time of Day", fontsize=12)
    ax.set_ylabel("Glucose Level (mg/dL)", fontsize=12)
    ax.set_title("Average Daily Glucose Variations", fontsize=14)

    ax.set_xticks(range(0, 25, 3))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 3)])
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 400)

    ax.axhline(y=70, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(y=180, color="red", linestyle="--", linewidth=1, alpha=0.7)

    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()


def _add_reference_lines(ax):
    """Adds reference lines to the chart."""
    ax.axhline(y=70, color="#FF6666", linestyle="--", linewidth=1)
    ax.axhline(y=180, color="#FF6666", linestyle="--", linewidth=1)
    ax.text(24, 72, "70 mg/dL", va="bottom", ha="right", color="#FF6666")
    ax.text(24, 182, "180 mg/dL", va="bottom", ha="right", color="#FF6666")


def _configure_daily_plot(ax, title: str):
    """Configures common elements of the daily chart."""
    ax.set_xlabel("Time of Day", fontsize=12, fontweight="bold")
    ax.set_ylabel("Glucose Level (mg/dL)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold")

    ax.legend(loc="upper left", frameon=True, fancybox=True, shadow=True)
    ax.set_ylim(0, 400)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 3)])

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(True, linestyle=":", alpha=0.6)


def _configure_overlapping_plot():
    """Configures the overlapping days chart."""
    plt.xlabel("Time of Day", fontsize=12)
    plt.ylabel("Glucose Level (mg/dL)", fontsize=12)
    plt.title("Overlapping Glucose Profiles", fontsize=14)

    plt.xticks(range(0, 25, 3), [f"{h:02d}:00" for h in range(0, 25, 3)])

    plt.axhline(y=70, color="red", linestyle="--", alpha=0.5)
    plt.axhline(y=180, color="red", linestyle="--", alpha=0.5)

    plt.axhspan(0, 70, facecolor="#ffcccb", alpha=0.2)
    plt.axhspan(70, 180, facecolor="#90ee90", alpha=0.2)
    plt.axhspan(180, 400, facecolor="#ffcccb", alpha=0.2)

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 400)
