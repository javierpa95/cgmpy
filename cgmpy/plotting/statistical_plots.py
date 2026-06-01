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


class StatisticalPlotter:
    """
    Class to generate statistical glucose charts.

    This class should be used as a mixin with GlucoseData.
    """

    def histogram(self, bin_width: int = 10):
        """
        Generates and displays the glucose histogram with fixed intervals.

        Args:
            bin_width: Width of each interval in mg/dL (default 10)
        """
        # Calculate bin edges
        min_glucose = 0  # Or you could use self.data['glucose'].min()
        max_glucose = 500  # Or you could use self.data['glucose'].max()
        bins = range(int(min_glucose), int(max_glucose) + bin_width, bin_width)

        # Create figure
        plt.figure(figsize=(12, 8))

        # Create histogram
        plt.hist(self.data["glucose"], bins=bins, edgecolor="black", alpha=0.7)

        # Configure glycemia zones
        plt.axvspan(0, 70, color="#ffcccb", alpha=0.3, label="Hypoglycemia")
        plt.axvspan(70, 180, color="#90ee90", alpha=0.3, label="Target range")
        plt.axvspan(180, 400, color="#ffcccb", alpha=0.3, label="Hyperglycemia")

        # Configure chart
        plt.xlabel("Glucose Level (mg/dL)", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title(f"Glucose Histogram ({bin_width} mg/dL bins)", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_time_in_range(self, pregnancy: bool = False):
        """
        Generates a pie chart of time in range.

        Args:
            pregnancy: If True, uses pregnancy-specific ranges
        """
        if pregnancy:
            # Pregnancy ranges
            tir_pregnancy = self.TIR_pregnancy()  # 63-140 mg/dL
            tbr = self.TBR(63)  # < 63 mg/dL
            tar = self.TAR(140)  # > 140 mg/dL

            labels = [
                "TIR Pregnancy\n(63-140 mg/dL)",
                "TBR\n(< 63 mg/dL)",
                "TAR\n(> 140 mg/dL)",
            ]
            sizes = [tir_pregnancy, tbr, tar]
            colors = ["#90ee90", "#ffcccb", "#ffa500"]
            title = "Time in Range - Pregnancy"
        else:
            # Standard ranges
            tir = self.TIR()  # 70-180 mg/dL
            tbr70 = self.TBR70()  # 55-70 mg/dL
            tbr55 = self.TBR55()  # < 55 mg/dL
            tar180 = self.TAR180()  # 180-250 mg/dL
            tar250 = self.TAR250()  # > 250 mg/dL

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

        # Filter values greater than 0 for the chart
        non_zero_data = [
            (label, size, color)
            for label, size, color in zip(labels, sizes, colors, strict=False)
            if size > 0
        ]
        if non_zero_data:
            labels, sizes, colors = zip(*non_zero_data, strict=False)

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Pie chart
        wedges, texts, autotexts = ax1.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 10},
        )

        ax1.set_title(title, fontsize=14, fontweight="bold")

        # Horizontal bar chart
        y_pos = np.arange(len(labels))
        bars = ax2.barh(y_pos, sizes, color=colors, alpha=0.7)

        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(labels, fontsize=10)
        ax2.set_xlabel("Percentage (%)", fontsize=12)
        ax2.set_title("Detailed Distribution", fontsize=14, fontweight="bold")

        # Add values on bars
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

        plt.tight_layout()
        plt.show()

    def plot_distribution_comparison(self, target_ranges: list[tuple] | None = None):
        """
        Compares the current distribution with target ranges.

        Args:
            target_ranges: List of tuples (min, max, label, color) to compare
        """
        if target_ranges is None:
            target_ranges = [
                (70, 180, "Target Range", "#90ee90"),
                (0, 70, "Hypoglycemia", "#ffcccb"),
                (180, 400, "Hyperglycemia", "#ffa500"),
            ]

        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Histogram with density
        ax1.hist(
            self.data["glucose"],
            bins=50,
            density=True,
            alpha=0.7,
            color="skyblue",
            edgecolor="black",
        )

        # Add target zones
        for min_val, max_val, label, color in target_ranges:
            ax1.axvspan(min_val, max_val, alpha=0.3, color=color, label=label)

        ax1.set_xlabel("Glucose (mg/dL)")
        ax1.set_ylabel("Density")
        ax1.set_title("Glucose Distribution")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Box plot
        ax2.boxplot(
            self.data["glucose"],
            vert=True,
            patch_artist=True,
            boxprops={"facecolor": "lightblue"},
        )
        ax2.set_ylabel("Glucose (mg/dL)")
        ax2.set_title("Glucose Box Plot")
        ax2.grid(True, alpha=0.3)

        # 3. Q-Q plot (comparison with normal distribution)
        from scipy import stats

        stats.probplot(self.data["glucose"], dist="norm", plot=ax3)
        ax3.set_title("Q-Q Plot (Normality)")
        ax3.grid(True, alpha=0.3)

        # 4. Summary statistics
        ax4.axis("off")
        stats_text = self._generate_statistics_text()
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

        plt.tight_layout()
        plt.show()

    def plot_correlation_matrix(self, time_segments: list[str] | None = None):
        """
        Generates a correlation matrix between different time segments.

        Args:
            time_segments: List of time segments to analyze
        """
        if time_segments is None:
            time_segments = ["00:00-06:00", "06:00-12:00", "12:00-18:00", "18:00-24:00"]

        # Prepare data by segments
        data_copy = self.data.copy()
        data_copy["hour"] = data_copy["time"].dt.hour
        data_copy["date"] = data_copy["time"].dt.date

        # Create DataFrame with averages by segment and day
        segment_data = {}

        for segment in time_segments:
            start_hour, end_hour = segment.split("-")
            start_h = int(start_hour.split(":")[0])
            end_h = int(end_hour.split(":")[0])

            if end_h == 0:  # Special case for 24:00
                end_h = 24

            if start_h < end_h:
                mask = (data_copy["hour"] >= start_h) & (data_copy["hour"] < end_h)
            else:  # Segment crossing midnight
                mask = (data_copy["hour"] >= start_h) | (data_copy["hour"] < end_h)

            segment_glucose = data_copy[mask].groupby("date")["glucose"].mean()
            segment_data[segment] = segment_glucose

        # Create correlation DataFrame
        correlation_df = pd.DataFrame(segment_data)
        correlation_matrix = correlation_df.corr()

        # Create figure
        plt.figure(figsize=(10, 8))

        # Heatmap
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
            fmt=".3f",
            cbar_kws={"shrink": 0.8},
        )

        plt.title(
            "Correlation Matrix between Time Segments",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

    def _generate_statistics_text(self) -> str:
        """Generates text with summary statistics."""
        glucose_data = self.data["glucose"]

        stats_text = f"""DESCRIPTIVE STATISTICS

Mean:           {glucose_data.mean():.1f} mg/dL
Median:         {glucose_data.median():.1f} mg/dL
Std Dev:        {glucose_data.std():.1f} mg/dL
CV:             {(glucose_data.std() / glucose_data.mean() * 100):.1f}%

Percentiles:
P5:             {glucose_data.quantile(0.05):.1f} mg/dL
P25:            {glucose_data.quantile(0.25):.1f} mg/dL
P75:            {glucose_data.quantile(0.75):.1f} mg/dL
P95:            {glucose_data.quantile(0.95):.1f} mg/dL

Time in Range:
TIR (70-180):   {self.TIR():.1f}%
TBR (<70):      {self.TBR(70):.1f}%
TAR (>180):     {self.TAR(180):.1f}%

GMI:            {self.gmi():.1f}%
"""
        return stats_text
