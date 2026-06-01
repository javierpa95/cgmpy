"""
Module for ambulatory glucose profile (AGP) plots.

This module contains functions to generate ambulatory profiles:
- Standard AGP
- AGP by day of week
- Helper functions for percentile calculation
"""

import matplotlib.pyplot as plt
import numpy as np


class AGPPlotter:
    """
    Class to generate ambulatory glucose profile (AGP) plots.

    This class should be used as a mixin with GlucoseData.
    """

    def plot_agp(self, smoothing_window: int = 15):
        """
        Generates and displays the enhanced Ambulatory Glucose Profile (AGP).

        Args:
            smoothing_window: Smoothing window in minutes (default 15)
        """
        # Prepare data
        data_copy = self.data.copy()
        data_copy["time_decimal"] = (
            data_copy["time"].dt.hour + data_copy["time"].dt.minute / 60.0
        ).round(2)

        # Calculate percentiles
        percentiles = data_copy.groupby("time_decimal")["glucose"].agg(
            [
                lambda x: np.percentile(x, 5),
                lambda x: np.percentile(x, 25),
                lambda x: np.percentile(x, 50),
                lambda x: np.percentile(x, 75),
                lambda x: np.percentile(x, 95),
            ]
        )

        # Rename columns
        percentiles.columns = [0.05, 0.25, 0.5, 0.75, 0.95]

        # Apply smoothing
        for col in percentiles.columns:
            percentiles[col] = (
                percentiles[col].rolling(window=smoothing_window, center=True, min_periods=1).mean()
            )

        # Ensure data is sorted
        percentiles = percentiles.sort_index()

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))

        # Configure glycemia zones
        self._add_glucose_zones(ax)

        # Plot percentiles
        self._plot_percentiles(ax, percentiles)

        # Configure chart
        self._configure_agp_plot(ax, "Ambulatory Glucose Profile (AGP)")

        plt.tight_layout()
        plt.show()

    def generate_week_agp(self, smoothing_window: int = 15, combined: bool = True):
        """
        Generates and displays the Ambulatory Glucose Profile (AGP) by day of week.

        Args:
            smoothing_window: Smoothing window in minutes (default 15)
            combined: If True, displays all days in a single chart.
                     If False, displays a subplot for each day.
        """
        # Prepare data
        data_copy = self.data.copy()
        data_copy["time_decimal"] = (
            data_copy["time"].dt.hour + data_copy["time"].dt.minute / 60.0
        ).round(2)
        data_copy["weekday"] = data_copy["time"].dt.day_name()

        if combined:
            self._plot_combined_week_agp(data_copy, smoothing_window)
        else:
            self._plot_separate_week_agp(data_copy, smoothing_window)

    def _add_glucose_zones(self, ax):
        """Adds the glycemia zones to the chart."""
        ax.axhspan(0, 70, facecolor="#ffcccb", alpha=0.3, label="Hypoglycemia")
        ax.axhspan(70, 180, facecolor="#90ee90", alpha=0.3, label="Target range")
        ax.axhspan(180, 400, facecolor="#ffcccb", alpha=0.3, label="Hyperglycemia")

        # Horizontal lines at 70 and 180 mg/dL
        ax.axhline(y=70, color="red", linestyle="--", linewidth=1)
        ax.axhline(y=180, color="red", linestyle="--", linewidth=1)

    def _plot_percentiles(self, ax, percentiles):
        """Plots the percentile lines."""
        # Median line
        ax.plot(
            percentiles.index,
            percentiles[0.5],
            label="Median",
            color="blue",
            linewidth=2,
        )

        # Interquartile range
        ax.fill_between(
            percentiles.index,
            percentiles[0.25],
            percentiles[0.75],
            color="blue",
            alpha=0.3,
            label="Interquartile Range",
        )

        # Percentiles 5-95%
        ax.fill_between(
            percentiles.index,
            percentiles[0.05],
            percentiles[0.95],
            color="lightblue",
            alpha=0.2,
            label="Percentiles 5-95%",
        )

    def _configure_agp_plot(self, ax, title: str):
        """Configures common elements of the AGP chart."""
        # Labels and title
        ax.set_xlabel("Time of Day", fontsize=12)
        ax.set_ylabel("Glucose Level (mg/dL)", fontsize=12)
        ax.set_title(title, fontsize=16, fontweight="bold")

        # Legend
        ax.legend(title="Legend", loc="upper left", fontsize=10)

        # Grid
        ax.grid(True, linestyle=":", alpha=0.6)

        # X-axis configuration
        ax.set_xticks(range(0, 25, 3))
        ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 3)])

        # Y-axis limits
        ax.set_ylim(0, 400)

    def _plot_combined_week_agp(self, data_copy, smoothing_window: int):
        """Plots combined AGP for all weekdays."""
        # Order of days and colors
        days = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        colors = [
            "#FF6B6B",
            "#4ECDC4",
            "#45B7D1",
            "#96CEB4",
            "#FFEEAD",
            "#D4A5A5",
            "#9B59B6",
        ]

        # Create figure
        fig, ax = plt.subplots(figsize=(15, 8))

        # Configure glycemia zones
        self._add_glucose_zones(ax)

        for day, color in zip(days, colors, strict=False):
            # Filter data for the specific day
            day_data = data_copy[data_copy["weekday"] == day]

            if not day_data.empty:
                # Calculate percentiles
                percentiles = self._calculate_day_percentiles(day_data, smoothing_window)

                # Plot median line
                ax.plot(
                    percentiles.index,
                    percentiles[0.5],
                    label=f"{day} (n={len(day_data['time'].dt.date.unique())} days)",
                    color=color,
                    linewidth=2,
                )

                # IQR area with transparency
                ax.fill_between(
                    percentiles.index,
                    percentiles[0.25],
                    percentiles[0.75],
                    color=color,
                    alpha=0.1,
                )

        # Chart configuration
        ax.set_title(
            "Ambulatory Glucose Profile (AGP) by Day of Week",
            fontsize=14,
            pad=20,
        )
        ax.set_xlabel("Time of Day", fontsize=12)
        ax.set_ylabel("Glucose Level (mg/dL)", fontsize=12)
        ax.set_ylim(0, 400)
        ax.grid(True, linestyle=":", alpha=0.6)

        # X-axis configuration
        ax.set_xticks(range(0, 25, 3))
        ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 3)])

        # Legend
        ax.legend(
            title="Days of the week",
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fontsize=10,
        )

        plt.tight_layout()
        plt.show()

    def _plot_separate_week_agp(self, data_copy, smoothing_window: int):
        """Plots separate AGP for each weekday."""
        days = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]

        # Create subplots
        fig, axes = plt.subplots(7, 1, figsize=(15, 20), sharex=True)
        fig.suptitle(
            "Ambulatory Glucose Profile (AGP) by Day of Week",
            fontsize=16,
            fontweight="bold",
            y=0.92,
        )

        for ax, day in zip(axes, days, strict=False):
            # Filter data for the specific day
            day_data = data_copy[data_copy["weekday"] == day]

            if not day_data.empty:
                # Calculate full percentiles
                percentiles = self._calculate_full_day_percentiles(day_data, smoothing_window)

                # Configure glycemia zones
                self._add_glucose_zones(ax)

                # Plot percentiles
                self._plot_percentiles(ax, percentiles)

                # Configure subplot
                ax.set_title(
                    f"{day} (n={len(day_data['time'].dt.date.unique())} days)",
                    fontsize=12,
                    pad=10,
                )
                ax.set_ylabel("Glucose (mg/dL)", fontsize=10)
                ax.set_ylim(0, 400)
                ax.grid(True, linestyle=":", alpha=0.6)

        # Configure x-axis only on the last subplot
        axes[-1].set_xlabel("Time of Day", fontsize=12)
        axes[-1].set_xticks(range(0, 25, 3))
        axes[-1].set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 3)])

        plt.tight_layout()
        plt.show()

    def _calculate_day_percentiles(self, dia_data, smoothing_window: int):
        """Calculate percentiles for data from a specific day (25, 50, 75)."""
        percentiles = dia_data.groupby("time_decimal")["glucose"].agg(
            [
                lambda x: np.percentile(x, 25),
                lambda x: np.percentile(x, 50),
                lambda x: np.percentile(x, 75),
            ]
        )

        # Renombrar columnas
        percentiles.columns = [0.25, 0.5, 0.75]

        # Aplicar suavizado
        for col in percentiles.columns:
            percentiles[col] = (
                percentiles[col].rolling(window=smoothing_window, center=True, min_periods=1).mean()
            )

        return percentiles

    def _calculate_full_day_percentiles(self, dia_data, smoothing_window: int):
        """Calculate full percentiles for data from a specific day (5, 25, 50, 75, 95)."""
        percentiles = dia_data.groupby("time_decimal")["glucose"].agg(
            [
                lambda x: np.percentile(x, 5),
                lambda x: np.percentile(x, 25),
                lambda x: np.percentile(x, 50),
                lambda x: np.percentile(x, 75),
                lambda x: np.percentile(x, 95),
            ]
        )

        # Renombrar columnas
        percentiles.columns = [0.05, 0.25, 0.5, 0.75, 0.95]

        # Aplicar suavizado
        for col in percentiles.columns:
            percentiles[col] = (
                percentiles[col].rolling(window=smoothing_window, center=True, min_periods=1).mean()
            )

        return percentiles
