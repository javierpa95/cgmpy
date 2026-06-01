"""
Main glucose analysis module.

This module combines all analysis functionality:
- Data handling (ModularGlucoseData)
- Metrics and statistics (metrics modules)
- Visualization (plotting modules)
"""

import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..data.core import ModularGlucoseData
from ..metrics.basic import BasicMetrics
from ..metrics.time_in_range import TimeInRangeMetrics
from ..metrics.variability import VariabilityMetrics
from ..plotting.agp import AGPPlotter
from ..plotting.daily_plots import DailyPlotter
from ..plotting.statistical_plots import StatisticalPlotter


class GlucoseAnalysis(
    ModularGlucoseData,
    BasicMetrics,
    TimeInRangeMetrics,
    VariabilityMetrics,
    AGPPlotter,
    DailyPlotter,
    StatisticalPlotter,
):
    """
    Class combining all glucose analysis functionality.

    Inherits from:
    - ModularGlucoseData: Data handling
    - BasicMetrics: Descriptive statistics (Mean, GMI, CV, ...)
    - TimeInRangeMetrics: Time in range metrics (TIR, TAR, TBR)
    - VariabilityMetrics: Variability metrics (MAGE, MODD, CONGA, etc.)
    - AGPPlotter: Ambulatory profile plots
    - DailyPlotter: Daily plots
    - StatisticalPlotter: Statistical plots
    """

    def __init__(
        self,
        data_source: str | pd.DataFrame,
        date_col: str = "time",
        glucose_col: str = "glucose",
        delimiter: str | None = None,
        header: int = 0,
        start_date: str | datetime.datetime | None = None,
        end_date: str | datetime.datetime | None = None,
        log: bool = False,
    ):
        """
        Initializes the full glucose analysis.

        Args:
            data_source: Data source (file or DataFrame)
            date_col: Name of the date column
            glucose_col: Name of the glucose column
            delimiter: File delimiter
            header: Header row number
            start_date: Start date for filtering data
            end_date: End date for filtering data
            log: Whether to enable detailed logs
        """
        # Initialize ModularGlucoseData
        super().__init__(
            data_source=data_source,
            date_col=date_col,
            glucose_col=glucose_col,
            delimiter=delimiter,
            header=header,
            start_date=start_date,
            end_date=end_date,
            log=log,
        )

    def get_comprehensive_report(self) -> dict[str, Any]:
        """
        Generates a comprehensive report with all available metrics.

        Returns:
            dict: Complete report with all metrics
        """
        report = {
            "basic_info": self.info(),
            "basic_metrics": self.calculate_all_metrics(),
            "time_statistics": self.time_statistics(),
            "variability_metrics": self.calculate_variability_metrics(),
            "data_quality": self.get_data_quality_metrics(),
        }

        return report

    def get_summary_string(self) -> str:
        """
        Generates a text summary of the analysis.

        Returns:
            str: Analysis summary
        """
        summary = []
        summary.append("=== COMPREHENSIVE GLUCOSE ANALYSIS ===")
        summary.append("")

        # Basic information
        info = self.info()
        summary.append("DATA:")
        summary.append(f"  - Records: {info['n_records']:,}")
        summary.append(
            f"  - Period: {info['start_date'].strftime('%d/%m/%Y')} - {info['end_date'].strftime('%d/%m/%Y')}"
        )
        summary.append(f"  - Availability: {info['completeness']:.1f}%")
        summary.append("")

        # Basic metrics
        basic = self.calculate_all_metrics()
        summary.append("BASIC METRICS:")
        summary.append(f"  - GMI: {basic['GMI']:.1f}%")
        summary.append(f"  - Mean: {basic['Mean']:.1f} mg/dL")
        summary.append(f"  - Median: {basic['Median']:.1f} mg/dL")
        summary.append(f"  - Std Dev: {basic['Std']:.1f} mg/dL")
        summary.append(f"  - CV: {basic['CV']:.1f}%")
        summary.append("")

        # Time in range (call individual methods; time_statistics() returns
        # only a subset of these keys).
        summary.append("TIME IN RANGE:")
        summary.append(f"  - TIR (70-180): {self.TIR():.1f}%")
        summary.append(f"  - TIR tight (70-140): {self.TIR_tight():.1f}%")
        summary.append(f"  - TBR70 (54-70): {self.TBR70():.1f}%")
        summary.append(f"  - TBR55 (<55): {self.TBR55():.1f}%")
        summary.append(f"  - TAR140 (>140): {self.TAR140():.1f}%")
        summary.append(f"  - TAR180 (181-250): {self.TAR180():.1f}%")
        summary.append(f"  - TAR250 (>250): {self.TAR250():.1f}%")
        summary.append("")

        # Variability
        variability = self.calculate_variability_metrics()
        summary.append("VARIABILITY:")
        summary.append(f"  - MAGE: {variability.get('MAGE', 'N/A')}")
        summary.append(f"  - MODD: {variability.get('MODD', 'N/A')}")
        summary.append(f"  - CONGA: {variability.get('CONGA', 'N/A')}")
        summary.append(f"  - SD total: {variability.get('SD_total', 'N/A')}")
        summary.append(f"  - SD within-day: {variability.get('SD_within_day', 'N/A')}")
        summary.append(f"  - SD between-day: {variability.get('SD_between_day', 'N/A')}")

        return "\n".join(summary)

    def plot_comprehensive_dashboard(self, figsize: tuple = (20, 12)):
        """
        Generates a comprehensive dashboard with multiple plots.

        Args:
            figsize: Figure size
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle("Comprehensive Glucose Analysis Dashboard", fontsize=16, fontweight="bold")

        # Plot 1: AGP
        self.plot_agp()
        axes[0, 0].set_title("Ambulatory Profile (AGP)")

        # Plot 2: Distribution
        self.histogram()
        axes[0, 1].set_title("Glucose Distribution")

        # Plot 3: Time in range
        self.plot_time_in_range()
        axes[0, 2].set_title("Time in Range")

        # Plot 4: Variability
        self.plot_variability_dashboard()
        axes[1, 0].set_title("Variability Analysis")

        # Plot 5: Overlapping days
        self.plot_overlapping_days()
        axes[1, 1].set_title("Overlapping Days")

        # Plot 6: Weekly boxplots
        self.plot_week_boxplots()
        axes[1, 2].set_title("Weekly Boxplots")

        plt.tight_layout()
        plt.show()

    def export_report(self, file_path: str, format: str = "json"):
        """
        Exports the complete report to a file.

        Args:
            file_path: Output file path
            format: Export format ('json', 'csv', 'excel')
        """
        report = self.get_comprehensive_report()

        if format.lower() == "json":
            import json

            with Path(file_path).open("w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str, ensure_ascii=False)

        elif format.lower() == "csv":
            # Convert report to flat DataFrame
            flat_report = self._flatten_report(report)
            flat_report.to_csv(file_path, index=False)

        elif format.lower() == "excel":
            # Create Excel with multiple sheets
            with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
                # Sheet 1: Basic info
                pd.DataFrame([report["basic_info"]]).to_excel(
                    writer, sheet_name="Basic_Info", index=False
                )

                # Sheet 2: Basic metrics
                pd.DataFrame([report["basic_metrics"]]).to_excel(
                    writer, sheet_name="Basic_Metrics", index=False
                )

                # Sheet 3: Time in range
                pd.DataFrame([report["time_statistics"]]).to_excel(
                    writer, sheet_name="Time_Range", index=False
                )

        else:
            raise ValueError(f"Unsupported format: {format}")

    def _flatten_report(self, report: dict[str, Any]) -> pd.DataFrame:
        """
        Converts the nested report to a flat DataFrame.

        Args:
            report: Nested report

        Returns:
            pd.DataFrame: Flat report
        """
        flat_data = {}

        for section, data in report.items():
            if isinstance(data, dict):
                for key, value in data.items():
                    flat_data[f"{section}_{key}"] = value
            else:
                flat_data[section] = data

        return pd.DataFrame([flat_data])
