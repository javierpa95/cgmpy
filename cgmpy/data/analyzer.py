"""
Module for basic glucose data analysis.
"""

import logging
import time
from typing import Any

import numpy as np
import pandas as pd


class DataAnalyzer:
    """
    Class responsible for basic analysis of glucose data.
    """

    def __init__(self, logger: logging.Logger | None = None):
        """
        Initializes the DataAnalyzer.

        :param logger: Logger to record operations
        """
        self.logger = logger or logging.getLogger(__name__)

    def calculate_typical_interval(
        self, time_diffs: pd.Series, log_performance: bool = False
    ) -> float:
        """
        Calculates the typical interval between measurements in minutes.

        :param time_diffs: Series with the time differences
        :param log_performance: If True, records performance metrics
        :return: Typical interval in minutes
        """
        if log_performance:
            t_start = time.time()
            self.logger.debug("\n--- TYPICAL INTERVAL CALCULATION ANALYSIS ---")

        # Convert to NumPy array for faster operations
        time_diffs_seconds = time_diffs.dt.total_seconds().values
        # Filter valid values (greater than 0)
        valid_diffs = time_diffs_seconds[time_diffs_seconds > 0]

        if len(valid_diffs) > 0:
            # Use NumPy to calculate the median (faster)
            interval = np.median(valid_diffs) / 60
        else:
            # Default value if no valid differences
            interval = 5.0

        if log_performance:
            t_end = time.time()
            self.logger.debug(f"Optimized median calculation: {t_end - t_start:.3f}s")
            self.logger.debug(f"Total interval calculation time: {t_end - t_start:.3f}s")
            self.logger.debug("--- END OF ANALYSIS ---\n")

        return abs(interval)

    def get_basic_info(
        self,
        data: pd.DataFrame,
        time_diffs: pd.Series,
        typical_interval: float,
        include_disconnections: bool = False,
    ) -> dict[str, Any]:
        """
        Generates basic information about the glucose data.

        :param data: DataFrame with the glucose data
        :param time_diffs: Series with the time differences
        :param typical_interval: Typical interval between measurements
        :param include_disconnections: Whether to include disconnection details
        :return: Dictionary with basic information
        """
        # Basic information
        n_records = len(data)
        start_date = data["time"].min()
        end_date = data["time"].max()

        # Disconnection analysis
        disconnection_threshold = pd.Timedelta(minutes=typical_interval + 10)
        disconnections = time_diffs[time_diffs > disconnection_threshold]
        n_disconnections = len(disconnections)

        # Total disconnection time
        total_disconnection_time = disconnections.sum()
        disconnection_hours = total_disconnection_time.total_seconds() / 3600

        # Memory usage
        memory_bytes = data.memory_usage(deep=True).sum()
        memory_mb = memory_bytes / (1024 * 1024)

        # Expected theoretical data
        total_time = (data["time"].max() - data["time"].min()).total_seconds() / 60

        # Avoid errors if total_time or typical_interval are invalid
        if pd.isna(total_time) or typical_interval <= 0:
            expected_data = 0
        else:
            expected_data = int(total_time / typical_interval)

        completeness = (n_records / expected_data * 100) if expected_data > 0 else 0.0

        # Create summary dictionary
        summary = {
            "n_records": n_records,
            "start_date": start_date,
            "end_date": end_date,
            "typical_interval": typical_interval,
            "expected_data": expected_data,
            "completeness": completeness,
            "n_disconnections": (
                f"{n_disconnections} disconnections (For more info, "
                "use info(include_disconnections=True))"
            ),
            "total_disconnection_time": disconnection_hours,
            "memory_usage_mb": memory_mb,
        }

        if include_disconnections:
            summary["disconnection_list"] = self._get_disconnection_details(data, disconnections)

        return summary

    def _get_disconnection_details(self, data: pd.DataFrame, disconnections: pd.Series) -> list:
        """
        Gets details of disconnections.

        :param data: DataFrame with data
        :param disconnections: Series with disconnections
        :return: List with disconnection details
        """
        disconnection_list = []

        if len(disconnections) > 0:
            for idx, index in enumerate(disconnections.index, 1):
                try:
                    current_pos = data.index.get_loc(index)
                    if current_pos > 0:
                        disconnection_end = data.iloc[current_pos]["time"]
                        disconnection_start = data.iloc[current_pos - 1]["time"]
                        duration_minutes = (
                            disconnection_end - disconnection_start
                        ).total_seconds() / 60
                        hours = int(duration_minutes // 60)
                        minutes = int(duration_minutes % 60)
                        disconnection_list.append(
                            {
                                "start": disconnection_start.strftime("%d/%m/%Y %H:%M"),
                                "end": disconnection_end.strftime("%d/%m/%Y %H:%M"),
                                "duration": f"{hours:02d} hours and {minutes:02d} minutes",
                            }
                        )
                except Exception as e:
                    self.logger.warning(f"Error processing disconnection {idx}: {e}")

        return disconnection_list

    def get_summary_string(self, info: dict[str, Any]) -> str:
        """
        Generates a string representation of basic information.

        :param info: Dictionary with basic information
        :return: String with summary
        """
        return (
            f"File contains {info['n_records']} records between {info['start_date']} and {info['end_date']}.\n"
            f"Typical interval between measurements: {info['typical_interval']:.1f} minutes.\n"
            f"Expected theoretical data: {info['expected_data']}\n"
            f"Data availability percentage: {info['completeness']:.1f}%\n"
            f"Detected {info['n_disconnections']}\n"
            f"Total disconnection time: {info['total_disconnection_time']:.1f} hours.\n"
            f"DataFrame memory usage: {info['memory_usage_mb']:.2f} MB"
        )

    def get_data_quality_metrics(
        self, data: pd.DataFrame, time_diffs: pd.Series, typical_interval: float
    ) -> dict[str, Any]:
        """
        Calculates data quality metrics.

        :param data: DataFrame with data
        :param time_diffs: Series with time differences
        :param typical_interval: Typical interval between measurements
        :return: Dictionary with quality metrics
        """
        # Calculate gaps in the data
        gap_threshold = pd.Timedelta(minutes=typical_interval * 2)
        gaps = time_diffs[time_diffs > gap_threshold]

        # Calculate interval statistics
        valid_intervals = time_diffs[time_diffs > pd.Timedelta(0)].dt.total_seconds() / 60

        return {
            "total_gaps": len(gaps),
            "max_gap_hours": gaps.max().total_seconds() / 3600 if len(gaps) > 0 else 0,
            "mean_interval": valid_intervals.mean() if len(valid_intervals) > 0 else 0,
            "std_interval": valid_intervals.std() if len(valid_intervals) > 0 else 0,
            "min_glucose": data["glucose"].min(),
            "max_glucose": data["glucose"].max(),
            "mean_glucose": data["glucose"].mean(),
            "std_glucose": data["glucose"].std(),
        }
