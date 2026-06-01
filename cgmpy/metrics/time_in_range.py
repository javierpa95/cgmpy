"""
Time in Range metrics module for glucose data.

This module contains metrics related to time in different ranges:
- Time in Range (TIR)
- Time Above Range (TAR)
- Time Below Range (TBR)
- Specific time statistics
"""

from typing import TYPE_CHECKING, Any

import pandas as pd

from .targets import GlucoseTargets


class TimeInRangeMetrics:
    """
    Class for glucose time in range metrics.

    This class should be used as a mixin with GlucoseData.
    It expects a 'targets' attribute of type GlucoseTargets.
    """

    if TYPE_CHECKING:
        data: pd.DataFrame
        typical_interval: float
        targets: GlucoseTargets

    @property
    def current_targets(self) -> GlucoseTargets:
        """Returns the current glucose targets."""
        if not hasattr(self, "targets"):
            from .targets import get_targets

            self.targets = get_targets("diabetes")
        return self.targets

    def _calculate_data_completeness(self, interval_minutes: float | None = None) -> dict[str, Any]:
        """
        Calculates the percentage of available data for the current DataFrame.

        Args:
            interval_minutes: Expected interval between measurements in minutes.
                             If None, it is calculated automatically.

        Returns:
            dict: Information about data completeness.
        """
        # If interval is not specified, calculate it as the median of differences
        if interval_minutes is None:
            interval_minutes = self.typical_interval

        # Create a copy of the data and sort correctly
        data = self.data.sort_values("time").copy()

        # Analysis for the entire period
        total_time = (data["time"].max() - data["time"].min()).total_seconds() / 60

        if pd.isna(total_time) or interval_minutes <= 0:
            expected_data = 0
        else:
            expected_data = int(total_time / interval_minutes)

        real_data = len(data)

        return {
            "start": data["time"].min(),
            "end": data["time"].max(),
            "interval": interval_minutes,
            "expected_data": expected_data,
            "real_data": real_data,
            "percentage": (real_data / expected_data) * 100 if expected_data > 0 else 0,
        }

    def data_completeness(self, interval_minutes: float | None = None) -> int:
        """
        Returns the percentage of available data.

        Args:
            interval_minutes: Expected interval between measurements.

        Returns:
            int: Percentage of data completeness.
        """
        return int(self._calculate_data_completeness(interval_minutes)["percentage"])

    def calculate_time_in_range(self, low_threshold: float, high_threshold: float) -> float:
        """
        Calculates Time In Range (TIR) of glycemia.

        Args:
            low_threshold: Lower threshold of the range.
            high_threshold: Upper threshold of the range.

        Returns:
            float: Percentage of time in range.
        """
        in_range = self.data[
            (self.data["glucose"] >= low_threshold) & (self.data["glucose"] <= high_threshold)
        ]
        return (len(in_range) / len(self.data)) * 100

    def TAR(self, threshold: float) -> float:
        """
        Calculates Time Above Range (TAR).

        Args:
            threshold: Hyperglycemia threshold.

        Returns:
            float: Percentage of readings above the threshold.
        """
        return (len(self.data[self.data["glucose"] > threshold]) / len(self.data)) * 100

    def TBR(self, threshold: float) -> float:
        """
        Calculates Time Below Range (TBR).

        Args:
            threshold: Hypoglycemia threshold.

        Returns:
            float: Percentage of readings below the threshold.
        """
        return (len(self.data[self.data["glucose"] < threshold]) / len(self.data)) * 100

    # Generic methods that adapt to current targets
    def TBR_total(self) -> float:
        """
        Calculates total Time Below Range.
        In diabetes: < 70 mg/dL.
        In pregnancy: < 63 mg/dL.
        """
        return self.TBR(self.current_targets.target_low)

    def TBR_L1(self) -> float:
        """
        Calculates Level 1 Hypoglycemia (TBR).
        In standard diabetes: 54-70 mg/dL.
        In pregnancy: 55-63 mg/dL.
        """
        return self.calculate_time_in_range(
            self.current_targets.hypo_level2, self.current_targets.hypo_level1
        )

    def TBR_L2(self) -> float:
        """
        Calculates Level 2 Hypoglycemia (TBR).
        In standard diabetes: < 54 mg/dL.
        In pregnancy: < 55 mg/dL.
        """
        return self.TBR(self.current_targets.hypo_level2)

    def TAR_total(self) -> float:
        """
        Calculates total Time Above Range.
        In diabetes: > 180 mg/dL.
        In pregnancy: > 140 mg/dL.
        """
        return self.TAR(self.current_targets.target_high)

    def TAR_L1(self) -> float:
        """
        Calculates Level 1 Hyperglycemia (TAR).
        In standard diabetes: 181-250 mg/dL.
        In pregnancy: 141-250 mg/dL.
        """
        return self.calculate_time_in_range(
            self.current_targets.target_high + 1, self.current_targets.hyper_level2
        )

    def TAR_L2(self) -> float:
        """
        Calculates Level 2 Hyperglycemia (TAR).
        In standard diabetes: > 250 mg/dL.
        In pregnancy: > 250 mg/dL.
        """
        return self.TAR(self.current_targets.hyper_level2)

    # Specific Time in Range metrics (keeping for compatibility, but calling generics)
    def TAR250(self) -> float:
        """Calculates Very High Time Above Range (> 250 mg/dL)."""
        return self.TAR_L2()

    def TAR180(self) -> float:
        """Calculates Level 1 Time Above Range (Standard 181-250 or adapts)."""
        return self.TAR_L1()

    def TAR140(self) -> float:
        """Calculates High Time Above Range (> 140 mg/dL)."""
        return self.TAR(140)

    def TIR(self) -> float:
        """Calculates Time in Range (TIR) based on current targets."""
        return self.calculate_time_in_range(
            self.current_targets.target_low, self.current_targets.target_high
        )

    def TIR_tight(self) -> float:
        """Calculates tight time in range between 70 and 140 mg/dL."""
        return self.calculate_time_in_range(70, 140)

    def TIR_pregnancy(self) -> float:
        """Calculates time in range for pregnancy (63-140 mg/dL)."""
        return self.calculate_time_in_range(63, 140)

    def TBR70(self) -> float:
        """Calculates Level 1 Hypoglycemia (TBR 54-70 or 55-63)."""
        return self.TBR_L1()

    def TBR63(self) -> float:
        """Calculates time below 63 mg/dL."""
        return self.TBR(63)

    def TBR55(self) -> float:
        """Calculates time below 55 mg/dL."""
        return self.TBR(55)

    def TBR_very_low(self) -> float:
        """Calculates Level 2 Hypoglycemia (TBR)."""
        return self.TBR_L2()

    def time_statistics(self) -> dict[str, Any]:
        """Calculates glucose time statistics based on current targets."""
        t = self.current_targets
        return {
            "target_name": t.name,
            "%Data": self.data_completeness(),
            "TIR": self.TIR(),
            "TBR_L1": self.TBR_L1(),
            "TBR_L2": self.TBR_L2(),
            "TAR_L1": self.TAR_L1(),
            "TAR_L2": self.TAR_L2(),
            "TBR_total": self.TBR_total(),
            "TAR_total": self.TAR_total(),
        }

    def time_range_summary(self) -> dict[str, Any]:
        """
        Complete summary of all time in range metrics.

        Returns:
            dict: Complete summary of TIR, TAR, and TBR.
        """
        return {
            "data_completeness": self.data_completeness(),
            "current_targets": {
                "name": self.current_targets.name,
                "TIR": self.TIR(),
                "TBR_total": self.TBR_total(),
                "TBR_L1": self.TBR_L1(),
                "TBR_L2": self.TBR_L2(),
                "TAR_total": self.TAR_total(),
                "TAR_L1": self.TAR_L1(),
                "TAR_L2": self.TAR_L2(),
            },
            "standard_ranges": {
                "TIR": self.calculate_time_in_range(70, 180),
                "TAR180": self.calculate_time_in_range(181, 250),
                "TAR250": self.TAR(250),
                "TBR70": self.calculate_time_in_range(54, 70),
                "TBR54": self.TBR(54),
            },
            "pregnancy_ranges": {
                "TIR_pregnancy": self.calculate_time_in_range(63, 140),
                "TBR63_L1": self.calculate_time_in_range(55, 63),
                "TBR55": self.TBR(55),
                "TAR140_L1": self.calculate_time_in_range(141, 180),
                "TAR140_total": self.TAR(140),
            },
        }
