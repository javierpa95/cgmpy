"""MODD (Mean Of Daily Differences) metric."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from ._base import VariabilityBase

if TYPE_CHECKING:
    pass


class MODDMetrics(VariabilityBase):
    """Mixin providing MODD (Mean Of Daily Differences) calculation."""

    if TYPE_CHECKING:
        data: pd.DataFrame
        log: bool
        logger: object

    def MODD(self, days: int = 1) -> dict:
        """
        Calculates MODD (Mean Of Daily Differences) for a specific day interval.
        Optimized vectorized version.

        :param days: Number of days to calculate differences (1-6).
        :return: Dictionary with MODD value and related statistics.
        """
        if not 1 <= days <= 6:
            raise ValueError("The number of days must be between 1 and 6")

        df = self.data[["time", "glucose"]].copy()
        target_delta = pd.Timedelta(days=days)

        # Use time as index for alignment
        df_indexed = df.set_index("time")

        # Shift back to compare with 'days' ago
        try:
            # We use a frequency-based shift to align exactly by time of day
            df_shifted = df_indexed.shift(1, freq=target_delta)

            # Join to align values at the same time of day
            merged = df_indexed.join(df_shifted, lsuffix="_current", rsuffix="_past", how="inner")

            if merged.empty:
                return {
                    "value": None,
                    "n_observations": 0,
                    "std": None,
                    "correlation": None,
                }

            abs_diffs = (merged["glucose_current"] - merged["glucose_past"]).abs()

            modd_value = float(abs_diffs.mean())
            std_value = float(abs_diffs.std()) if len(abs_diffs) > 1 else 0.0
            correlation = float(merged["glucose_current"].corr(merged["glucose_past"]))

            return {
                "value": modd_value,
                "n_observations": len(abs_diffs),
                "std": std_value,
                "correlation": correlation,
            }
        except Exception as e:
            if getattr(self, "log", False):
                self.logger.error("Error in vectorized MODD: %s", e)
            return {
                "value": None,
                "n_observations": 0,
                "std": None,
                "correlation": None,
            }
