"""
Main glucose analysis module.

This module combines all analysis functionality:
- Data handling (GlucoseData)
- Metrics and statistics (metrics modules)
- Visualization (plotting modules)
"""

import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from ..data.core import GlucoseData
from ..metrics.targets import GlucoseTargets


class GlucoseAnalysis:
    """Full CGM analysis by composition.

    Wraps a :class:`GlucoseData` instance and provides access to all metrics,
    plots, and reports via delegation to pure functions.

    Args:
        data: A :class:`GlucoseData` instance, or a file path / DataFrame
            (backward-compatible shorthand that wraps in ``GlucoseData``).
    """

    def __init__(
        self,
        data: GlucoseData | str | pd.DataFrame,
        date_col: str = "time",
        glucose_col: str = "glucose",
        delimiter: str | None = None,
        header: int = 0,
        start_date: str | datetime.datetime | None = None,
        end_date: str | datetime.datetime | None = None,
        log: bool = False,
        target_type: str = "diabetes",
    ):
        if isinstance(data, GlucoseData):
            self._data = data
        else:
            self._data = GlucoseData(
                data_source=data,
                date_col=date_col,
                glucose_col=glucose_col,
                delimiter=delimiter,
                header=header,
                start_date=start_date,
                end_date=end_date,
                log=log,
                target_type=target_type,
            )

        self._cache: dict[str, Any] = {}

    # ── Cache ────────────────────────────────────────────────────────────

    def _cached(self, key: str, func: Callable) -> Any:
        if key not in self._cache:
            self._cache[key] = func()
        return self._cache[key]

    def clear_cache(self) -> None:
        """Clear all cached computed values."""
        self._cache = {}

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def glucose(self) -> pd.Series:
        return self._data.glucose

    @property
    def timestamps(self) -> pd.Series:
        return self._data.timestamps

    @property
    def targets(self) -> GlucoseTargets:
        return self._data.targets

    @property
    def data(self) -> pd.DataFrame:
        """Raw glucose DataFrame (passthrough to :attr:`GlucoseData.data`)."""
        return self._data.data

    # ── Data passthrough ─────────────────────────────────────────────────

    def info(self, include_disconnections: bool = False) -> dict[str, Any]:
        return self._data.info(include_disconnections)

    def get_data_quality_metrics(self) -> dict[str, Any]:
        return self._data.get_data_quality_metrics()

    def get_logs(self) -> dict[str, Any]:
        return self._data.get_logs()

    def to_parquet(self, file_path: str, compression: str = "snappy", sort: bool = True):
        self._data.to_parquet(file_path, compression, sort)

    def to_csv(self, file_path: str, separator: str = ",", include_index: bool = False):
        self._data.to_csv(file_path, separator, include_index)

    def to_excel(self, file_path: str, sheet_name: str = "glucose_data"):
        self._data.to_excel(file_path, sheet_name)

    def get_raw_data(self) -> pd.DataFrame:
        return self._data.get_raw_data()

    def get_glucose_values(self) -> pd.Series:
        return self._data.get_glucose_values()

    def get_timestamps(self) -> pd.Series:
        return self._data.get_timestamps()

    def get_time_differences(self) -> pd.Series:
        return self._data.get_time_differences()

    def get_typical_interval(self) -> float:
        return self._data.get_typical_interval()

    def filter_by_date_range(
        self,
        start: str | datetime.datetime,
        end: str | datetime.datetime,
    ) -> "GlucoseAnalysis":
        return GlucoseAnalysis(self._data.filter_by_date_range(start, end))

    def filter_by_glucose_range(
        self,
        min_g: float,
        max_g: float,
    ) -> "GlucoseAnalysis":
        return GlucoseAnalysis(self._data.filter_by_glucose_range(min_g, max_g))

    # ── Basic metrics ────────────────────────────────────────────────────

    def mean(self) -> float:
        from ..metrics.basic import mean

        return self._cached("_mean", lambda: mean(self._data.glucose))

    def median(self) -> float:
        from ..metrics.basic import median

        return self._cached("_median", lambda: median(self._data.glucose))

    def sd(self) -> float:
        from ..metrics.basic import sd

        return self._cached("_sd", lambda: sd(self._data.glucose))

    def cv(self) -> float:
        from ..metrics.basic import cv

        return cv(self._data.glucose)

    def gmi(self) -> float:
        from ..metrics.basic import gmi

        return gmi(self.mean())

    def percentile(self, p: float) -> float:
        from ..metrics.basic import percentile

        return percentile(self._data.glucose, p)

    def distribution_analysis(self) -> dict[str, Any]:
        return {
            "mean": self.mean(),
            "median": self.median(),
            "std": self.sd(),
            "cv": self.cv(),
            "skewness": self._data.glucose.skew(),
            "kurtosis": self._data.glucose.kurtosis(),
            "percentiles": {
                "p5": self.percentile(5),
                "p25": self.percentile(25),
                "p50": self.percentile(50),
                "p75": self.percentile(75),
                "p95": self.percentile(95),
                "IQR": self.percentile(75) - self.percentile(25),
            },
        }

    def calculate_all_metrics(self) -> dict[str, float]:
        return {
            "GMI": self.gmi(),
            "Mean": self.mean(),
            "Median": self.median(),
            "Std": self.sd(),
            "CV": self.cv(),
            "P5": self.percentile(5),
            "P25": self.percentile(25),
            "P75": self.percentile(75),
            "P95": self.percentile(95),
            "Skewness": self._data.glucose.skew(),
            "Kurtosis": self._data.glucose.kurtosis(),
        }

    def __str__(self) -> str:
        stats = self.calculate_all_metrics()
        return (
            "BASIC GLUCOSE METRICS\n"
            "\n"
            "CENTRAL STATISTICS:\n"
            f"  - GMI (HbA1c est.):       {stats['GMI']:.1f}%\n"
            f"  - Mean:                  {stats['Mean']:.1f} mg/dL\n"
            f"  - Median:                {stats['Median']:.1f} mg/dL\n"
            f"  - Std Dev:               {stats['Std']:.1f} mg/dL\n"
            f"  - CV:                    {stats['CV']:.1f}%\n"
            "\n"
            "PERCENTILES:\n"
            f"  - P5:                    {stats['P5']:.1f} mg/dL\n"
            f"  - P25:                   {stats['P25']:.1f} mg/dL\n"
            f"  - P75:                   {stats['P75']:.1f} mg/dL\n"
            f"  - P95:                   {stats['P95']:.1f} mg/dL\n"
            "\n"
            "DISTRIBUTION SHAPE:\n"
            f"  - Skewness:              {stats['Skewness']:.2f}\n"
            f"  - Kurtosis:              {stats['Kurtosis']:.2f}\n"
        )

    # ── Time in range ───────────────────────────────────────────────────

    def _tir_pure(self, low: float, high: float) -> float:
        from ..metrics.time_in_range import tir

        return tir(self._data.glucose, low, high)

    def _tar_pure(self, threshold: float) -> float:
        from ..metrics.time_in_range import tar

        return tar(self._data.glucose, threshold)

    def _tbr_pure(self, threshold: float) -> float:
        from ..metrics.time_in_range import tbr

        return tbr(self._data.glucose, threshold)

    def _data_completeness_pure(self, interval_minutes: float = 5) -> float:
        from ..metrics.time_in_range import data_completeness

        return data_completeness(self._data.glucose, self._data.timestamps, interval_minutes)

    @property
    def _current_targets(self) -> GlucoseTargets:
        return self._data.targets

    def TIR(self, targets: GlucoseTargets | None = None) -> float:
        if targets is None:
            targets = self._current_targets
        return self._cached(
            f"_tir_{targets.name}", lambda: self._tir_pure(targets.target_low, targets.target_high)
        )

    def TIR_tight(self, targets: GlucoseTargets | None = None) -> float:
        return self._tir_pure(70, 140)

    def TIR_pregnancy(self, targets: GlucoseTargets | None = None) -> float:
        return self._tir_pure(63, 140)

    def TAR140(self) -> float:
        return self._tar_pure(140)

    def TAR180(self) -> float:
        return self._cached("_tar180", lambda: self._tar_pure(180))

    def TAR250(self) -> float:
        return self._tar_pure(250)

    def TBR70(self) -> float:
        return self._cached("_tbr70", lambda: self._tbr_pure(70))

    def TBR63(self) -> float:
        return self._tbr_pure(63)

    def TBR55(self) -> float:
        return self._tbr_pure(55)

    def TBR_very_low(self, threshold: float = 54) -> float:
        return self._tbr_pure(self._current_targets.hypo_level2)

    def TBR(self, threshold: float) -> float:
        return self._tbr_pure(threshold)

    def TAR(self, threshold: float) -> float:
        return self._tar_pure(threshold)

    def TAR_total(self) -> float:
        return self._tar_pure(self._current_targets.target_high)

    def TBR_total(self) -> float:
        return self._tbr_pure(self._current_targets.target_low)

    def data_completeness(self, expected_interval_minutes: float | None = None) -> int:
        if expected_interval_minutes is None:
            expected_interval_minutes = self._data.get_typical_interval()
        return self._cached(
            "_completeness", lambda: int(self._data_completeness_pure(expected_interval_minutes))
        )

    def time_statistics(self) -> dict[str, Any]:
        t = self._current_targets
        return {
            "target_name": t.name,
            "%Data": self.data_completeness(),
            "TIR": self.TIR(),
            "TBR_L1": self._tir_pure(t.hypo_level2, t.hypo_level1),
            "TBR_L2": self._tbr_pure(t.hypo_level2),
            "TAR_L1": self._tir_pure(t.target_high + 1, t.hyper_level2),
            "TAR_L2": self._tar_pure(t.hyper_level2),
            "TBR_total": self._tbr_pure(t.target_low),
            "TAR_total": self._tar_pure(t.target_high),
        }

    def time_range_summary(self) -> dict[str, Any]:
        t = self._current_targets
        return {
            "data_completeness": self.data_completeness(),
            "current_targets": {
                "name": t.name,
                "TIR": self.TIR(),
                "TBR_total": self._tbr_pure(t.target_low),
                "TBR_L1": self._tir_pure(t.hypo_level2, t.hypo_level1),
                "TBR_L2": self._tbr_pure(t.hypo_level2),
                "TAR_total": self._tar_pure(t.target_high),
                "TAR_L1": self._tir_pure(t.target_high + 1, t.hyper_level2),
                "TAR_L2": self._tar_pure(t.hyper_level2),
            },
            "standard_ranges": {
                "TIR": self._tir_pure(70, 180),
                "TAR180": self._tir_pure(181, 250),
                "TAR250": self._tar_pure(250),
                "TBR70": self._tir_pure(54, 70),
                "TBR54": self._tbr_pure(54),
            },
            "pregnancy_ranges": {
                "TIR_pregnancy": self._tir_pure(63, 140),
                "TBR63_L1": self._tir_pure(55, 63),
                "TBR55": self._tbr_pure(55),
                "TAR140_L1": self._tir_pure(141, 180),
                "TAR140_total": self._tar_pure(140),
            },
        }

    # ── Variability: SD metrics ─────────────────────────────────────────

    def sd_total(self) -> dict:
        from ..metrics.variability.sd import mean_global, sd_global

        return {"sd": sd_global(self._data.glucose), "mean": mean_global(self._data.glucose)}

    def sd_within_day(self, min_count_threshold: float = 0.5) -> dict:
        from ..metrics.variability.sd import sd_within_day

        return sd_within_day(self._data.glucose, self._data.timestamps, min_count_threshold)

    def sdw(self, min_count_threshold: float = 0.5) -> float:
        from ..metrics.variability.sd import sdw

        return sdw(self._data.glucose, self._data.timestamps, min_count_threshold)

    def sd_within_day_segment(self, start_time: str, duration_hours: int) -> dict:
        segment_data = self._get_segment_data(start_time, duration_hours)
        if segment_data.empty:
            return {"sd": 0.0, "mean": 0.0}
        daily_stats = segment_data.groupby(segment_data["time"].dt.date)["glucose"].agg(
            ["std", "mean"]
        )
        return {
            "sd": daily_stats["std"].mean() if not daily_stats.empty else 0.0,
            "mean": daily_stats["mean"].mean() if not daily_stats.empty else 0.0,
        }

    def sd_between_timepoints(
        self,
        min_count_threshold: float = 0.5,
        filter_outliers: bool = True,
        group_by_intervals: bool = False,
        interval_minutes: int = 5,
    ) -> dict[str, Any]:
        from ..metrics.variability.sd import sd_between_timepoints

        return sd_between_timepoints(
            self._data.glucose,
            self._data.timestamps,
            min_count_threshold,
            filter_outliers,
            group_by_intervals,
            interval_minutes,
        )

    def sd_between_timepoints_segment(self, start_time: str, duration_hours: int) -> dict:
        segment_data = self._get_segment_data(start_time, duration_hours)
        time_avg = segment_data.groupby(segment_data["time"].dt.strftime("%H:%M"))["glucose"].mean()
        return {
            "sd": time_avg.std() if not time_avg.empty else 0.0,
            "mean": time_avg.mean() if not time_avg.empty else 0.0,
        }

    def sd_within_series(self, hours: int = 1) -> dict:
        from ..metrics.variability.sd import sd_within_series

        return sd_within_series(self._data.glucose, self._data.timestamps, hours)

    def sd_daily_mean(self, min_count_threshold: float = 0.5) -> dict:
        from ..metrics.variability.sd import sd_daily_mean

        return sd_daily_mean(self._data.glucose, self._data.timestamps, min_count_threshold)

    def sd_same_timepoint(
        self,
        min_count_threshold: float = 0.5,
        filter_outliers: bool = True,
        group_by_intervals: bool = False,
        interval_minutes: int = 5,
    ) -> dict:
        from ..metrics.variability.sd import sd_same_timepoint

        return sd_same_timepoint(
            self._data.glucose,
            self._data.timestamps,
            min_count_threshold,
            filter_outliers,
            group_by_intervals,
            interval_minutes,
        )

    def sd_same_timepoint_adjusted(self) -> dict:
        from ..metrics.variability.sd import sd_same_timepoint_adjusted

        return sd_same_timepoint_adjusted(self._data.glucose, self._data.timestamps)

    def sd_interaction(self) -> dict:
        from ..metrics.variability.sd import sd_interaction

        return sd_interaction(self._data.glucose, self._data.timestamps)

    def sd_segment(self, start_time: str, duration_hours: int) -> dict:
        from ..metrics.variability.sd import sd_segment

        return sd_segment(self._data.glucose, self._data.timestamps, start_time, duration_hours)

    def _get_segment_data(self, start_time: str, duration_hours: int) -> pd.DataFrame:
        from ..metrics.variability.sd import _get_segment_mask

        mask = _get_segment_mask(self._data.timestamps, start_time, duration_hours)
        return self._data.data[mask].copy()

    def calculate_all_sd_metrics(self) -> dict:
        return {
            "SDT": self.sd_total()["sd"],
            "SDw": self.sd_within_day()["sd"],
            "SDhh:mm": self.sd_between_timepoints()["sd"],
            "SDNight": self.sd_segment("00:00", 8)["sd"],
            "Day": self.sd_segment("08:00", 8)["sd"],
            "SDEvening": self.sd_segment("16:00", 8)["sd"],
            "SDws_1h": self.sd_within_series(hours=1)["sd"],
            "SDws_6h": self.sd_within_series(hours=6)["sd"],
            "SDws_24h": self.sd_within_series(hours=24)["sd"],
            "SDdm": self.sd_daily_mean()["sd"],
            "SDbhh:mm": self.sd_same_timepoint()["sd"],
            "SDbhh:mm_dm": self.sd_same_timepoint_adjusted()["sd"],
            "SDI": self.sd_interaction()["sd"],
        }

    def calculate_all_cv_metrics(self) -> dict:
        return {
            "CVT": self.sd_total()["sd"] / self.sd_total()["mean"] * 100,
            "CVw": self.sd_within_day()["sd"] / self.sd_within_day()["mean"] * 100,
            "CVhh:mm": self.sd_between_timepoints()["sd"]
            / self.sd_between_timepoints()["mean"]
            * 100,
            "CVNight": self.sd_segment("00:00", 8)["sd"]
            / self.sd_segment("00:00", 8)["mean"]
            * 100,
            "CVDay": self.sd_segment("08:00", 8)["sd"] / self.sd_segment("08:00", 8)["mean"] * 100,
            "CVEvening": self.sd_segment("16:00", 8)["sd"]
            / self.sd_segment("16:00", 8)["mean"]
            * 100,
            "CVSDws_1h": self.sd_within_series(hours=1)["sd"]
            / self.sd_within_series(hours=1)["mean"]
            * 100,
            "CVSDws_6h": self.sd_within_series(hours=6)["sd"]
            / self.sd_within_series(hours=6)["mean"]
            * 100,
            "CVSDws_24h": self.sd_within_series(hours=24)["sd"]
            / self.sd_within_series(hours=24)["mean"]
            * 100,
            "CVdm": self.sd_daily_mean()["sd"] / self.sd_daily_mean()["mean"] * 100,
            "CVbhh:mm": self.sd_same_timepoint()["sd"] / self.sd_same_timepoint()["mean"] * 100,
            "CVbhh:mm_dm": self.sd_same_timepoint_adjusted()["sd"]
            / self.sd_same_timepoint_adjusted()["mean"]
            * 100,
            "CVSDI": self.sd_interaction()["sd"] / self.sd_interaction()["mean"] * 100,
        }

    # ── Variability: MAGE ───────────────────────────────────────────────

    def MAGE(self, threshold: float | None = None) -> float:
        from ..metrics.variability.mage import mage_simple

        if threshold is None:
            threshold = self.sd()
        return mage_simple(self._data.glucose, threshold)

    def MAGE_Baghurst(
        self,
        threshold: float | None = None,
        threshold_sd: int | None = None,
        approach: int = 1,
        plot: bool = False,
    ) -> dict:
        from ..metrics.variability.mage import (
            mage_baghurst,
            mage_baghurst_direct_elimination,
            mage_baghurst_simplified,
            mage_baghurst_smoothing,
        )

        if threshold is None:
            if threshold_sd is not None:
                threshold = threshold_sd * self.sd()
            else:
                threshold = self.sd()

        glucose_vals = self._data.glucose

        # Guard: short-circuit small / constant datasets
        if len(glucose_vals) < 9 or self.sd() == 0:
            return {
                "MAGE+": 0.0,
                "MAGE-": 0.0,
                "MAGE_avg": 0.0,
                "SD_used": round(float(self.sd()), 2),
                "threshold": round(float(threshold), 2),
                "num_excursions": 0,
                "turning_points": [],
            }

        if not plot:
            fn = {
                1: mage_baghurst_smoothing,
                2: mage_baghurst_direct_elimination,
                3: mage_baghurst_simplified,
            }.get(approach, mage_baghurst)
            return fn(glucose_vals, threshold)

        # Plot path: collect turning points from all 3 approaches, then plot
        from ..plotting.mage_excursions import plot_mage_excursions

        all_results: dict[int, dict] = {}
        for a, fn in [
            (1, mage_baghurst_smoothing),
            (2, mage_baghurst_direct_elimination),
            (3, mage_baghurst_simplified),
        ]:
            all_results[a] = fn(glucose_vals, threshold)

        turning_points = {a: r.get("turning_points", []) for a, r in all_results.items()}

        plot_mage_excursions(
            glucose=glucose_vals.values,
            times=self._data.timestamps.values,
            turning_points_approaches=turning_points,
            threshold=threshold,
            threshold_sd=threshold_sd or 1,
            mean_glucose=self.mean(),
        )

        return all_results.get(approach, all_results[1])

    # ── Variability: MODD, CONGA, Lability ─────────────────────────────

    def MODD(self, days: int = 1) -> dict:
        from ..metrics.variability.modd import modd

        return modd(self._data.glucose, self._data.timestamps, days)

    def CONGA(self, hours: int = 4, max_gap_minutes: float | None = None) -> dict:
        from ..metrics.variability.conga import conga

        return conga(self._data.glucose, self._data.timestamps, hours, max_gap_minutes)

    def Lability_index(self, interval: int = 1, period: str = "week") -> dict:
        from ..metrics.variability.lability import lability_index

        return lability_index(self._data.glucose, self._data.timestamps, interval, period)

    # ── Variability: Risk ──────────────────────────────────────────────

    def M_Value(self, reference_glucose: int = 90) -> float:
        from ..metrics.variability.risk import m_value

        return m_value(self._data.glucose, reference_glucose)

    def j_index(self) -> float:
        from ..metrics.variability.risk import j_index

        return j_index(self.mean(), self.sd())

    def GRADE(self, unit: str = "mg/dL") -> dict:
        from ..metrics.variability.risk import grade

        return grade(self._data.glucose, unit)

    def LBGI(self) -> float:
        from ..metrics.variability.risk import lbgi

        return lbgi(self._data.glucose)

    def HBGI(self) -> float:
        from ..metrics.variability.risk import hbgi

        return hbgi(self._data.glucose)

    def GRI(self, pregnancy: bool = False) -> dict:
        from ..metrics.variability.risk import gri

        return gri(self._data.glucose, pregnancy)

    def ADRR(self) -> dict:
        from ..metrics.variability.risk import adrr

        return adrr(self._data.glucose, self._data.timestamps)

    # ── Variability: Composite ──────────────────────────────────────────

    def calculate_variability_metrics(self) -> dict:
        try:
            metrics: dict[str, Any] = {
                "data_completeness": self.data_completeness(),
                "Mean": self.mean(),
                "Median": self.median(),
                "Std": self.sd(),
                "CV": self.cv(),
                "GMI": self.gmi(),
                "TIR": self.TIR(),
                "TIR_tight": self.TIR_tight(),
                "TIR_pregnancy": self.TIR_pregnancy(),
                "TAR180": self.TAR180(),
                "TAR250": self.TAR250(),
                "TAR140": self.TAR140(),
                "TBR70": self.TBR70(),
                "TBR63": self.TBR63(),
                "TBR55": self.TBR55(),
                "Skewness": float(self._data.glucose.skew()),
                "Kurtosis": float(self._data.glucose.kurtosis()),
            }

            sd = {
                "SDT": self.sd_total().get("sd"),
                "SDW": self.sd_within_day().get("sd"),
                "SD_timepoints": self.sd_between_timepoints().get("sd"),
                "SD_night": self.sd_segment("00:00", 8).get("sd"),
                "SD_day": self.sd_segment("08:00", 8).get("sd"),
                "SD_evening": self.sd_segment("16:00", 8).get("sd"),
                "SD_1h": self.sd_within_series(hours=1).get("sd"),
                "SD_6h": self.sd_within_series(hours=6).get("sd"),
                "SD_24h": self.sd_within_series(hours=24).get("sd"),
                "SD_daily_mean": self.sd_daily_mean().get("sd"),
                "SD_same_timepoint": self.sd_same_timepoint().get("sd"),
                "SD_same_timepoint_adj": self.sd_same_timepoint_adjusted().get("sd"),
                "SD_interaction": self.sd_interaction().get("sd"),
            }
            metrics.update(sd)

            conga = {
                "CONGA1": self.CONGA(hours=1).get("value"),
                "CONGA2": self.CONGA(hours=2).get("value"),
                "CONGA4": self.CONGA(hours=4).get("value"),
                "CONGA6": self.CONGA(hours=6).get("value"),
                "CONGA24": self.CONGA(hours=24).get("value"),
            }
            metrics.update(conga)

            try:
                mage_results = self.MAGE_Baghurst()
                metrics.update(
                    {
                        "mage_plus": mage_results.get("MAGE+"),
                        "mage_minus": mage_results.get("MAGE-"),
                        "mage_avg": mage_results.get("MAGE_avg"),
                        "mage_sd": mage_results.get("SD_used"),
                        "mage_threshold": mage_results.get("threshold"),
                        "mage_excursions": mage_results.get("num_excursions"),
                    }
                )
            except Exception:
                pass

            try:
                modd_result = self.MODD()
                metrics.update(
                    {
                        "modd": modd_result.get("value"),
                        "modd_sd": modd_result.get("std"),
                    }
                )
            except Exception:
                pass

            try:
                lgbi = self.LBGI()
                hbgi = self.HBGI()
                adrr = self.ADRR()
                gri = self.GRI()
                gri_pregnancy = self.GRI(pregnancy=True)
                grade = self.GRADE()
                m_value = self.M_Value()
                j_index = self.j_index()

                risk = {
                    "LBGI": lgbi,
                    "HBGI": hbgi,
                    "ADRR": adrr.get("adrr") if isinstance(adrr, dict) else adrr,
                    "GRI": gri.get("GRI") if isinstance(gri, dict) else gri,
                    "GRI_high": gri.get("derived_metrics", {}).get("hyper_component", 0),
                    "GRI_low": gri.get("derived_metrics", {}).get("hypo_component", 0),
                    "GRI_pregnancy": gri_pregnancy.get("GRI")
                    if isinstance(gri_pregnancy, dict)
                    else gri_pregnancy,
                    "GRI_pregnancy_high": gri_pregnancy.get("derived_metrics", {}).get(
                        "hyper_component", 0
                    ),
                    "GRI_pregnancy_low": gri_pregnancy.get("derived_metrics", {}).get(
                        "hypo_component", 0
                    ),
                    "GRADE": grade.get("grade_score") if isinstance(grade, dict) else grade,
                    "M_Value": m_value if not isinstance(m_value, dict) else m_value.get("M_Value"),  # type: ignore[attr-defined]
                    "J_Index": j_index,
                }

                metrics.update(risk)

            except Exception:
                pass

            return metrics

        except Exception as e:
            return {"error": str(e), "message": "Error calculating metrics"}

    # ── Plots ────────────────────────────────────────────────────────────

    def plot_agp(self, smoothing_window: int = 15):
        from ..plotting.agp import plot_agp as _plot_agp

        return _plot_agp(self._data.data, smoothing_window)

    def plot_daily(self, date: str | None = None):
        from ..plotting.daily_plots import day_graph

        return day_graph(self._data.data, date)

    def plot_daily_overlay(self):
        return self.plot_overlapping_days()

    def plot_overlapping_days(self):
        from ..plotting.daily_plots import plot_overlapping_days

        return plot_overlapping_days(self._data.data)

    def plot_week_boxplots(self):
        from ..plotting.daily_plots import plot_week_boxplots

        return plot_week_boxplots(self._data.data)

    def generate_week_agp(self, smoothing_window: int = 15, combined: bool = True):
        from ..plotting.agp import generate_week_agp as _generate_week_agp

        return _generate_week_agp(
            self._data.data, smoothing_window=smoothing_window, combined=combined
        )

    def plot_daily_variations(self):
        from ..plotting.daily_plots import plot_daily_variations

        return plot_daily_variations(self._data.data)

    def plot_correlation_matrix(self, time_segments: list | None = None):
        from ..plotting.statistical_plots import plot_correlation_matrix

        return plot_correlation_matrix(self._data.data, time_segments)

    def plot_time_in_range(self, pregnancy: bool = False):
        from ..plotting.statistical_plots import plot_time_in_range as _plot_time_in_range

        return _plot_time_in_range(
            self._data.data,
            pregnancy=pregnancy,
            tir_pregnancy=self.TIR_pregnancy() if pregnancy else 0,
            tbr63=self.TBR(63) if pregnancy else 0,
            tar140=self.TAR140() if pregnancy else 0,
            tir=self.TIR() if not pregnancy else 0,
            tbr70=self.TBR70() if not pregnancy else 0,
            tbr55=self.TBR55() if not pregnancy else 0,
            tar180=self.TAR180() if not pregnancy else 0,
            tar250=self.TAR250() if not pregnancy else 0,
        )

    def plot_distribution_comparison(self, target_ranges: list[tuple] | None = None):
        from ..plotting.statistical_plots import plot_distribution_comparison

        return plot_distribution_comparison(
            self._data.data,
            target_ranges=target_ranges,
            tir_val=self.TIR(),
            tbr_val=self.TBR(70),
            tar_val=self.TAR(180),
            gmi_val=self.gmi(),
        )

    def plot_distribution(self, target_ranges: list[tuple] | None = None):
        return self.plot_distribution_comparison(target_ranges)

    def histogram(self, bin_width: int = 10):
        from ..plotting.statistical_plots import histogram as _histogram

        return _histogram(self._data.data, bin_width)

    def plot_variability_dashboard(self, figsize: tuple = (14, 10)):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle("Glucose Variability Dashboard", fontsize=16, fontweight="bold")

        # Plot 1: SD metrics bar chart
        sd_metrics = self.calculate_all_sd_metrics()
        sd_keys = list(sd_metrics.keys())
        sd_vals = [sd_metrics[k] if sd_metrics[k] is not None else 0 for k in sd_keys]
        axes[0, 0].bar(range(len(sd_keys)), sd_vals, color="steelblue", alpha=0.7)
        axes[0, 0].set_xticks(range(len(sd_keys)))
        axes[0, 0].set_xticklabels(sd_keys, rotation=45, ha="right", fontsize=8)
        axes[0, 0].set_title("Standard Deviation Metrics")
        axes[0, 0].set_ylabel("SD (mg/dL)")
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: CONGA values bar chart
        conga_keys = ["CONGA1", "CONGA2", "CONGA4", "CONGA6", "CONGA24"]
        conga_vals = []
        for k in conga_keys:
            v = self.CONGA(hours=int(k.replace("CONGA", "")))
            conga_vals.append(v.get("value", 0))
        axes[0, 1].bar(conga_keys, conga_vals, color="coral", alpha=0.7)
        axes[0, 1].set_title("CONGA Metrics")
        axes[0, 1].set_ylabel("CONGA (mg/dL)")
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Risk metrics
        risk_text = (
            f"LBGI: {self.LBGI():.2f}\n"
            f"HBGI: {self.HBGI():.2f}\n"
            f"J-Index: {self.j_index():.1f}\n"
            f"M-Value: {self.M_Value():.2f}\n"
            f"MAGE: {self.MAGE():.1f} mg/dL\n"
            f"MODD: {self.MODD().get('value', 0):.1f} mg/dL"
        )
        axes[1, 0].text(
            0.1,
            0.5,
            risk_text,
            transform=axes[1, 0].transAxes,
            fontsize=12,
            fontfamily="monospace",
            verticalalignment="center",
        )
        axes[1, 0].axis("off")
        axes[1, 0].set_title("Risk & Excursion Metrics")

        # Plot 4: CV metrics
        cv_metrics = self.calculate_all_cv_metrics()
        cv_keys = list(cv_metrics.keys())
        cv_vals = [cv_metrics[k] if cv_metrics[k] is not None else 0 for k in cv_keys]
        axes[1, 1].bar(range(len(cv_keys)), cv_vals, color="seagreen", alpha=0.7)
        axes[1, 1].set_xticks(range(len(cv_keys)))
        axes[1, 1].set_xticklabels(cv_keys, rotation=45, ha="right", fontsize=8)
        axes[1, 1].set_title("Coefficient of Variation Metrics")
        axes[1, 1].set_ylabel("CV (%)")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_glucose_statistics(self, figsize: tuple = (14, 10)):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle("Glucose Statistics", fontsize=16, fontweight="bold")

        glucose_series = self._data.glucose

        # Plot 1: Histogram
        ax = axes[0, 0]
        ax.hist(glucose_series, bins=50, edgecolor="black", alpha=0.7, color="skyblue")
        ax.axvspan(0, 70, color="#ffcccb", alpha=0.3)
        ax.axvspan(70, 180, color="#90ee90", alpha=0.3)
        ax.axvspan(180, 400, color="#ffcccb", alpha=0.3)
        ax.set_title("Glucose Distribution")
        ax.set_xlabel("Glucose (mg/dL)")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)

        # Plot 2: Box plot
        ax2 = axes[0, 1]
        ax2.boxplot(
            glucose_series,
            vert=True,
            patch_artist=True,
            boxprops={"facecolor": "lightblue"},
        )
        ax2.set_title("Glucose Box Plot")
        ax2.set_ylabel("Glucose (mg/dL)")
        ax2.grid(True, alpha=0.3)

        # Plot 3: Stats text
        stats = self.calculate_all_metrics()
        stats_text = (
            f"DESCRIPTIVE STATISTICS\n\n"
            f"Mean:           {stats['Mean']:.1f} mg/dL\n"
            f"Median:         {stats['Median']:.1f} mg/dL\n"
            f"Std Dev:        {stats['Std']:.1f} mg/dL\n"
            f"CV:             {stats['CV']:.1f}%\n\n"
            f"Percentiles:\n"
            f"P5:             {stats['P5']:.1f} mg/dL\n"
            f"P25:            {stats['P25']:.1f} mg/dL\n"
            f"P75:            {stats['P75']:.1f} mg/dL\n"
            f"P95:            {stats['P95']:.1f} mg/dL\n\n"
            f"Time in Range:\n"
            f"TIR (70-180):   {self.TIR():.1f}%\n"
            f"TBR (<70):      {self.TBR70():.1f}%\n"
            f"TAR (>180):     {self.TAR180():.1f}%\n\n"
            f"GMI:            {self.gmi():.1f}%"
        )
        axes[1, 0].text(
            0.1,
            0.9,
            stats_text,
            transform=axes[1, 0].transAxes,
            fontsize=10,
            fontfamily="monospace",
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "lightgray", "alpha": 0.8},
        )
        axes[1, 0].axis("off")
        axes[1, 0].set_title("Summary Statistics")

        # Plot 4: Time in range pie chart
        ax4 = axes[1, 1]
        tir = self.TIR()
        tbr70 = self.TBR70()
        tbr55 = self.TBR55()
        tar180 = self.TAR180()
        tar250 = self.TAR250()
        labels = [
            "TIR\n(70-180)",
            "TBR L1\n(55-70)",
            "TBR L2\n(<55)",
            "TAR L1\n(180-250)",
            "TAR L2\n(>250)",
        ]
        sizes = [tir, tbr70, tbr55, tar180, tar250]
        colors_pie = ["#90ee90", "#ffeb9c", "#ffcccb", "#ffa500", "#ff6666"]
        non_zero = [
            (label, sz, clr)
            for label, sz, clr in zip(labels, sizes, colors_pie, strict=False)
            if sz > 0
        ]
        if non_zero:
            labels, sizes, colors_pie = map(list, zip(*non_zero, strict=False))
        ax4.pie(sizes, labels=labels, colors=colors_pie, autopct="%1.1f%%", startangle=90)
        ax4.set_title("Time in Range")

        plt.tight_layout()
        plt.show()

    def plot_comprehensive_dashboard(self, figsize: tuple = (20, 12)):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle("Comprehensive Glucose Analysis Dashboard", fontsize=16, fontweight="bold")

        self.plot_agp()
        axes[0, 0].set_title("Ambulatory Profile (AGP)")

        self.histogram()
        axes[0, 1].set_title("Glucose Distribution")

        self.plot_time_in_range()
        axes[0, 2].set_title("Time in Range")

        self.plot_variability_dashboard()
        axes[1, 0].set_title("Variability Analysis")

        self.plot_overlapping_days()
        axes[1, 1].set_title("Overlapping Days")

        self.plot_week_boxplots()
        axes[1, 2].set_title("Weekly Boxplots")

        plt.tight_layout()
        plt.show()

    # ── Reports ─────────────────────────────────────────────────────────

    def report(self) -> dict[str, Any]:
        return {
            "basic_info": self.info(),
            "basic_metrics": self.calculate_all_metrics(),
            "time_statistics": self.time_statistics(),
            "variability_metrics": self.calculate_variability_metrics(),
            "data_quality": self.get_data_quality_metrics(),
        }

    get_comprehensive_report = report

    def get_summary_string(self) -> str:
        summary: list[str] = []
        summary.append("=== COMPREHENSIVE GLUCOSE ANALYSIS ===")
        summary.append("")

        info = self.info()
        summary.append("DATA:")
        summary.append(f"  - Records: {info['n_records']:,}")
        summary.append(
            f"  - Period: {info['start_date'].strftime('%d/%m/%Y')} - "
            f"{info['end_date'].strftime('%d/%m/%Y')}"
        )
        summary.append(f"  - Availability: {info['completeness']:.1f}%")
        summary.append("")

        basic = self.calculate_all_metrics()
        summary.append("BASIC METRICS:")
        summary.append(f"  - GMI: {basic['GMI']:.1f}%")
        summary.append(f"  - Mean: {basic['Mean']:.1f} mg/dL")
        summary.append(f"  - Median: {basic['Median']:.1f} mg/dL")
        summary.append(f"  - Std Dev: {basic['Std']:.1f} mg/dL")
        summary.append(f"  - CV: {basic['CV']:.1f}%")
        summary.append("")

        summary.append("TIME IN RANGE:")
        summary.append(f"  - TIR (70-180): {self.TIR():.1f}%")
        summary.append(f"  - TIR tight (70-140): {self.TIR_tight():.1f}%")
        summary.append(f"  - TBR70 (54-70): {self.TBR70():.1f}%")
        summary.append(f"  - TBR55 (<55): {self.TBR55():.1f}%")
        summary.append(f"  - TAR140 (>140): {self.TAR140():.1f}%")
        summary.append(f"  - TAR180 (181-250): {self.TAR180():.1f}%")
        summary.append(f"  - TAR250 (>250): {self.TAR250():.1f}%")
        summary.append("")

        variability = self.calculate_variability_metrics()
        summary.append("VARIABILITY:")
        summary.append(f"  - MAGE: {variability.get('mage_avg', 'N/A')}")
        summary.append(f"  - MODD: {variability.get('modd', 'N/A')}")
        summary.append(f"  - CONGA4: {variability.get('CONGA4', 'N/A')}")
        summary.append(f"  - SD total: {variability.get('SDT', 'N/A')}")
        summary.append(f"  - SD within-day: {variability.get('SDW', 'N/A')}")
        summary.append(f"  - SD between-day: {variability.get('SD_timepoints', 'N/A')}")

        return "\n".join(summary)

    def export_report(self, file_path: str, format: str = "json"):
        report = self.report()

        if format.lower() == "json":
            import json

            with Path(file_path).open("w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str, ensure_ascii=False)

        elif format.lower() == "csv":
            flat_report = self._flatten_report(report)
            flat_report.to_csv(file_path, index=False)

        elif format.lower() == "excel":
            with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
                pd.DataFrame([report["basic_info"]]).to_excel(
                    writer, sheet_name="Basic_Info", index=False
                )
                pd.DataFrame([report["basic_metrics"]]).to_excel(
                    writer, sheet_name="Basic_Metrics", index=False
                )
                pd.DataFrame([report["time_statistics"]]).to_excel(
                    writer, sheet_name="Time_Range", index=False
                )

        else:
            raise ValueError(f"Unsupported format: {format}")

    @staticmethod
    def _flatten_report(report: dict[str, Any]) -> pd.DataFrame:
        flat_data: dict[str, Any] = {}
        for section, data in report.items():
            if isinstance(data, dict):
                for key, value in data.items():
                    flat_data[f"{section}_{key}"] = value
            else:
                flat_data[section] = data
        return pd.DataFrame([flat_data])
