"""Standard deviation and coefficient-of-variation metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ._base import VariabilityBase

if TYPE_CHECKING:
    pass


class SDMetrics(VariabilityBase):
    """Mixin providing SD/CV metrics for glucose data.

    All methods assume the host class also provides the
    :class:`~cgmpy.data.core.ModularGlucoseData` interface (``data``,
    ``typical_interval``, ``mean``, ``sd``, ``cv``, etc.).
    """

    if TYPE_CHECKING:
        data: pd.DataFrame
        typical_interval: float

        def sd(self) -> float: ...
        def mean(self) -> float: ...
        def median(self) -> float: ...
        def cv(self) -> float: ...
        def gmi(self) -> float: ...
        def data_completeness(self) -> float: ...

    def sd_total(self) -> dict:
        """
        Calculates total standard deviation (SDT) and global mean.
        Returns: {'sd': float, 'mean': float}
        """
        return {"sd": self.sd(), "mean": self.mean()}

    def sd_within_day(self, min_count_threshold: float = 0.5) -> dict:
        """
        Calculates within-day standard deviation (SDw).

        This metric reflects variability within each day, averaged across
        all available days.

        Optimized version for large datasets.

        :param min_count_threshold: Threshold to consider a day valid
                               (proportion of count median).
        :return: Dictionary with SDw value and related statistics.
        """
        # Create efficient copy with only the necessary columns
        df = self.data[["time", "glucose"]].copy()

        # Extract date in a vectorized way
        df["date"] = df["time"].dt.date

        # Calculate per-day statistics vectorized
        daily_stats = df.groupby("date").agg({"glucose": ["std", "mean", "count"]})

        # Calculate threshold to filter days with few data points
        median_count = daily_stats[("glucose", "count")].median()
        threshold = median_count * min_count_threshold

        # Store information about all days
        all_days_sds = daily_stats[("glucose", "std")].to_dict()
        all_days_means = daily_stats[("glucose", "mean")].to_dict()
        all_days_counts = daily_stats[("glucose", "count")].to_dict()

        # Filter days with enough data
        valid_days = daily_stats[daily_stats[("glucose", "count")] >= threshold]

        if valid_days.empty:
            return {
                "sd": 0.0,
                "mean": 0.0,
                "valid_days": 0,
                "total_days": len(daily_stats),
                "daily_sds": {},
                "daily_means": {},
                "daily_counts": {},
                "all_days_sds": all_days_sds,
                "all_days_means": all_days_means,
                "all_days_counts": all_days_counts,
                "threshold": threshold,
            }

        # Calculate SDw (average of daily standard deviations)
        sd_value = valid_days[("glucose", "std")].mean()
        mean_value = valid_days[("glucose", "mean")].mean()

        # Prepare result with additional statistics
        result = {
            "sd": sd_value,
            "mean": mean_value,
            "valid_days": len(valid_days),
            "total_days": len(daily_stats),
            "daily_sds": valid_days[("glucose", "std")].to_dict(),
            "daily_means": valid_days[("glucose", "mean")].to_dict(),
            "daily_counts": valid_days[("glucose", "count")].to_dict(),
            "all_days_sds": all_days_sds,
            "all_days_means": all_days_means,
            "all_days_counts": all_days_counts,
            "threshold": threshold,
        }

        return result

    def sdw(self, min_count_threshold: float = 0.5) -> float:
        """
        Calculates within-day standard deviation (SDw).
        This is a simplified method returning only the SD value.

        :param min_count_threshold: Threshold to consider a day valid
                                   (proportion of count median). Default 0.5 (50%).
        :return: SDw value (float)
        """
        return self.sd_within_day(min_count_threshold)["sd"]

    def sd_within_day_segment(self, start_time: str, duration_hours: int) -> dict[str, float]:
        """
        Calculates within-day standard deviation for a specific day segment.
        For each day, calculates the SD of the specified time segment and then
        averages these daily SDs.

        :param start_time: Start time in "HH:MM" format
        :param duration_hours: Duration of the segment in hours
        :return: Average of segment SDs for each day and average of daily means

        Example:
            sd_within_day_segment("00:00", 8)  # Average SD of night segment (00:00-08:00)
            sd_within_day_segment("08:00", 8)  # Average SD of day segment (08:00-16:00)
        """
        # Get data for the segment
        segment_data = self._get_segment_data(start_time, duration_hours)

        if segment_data.empty:
            return {"sd": 0.0, "mean": 0.0}

        # Calculate SD and mean for each day's segment
        daily_segment_stats = segment_data.groupby(segment_data["time"].dt.date)["glucose"].agg(
            ["std", "mean"]
        )

        return {
            "sd": daily_segment_stats["std"].mean() if not daily_segment_stats.empty else 0.0,
            "mean": daily_segment_stats["mean"].mean() if not daily_segment_stats.empty else 0.0,
        }

    def sd_between_timepoints(
        self,
        min_count_threshold: float = 0.5,
        filter_outliers: bool = True,
        agrupar_por_intervalos: bool = False,
        intervalo_minutos: int = 5,
    ) -> dict[str, Any]:
        """
        Calculates standard deviation between timepoints (SDhh:mm).
        Calculates the mean of a timestamp and then the standard deviation of those means.

        This metric measures the variability of the glucose pattern throughout the day.

        Optimized version for large datasets.

        :param min_count_threshold: Threshold to consider a timestamp valid
                               (proportion of count median). Default 0.5 (50%).
        :param filter_outliers: If True, filters timestamps with few data points.
        :param agrupar_por_intervalos: If True, groups data into regular time intervals.
        :param intervalo_minutos: Interval size in minutes for grouping (default 5 min).
        :return: Dictionary with SDhh:mm value and related statistics.
        """
        # Create a copy of the data with only the necessary columns
        df = self.data[["time", "glucose"]].copy()

        # Extract only hour and minute to reduce the computational load
        df["hour_min"] = df["time"].apply(lambda x: x.hour * 60 + x.minute)

        if agrupar_por_intervalos:
            # Group by time intervals to reduce the number of time points
            df["interval"] = (df["hour_min"] // intervalo_minutos) * intervalo_minutos
            # Use groupby with transform, more efficient than apply for large datasets
            grouped = df.groupby(["day", "interval"])
            df_agg = grouped.agg({"glucose": "mean"}).reset_index()

            # Calculate hour and minute from the interval for the final result
            df_agg["hour"] = df_agg["interval"] // 60
            df_agg["minute"] = df_agg["interval"] % 60

            # Use vectorized descriptive statistics
            timepoint_means = df_agg.groupby(["hour", "minute"])["glucose"].mean()
            df_final = pd.DataFrame({"mean": timepoint_means})

            # Count number of days with data for each time point
            timepoint_counts = df_agg.groupby(["hour", "minute"]).size()
            df_final["count"] = timepoint_counts
        else:
            # Extract time features in a vectorized way
            df["hour"] = df["time"].dt.hour
            df["minute"] = df["time"].dt.minute
            df["day"] = df["time"].dt.date

            # Calculate hour:minute statistics in a vectorized way
            grouped = df.groupby(["hour", "minute", "day"])
            daily_means = grouped["glucose"].mean().reset_index()

            # Group again to obtain the means per time point
            timepoint_stats = daily_means.groupby(["hour", "minute"])

            # Calculate statistics in a vectorized way
            df_final = pd.DataFrame(
                {
                    "mean": timepoint_stats["glucose"].mean(),
                    "count": timepoint_stats.size(),
                }
            )

        # Filter points with few data points if requested
        if filter_outliers:
            median_count = df_final["count"].median()
            threshold = median_count * min_count_threshold
            valid_timepoints = df_final[df_final["count"] >= threshold]
        else:
            valid_timepoints = df_final

        # Calculate SDhh:mm (standard deviation of the average pattern)
        sd_value = valid_timepoints["mean"].std()
        mean_value = valid_timepoints["mean"].mean()

        # Create the result as a dictionary
        result = {
            "sd": sd_value,
            "mean": mean_value,
            "valid_timepoints": len(valid_timepoints),
            "total_timepoints": len(df_final),
            "median_count": df_final["count"].median(),
            "min_count": df_final["count"].min(),
            "max_count": df_final["count"].max(),
        }

        return result

    def sd_between_timepoints_segment(self, start_time: str, duration_hours: int) -> dict:
        """
        Calculates SDhh:mm for a specific day segment.

        :param start_time: Start time in "HH:MM" format
        :param duration_hours: Duration of the segment in hours
        :return: SD of the average pattern by time of day in the segment
        """
        # Filter the segment first
        segment_data = self._get_segment_data(start_time, duration_hours)

        # Group by "HH:MM" timestamp within the segment
        time_avg = segment_data.groupby(segment_data["time"].dt.strftime("%H:%M"))["glucose"].mean()
        return {
            "sd": time_avg.std() if not time_avg.empty else 0.0,
            "mean": time_avg.mean() if not time_avg.empty else 0.0,
        }

    def sd_within_series(self, hours: int = 1) -> dict:
        """
        Calculates SDws and mean of time series.

        The fewer hours, the smaller the SDws value because
        it gives less time for glucose to vary.

        Optimized version for large datasets.

        :param hours: Window size in hours
        :return: Dictionary with average SD and mean of time series
        """
        # Create efficient copy with only the necessary columns
        df = self.data[["time", "glucose"]].copy()

        # Ensure data is sorted by time
        df = df.sort_values("time")

        # Convert hours to nanoseconds for the time window
        # window_ns = pd.Timedelta(hours=hours).value

        # Arrays to store results
        sd_values = []
        mean_values = []

        # Sample at regular intervals to reduce the computational load
        # Adjust step based on your dataset size (larger step = faster but less precise)
        step = max(1, len(df) // 1000)  # Cap at ~1000 windows maximum

        for i in range(0, len(df), step):
            start_time = df.iloc[i]["time"]
            end_time = start_time + pd.Timedelta(hours=hours)

            # Filter data within the current time window
            window_data = df[(df["time"] >= start_time) & (df["time"] < end_time)]

            # Only compute statistics if there are enough points in the window
            if len(window_data) > 2:  # Need at least 3 points for a good estimate
                sd_values.append(window_data["glucose"].std())
                mean_values.append(window_data["glucose"].mean())

        # Compute averages
        result = {
            "sd": np.mean(sd_values) if sd_values else 0.0,
            "mean": np.mean(mean_values) if mean_values else 0.0,
            "windows_analyzed": len(sd_values),
        }

        return result

    def sd_daily_mean(self, min_count_threshold: float = 0.5) -> dict:
        """
        Calculates standard deviation of daily means (SDdm).

        This metric reflects variability between different days.

        Optimized version for large datasets.

        :param min_count_threshold: Threshold to consider a day valid
                               (proportion of count median).
        :return: Dictionary with SDdm value and related statistics.
        """
        # Create efficient copy with only the necessary columns
        df = self.data[["time", "glucose"]].copy()

        # Extract date in a vectorized way
        df["date"] = df["time"].dt.date

        # Calculate per-day statistics vectorized
        daily_stats = df.groupby("date").agg({"glucose": ["mean", "count"]})

        # Calculate threshold to filter days with few data points
        median_count = daily_stats[("glucose", "count")].median()
        threshold = median_count * min_count_threshold

        # Filter days with enough data
        valid_days = daily_stats[daily_stats[("glucose", "count")] >= threshold]

        if valid_days.empty:
            return {"sd": 0.0, "mean": 0.0}

        # Calculate SDdm (SD of daily means)
        sd_value = valid_days[("glucose", "mean")].std()
        mean_value = valid_days[("glucose", "mean")].mean()

        # Prepare result with additional statistics
        result = {
            "sd": sd_value,
            "mean": mean_value,
            "valid_days": len(valid_days),
            "total_days": len(daily_stats),
            "daily_means": valid_days[("glucose", "mean")].to_dict(),
            "daily_counts": valid_days[("glucose", "count")].to_dict(),
        }

        return result

    def sd_same_timepoint(
        self,
        min_count_threshold: float = 0.5,
        filter_outliers: bool = True,
        agrupar_por_intervalos: bool = False,
        intervalo_minutos: int = 5,
    ) -> dict:
        """
        Calculates between-day standard deviation for each timepoint (SDbhh:mm).

        This function measures glucose variability for each specific timepoint
        across different days, reflecting day-to-day consistency.

        Optimized version for large datasets.

        :param min_count_threshold: Threshold to consider a timestamp valid
                               (proportion of count median).
        :param filter_outliers: If True, filters timestamps with few data points.
        :param agrupar_por_intervalos: If True, groups data into regular intervals.
        :param intervalo_minutos: Interval size in minutes (default 5 min).
        :return: Dictionary with SDbhh:mm value and related statistics.
        """
        # Create efficient copy with only the necessary columns
        df = self.data[["time", "glucose"]].copy()

        # Extract time features in a vectorized way
        df["hour"] = df["time"].dt.hour
        df["minute"] = df["time"].dt.minute
        df["day"] = df["time"].dt.date

        if agrupar_por_intervalos:
            # Group by time intervals
            minutes_of_day = df["hour"] * 60 + df["minute"]
            df["interval"] = (minutes_of_day // intervalo_minutos) * intervalo_minutos
            df["hour"] = df["interval"] // 60
            df["minute"] = df["interval"] % 60

        # Create time key for grouping
        df["time_key"] = (
            df["hour"].astype(str).str.zfill(2) + ":" + df["minute"].astype(str).str.zfill(2)
        )

        # Group by day and time point
        grouped = df.groupby(["day", "time_key"])

        # Calculate daily means per time point
        daily_means = grouped["glucose"].mean().reset_index()

        # Group by time point and calculate statistics
        timepoint_stats = daily_means.groupby("time_key")

        # Calculate per-time-point statistics in a vectorized way
        sd_por_marca = timepoint_stats["glucose"].std()
        valores_por_marca = timepoint_stats["glucose"].mean()
        conteo_por_marca = timepoint_stats["glucose"].count()

        # Calculate threshold to filter time points with few data points
        median_count = conteo_por_marca.median()
        threshold = median_count * min_count_threshold

        # Store total time points before filtering
        total_timepoints = len(sd_por_marca)

        # Filter time points with enough data if requested
        if filter_outliers:
            valid_mask = conteo_por_marca >= threshold
            sd_por_marca = sd_por_marca[valid_mask]
            valores_por_marca = valores_por_marca[valid_mask]
            conteo_por_marca = conteo_por_marca[valid_mask]

        if len(sd_por_marca) == 0:
            return {
                "sd": 0.0,
                "mean": 0.0,
                "threshold": threshold,
                "total_timepoints": total_timepoints,
            }

        # Calculate SDbhh:mm (weighted average of standard deviations)
        weights = conteo_por_marca / conteo_por_marca.sum()
        sd_value = (sd_por_marca * weights).sum()
        mean_value = valores_por_marca.mean()

        # Prepare result with additional statistics
        result = {
            "sd": sd_value,
            "mean": mean_value,
            "sd_por_marca": sd_por_marca.to_dict(),
            "valores_por_marca": valores_por_marca.to_dict(),
            "conteo_por_marca": conteo_por_marca.to_dict(),
            "threshold": threshold,
            "total_timepoints": total_timepoints,
        }

        return result

    def sd_same_timepoint_adjusted(self) -> dict:
        """
        Calculates between-day SD for each timepoint, adjusted for changes in daily means.

        The process is:
        1. Adjust glucose values: Adjusted_Glucose = Glucose - Daily_Mean + Total_Mean
        2. Calculate between-day SD for each timepoint using adjusted values
        3. Average the resulting SDs

        Returns:
            dict: Between-day SD adjusted for daily means
        """
        # Calculate the grand mean
        grand_mean = self.data["glucose"].mean()

        # Calculate daily means
        daily_means = self.data.groupby(self.data["time"].dt.date)["glucose"].transform("mean")

        # Adjust glucose values
        adjusted_glucose = self.data["glucose"] - daily_means + grand_mean

        # Create DataFrame with adjusted values
        adjusted_data = self.data.copy()
        adjusted_data["glucose"] = adjusted_glucose

        # Group by specific time of day (HH:MM)
        time_groups = adjusted_data.groupby(adjusted_data["time"].dt.strftime("%H:%M"))

        # Calculate SD for each specific time using adjusted values
        time_sds = time_groups["glucose"].std()

        # Average all SDs
        return {
            "sd": time_sds.mean() if not time_sds.empty else 0.0,
            "mean": grand_mean if not time_sds.empty else 0.0,
        }

    def sd_interaction(self) -> dict:
        """
        Calculates standard deviation of interaction (SDI).

        SDI quantifies daily variability in the glycemic pattern,
        considering interactions between time of day and specific day.

        Optimized version for large datasets.

        :return: Dictionary with SDI value and related statistics.
        """
        # Create efficient copy with only the necessary columns
        df = self.data[["time", "glucose"]].copy()

        # Extract time features in a vectorized way
        df["hour"] = df["time"].dt.hour
        df["minute"] = df["time"].dt.minute
        df["day"] = df["time"].dt.date
        df["time_key"] = (
            df["hour"].astype(str).str.zfill(2) + ":" + df["minute"].astype(str).str.zfill(2)
        )

        # Calculate values required for the SDI formula

        # 1. Calculate the global mean
        global_mean = df["glucose"].mean()

        # 2. Calculate daily means
        daily_means = df.groupby("day")["glucose"].mean()

        # 3. Calculate means per time point
        timepoint_means = df.groupby("time_key")["glucose"].mean()

        # 4. Calculate deviations for each observation
        # Use merge for efficient vectorized operations
        df_temp = df.copy()

        # Convert day_mean and timepoint_mean to DataFrames for merge
        day_mean_df = pd.DataFrame(daily_means).reset_index()
        day_mean_df.columns = ["day", "day_mean"]

        timepoint_mean_df = pd.DataFrame(timepoint_means).reset_index()
        timepoint_mean_df.columns = ["time_key", "timepoint_mean"]

        # Merge efficiently
        df_temp = df_temp.merge(day_mean_df, on="day")
        df_temp = df_temp.merge(timepoint_mean_df, on="time_key")

        # Calculate the interaction for each point
        df_temp["expected"] = (
            global_mean
            + (df_temp["day_mean"] - global_mean)
            + (df_temp["timepoint_mean"] - global_mean)
        )
        df_temp["interaction"] = df_temp["glucose"] - df_temp["expected"]

        # Calculate SDI (standard deviation of the interaction)
        sdi = df_temp["interaction"].std()

        return {"sd": sdi, "mean": global_mean}

    def sd_segment(self, start_time: str, duration_hours: int) -> dict:
        """
        Calculates standard deviation within a specific segment (SDws).

        Useful for analyzing segments like night, day, afternoon, etc.

        :param start_time: Start time in "HH:MM".
        :param duration_hours: Duration of the segment in hours.
        :return: Standard deviation of readings within the segment.
        """
        from datetime import datetime, time, timedelta

        # Convert start_time to a time object
        start_hour, start_minute = map(int, start_time.split(":"))
        start = time(hour=start_hour, minute=start_minute)

        # Calculate the end time (considering midnight crossing)
        base = datetime(2000, 1, 1, start.hour, start.minute)
        end_dt = base + timedelta(hours=duration_hours)
        end = end_dt.time()

        # Filter data based on whether the segment crosses midnight
        if start <= end:
            segment_data = self.data[self.data["time"].apply(lambda dt: start <= dt.time() < end)]
        else:
            segment_data = self.data[
                self.data["time"].apply(lambda dt: dt.time() >= start or dt.time() < end)
            ]

        return {
            "sd": segment_data["glucose"].std() if not segment_data.empty else 0.0,
            "mean": segment_data["glucose"].mean() if not segment_data.empty else 0.0,
        }

    def calculate_all_sd_metrics(self) -> dict:
        """
        Calculates all available standard deviation metrics.
        """
        return {
            "SDT": self.sd_total()["sd"],
            "SDw": self.sd_within_day()["sd"],
            "SDhh:mm": self.sd_between_timepoints()["sd"],
            "Noche": self.sd_segment("00:00", 8)["sd"],
            "Day": self.sd_segment("08:00", 8)["sd"],
            "Tarde": self.sd_segment("16:00", 8)["sd"],
            "SDws_1h": self.sd_within_series(hours=1)["sd"],
            "SDws_6h": self.sd_within_series(hours=6)["sd"],
            "SDws_24h": self.sd_within_series(hours=24)["sd"],
            "SDdm": self.sd_daily_mean()["sd"],
            "SDbhh:mm": self.sd_same_timepoint()["sd"],
            "SDbhh:mm_dm": self.sd_same_timepoint_adjusted()["sd"],
            "SDI": self.sd_interaction()["sd"],
        }

    def calculate_all_cv_metrics(self) -> dict:
        """
        Calculates all available coefficient of variation metrics.
        Todo: Check if means are correct.
        """
        return {
            "CVT": self.sd_total()["sd"] / self.sd_total()["mean"] * 100,
            "CVw": self.sd_within_day()["sd"] / self.sd_within_day()["mean"] * 100,
            "CVhh:mm": self.sd_between_timepoints()["sd"]
            / self.sd_between_timepoints()["mean"]
            * 100,
            "CVNoche": self.sd_segment("00:00", 8)["sd"]
            / self.sd_segment("00:00", 8)["mean"]
            * 100,
            "CVDay": self.sd_segment("08:00", 8)["sd"] / self.sd_segment("08:00", 8)["mean"] * 100,
            "CVTarde": self.sd_segment("16:00", 8)["sd"]
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

    def _get_segment_data(self, start_time: str, duration_hours: int) -> pd.DataFrame:
        """
        Helper method to get data for a specific time segment.

        :param start_time: Start time in "HH:MM" format
        :param duration_hours: Duration of the segment in hours
        :return: DataFrame with data within the specified segment
        """
        from datetime import datetime, time, timedelta

        # Convert start_time to time object
        start_hour, start_minute = map(int, start_time.split(":"))
        start = time(hour=start_hour, minute=start_minute)

        # Calculate end time (handling midnight crossing)
        base = datetime(2000, 1, 1, start.hour, start.minute)
        end_dt = base + timedelta(hours=duration_hours)
        end = end_dt.time()

        # Optimize filtering using vectorized integer comparisons
        # Object-based time comparisons (dt.time) are slow in pandas
        times = self.data["time"]
        minutes_from_midnight = times.dt.hour * 60 + times.dt.minute

        start_min = start.hour * 60 + start.minute
        # Handling end time that might be 24:00 (represented as 00:00 next day)
        end_min = end.hour * 60 + end.minute
        if end_min == 0 and duration_hours > 0:
            end_min = 1440

        if start_min < end_min:
            mask = (minutes_from_midnight >= start_min) & (minutes_from_midnight < end_min)
        else:
            # Crosses midnight (e.g. 22:00 to 06:00)
            mask = (minutes_from_midnight >= start_min) | (minutes_from_midnight < end_min)

        return self.data[mask].copy()
