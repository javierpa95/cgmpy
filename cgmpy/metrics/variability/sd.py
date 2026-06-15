"""Standard deviation and coefficient-of-variation metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass


# ──────────────────────────────────────────────
# Pure functions
# ──────────────────────────────────────────────


def sd_global(glucose: pd.Series) -> float:
    """Calculates total standard deviation (SDT).

    Args:
        glucose: Glucose values.

    Returns:
        Standard deviation of all glucose values.
    """
    return glucose.std()


def mean_global(glucose: pd.Series) -> float:
    """Calculates total mean.

    Args:
        glucose: Glucose values.

    Returns:
        Mean of all glucose values.
    """
    return glucose.mean()


def sd_within_day(
    glucose: pd.Series,
    timestamps: pd.Series,
    min_count_threshold: float = 0.5,
) -> dict:
    """Calculates within-day standard deviation (SDw).

    This metric reflects variability within each day, averaged across
    all available days.

    Args:
        glucose: Glucose values.
        timestamps: Corresponding timestamps.
        min_count_threshold: Threshold proportion of median count to consider
            a day valid.

    Returns:
        Dictionary with SDw value and related statistics.
    """
    df = pd.DataFrame({"time": timestamps, "glucose": glucose})
    df["date"] = df["time"].dt.date
    daily_stats = df.groupby("date").agg({"glucose": ["std", "mean", "count"]})
    median_count = daily_stats[("glucose", "count")].median()
    threshold = median_count * min_count_threshold
    all_days_sds = daily_stats[("glucose", "std")].to_dict()
    all_days_means = daily_stats[("glucose", "mean")].to_dict()
    all_days_counts = daily_stats[("glucose", "count")].to_dict()
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
    sd_value = valid_days[("glucose", "std")].mean()
    mean_value = valid_days[("glucose", "mean")].mean()
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


def sdw(
    glucose: pd.Series,
    timestamps: pd.Series,
    min_count_threshold: float = 0.5,
) -> float:
    """Calculates within-day standard deviation (SDw), returning only the SD value.

    Args:
        glucose: Glucose values.
        timestamps: Corresponding timestamps.
        min_count_threshold: Threshold proportion of median count to consider
            a day valid.

    Returns:
        SDw value.
    """
    return sd_within_day(glucose, timestamps, min_count_threshold)["sd"]


def sd_daily_mean(
    glucose: pd.Series,
    timestamps: pd.Series,
    min_count_threshold: float = 0.5,
) -> dict:
    """Calculates standard deviation of daily means (SDdm).

    This metric reflects variability between different days.

    Args:
        glucose: Glucose values.
        timestamps: Corresponding timestamps.
        min_count_threshold: Threshold proportion of median count to consider
            a day valid.

    Returns:
        Dictionary with SDdm value and related statistics.
    """
    df = pd.DataFrame({"time": timestamps, "glucose": glucose})
    df["date"] = df["time"].dt.date
    daily_stats = df.groupby("date").agg({"glucose": ["mean", "count"]})
    median_count = daily_stats[("glucose", "count")].median()
    threshold = median_count * min_count_threshold
    valid_days = daily_stats[daily_stats[("glucose", "count")] >= threshold]
    if valid_days.empty:
        return {"sd": 0.0, "mean": 0.0}
    sd_value = valid_days[("glucose", "mean")].std()
    mean_value = valid_days[("glucose", "mean")].mean()
    result = {
        "sd": sd_value,
        "mean": mean_value,
        "valid_days": len(valid_days),
        "total_days": len(daily_stats),
        "daily_means": valid_days[("glucose", "mean")].to_dict(),
        "daily_counts": valid_days[("glucose", "count")].to_dict(),
    }
    return result


def sd_between_timepoints(
    glucose: pd.Series,
    timestamps: pd.Series,
    min_count_threshold: float = 0.5,
    filter_outliers: bool = True,
    group_by_intervals: bool = False,
    interval_minutes: int = 5,
) -> dict[str, Any]:
    """Calculates standard deviation between timepoints (SDhh:mm).

    Calculates the mean of each timestamp and then the standard deviation
    of those means. This metric measures the variability of the glucose
    pattern throughout the day.

    Args:
        glucose: Glucose values.
        timestamps: Corresponding timestamps.
        min_count_threshold: Threshold proportion of median count to consider
            a timestamp valid.
        filter_outliers: If True, filters timestamps with few data points.
        group_by_intervals: If True, groups data into regular time intervals.
        interval_minutes: Interval size in minutes for grouping.

    Returns:
        Dictionary with SDhh:mm value and related statistics.
    """
    df = pd.DataFrame({"time": timestamps, "glucose": glucose})
    df["hour_min"] = df["time"].apply(lambda x: x.hour * 60 + x.minute)
    if group_by_intervals:
        df["interval"] = (df["hour_min"] // interval_minutes) * interval_minutes
        df["day"] = df["time"].dt.date
        grouped = df.groupby(["day", "interval"])
        df_agg = grouped.agg({"glucose": "mean"}).reset_index()
        df_agg["hour"] = df_agg["interval"] // 60
        df_agg["minute"] = df_agg["interval"] % 60
        timepoint_means = df_agg.groupby(["hour", "minute"])["glucose"].mean()
        df_final = pd.DataFrame({"mean": timepoint_means})
        timepoint_counts = df_agg.groupby(["hour", "minute"]).size()
        df_final["count"] = timepoint_counts
    else:
        df["hour"] = df["time"].dt.hour
        df["minute"] = df["time"].dt.minute
        df["day"] = df["time"].dt.date
        grouped = df.groupby(["hour", "minute", "day"])
        daily_means = grouped["glucose"].mean().reset_index()
        timepoint_stats = daily_means.groupby(["hour", "minute"])
        df_final = pd.DataFrame(
            {
                "mean": timepoint_stats["glucose"].mean(),
                "count": timepoint_stats.size(),
            }
        )
    if filter_outliers:
        median_count = df_final["count"].median()
        threshold = median_count * min_count_threshold
        valid_timepoints = df_final[df_final["count"] >= threshold]
    else:
        valid_timepoints = df_final
    sd_value = valid_timepoints["mean"].std()
    mean_value = valid_timepoints["mean"].mean()
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


def sd_same_timepoint(
    glucose: pd.Series,
    timestamps: pd.Series,
    min_count_threshold: float = 0.5,
    filter_outliers: bool = True,
    group_by_intervals: bool = False,
    interval_minutes: int = 5,
) -> dict:
    """Calculates between-day standard deviation for each timepoint (SDbhh:mm).

    This function measures glucose variability for each specific timepoint
    across different days, reflecting day-to-day consistency.

    Args:
        glucose: Glucose values.
        timestamps: Corresponding timestamps.
        min_count_threshold: Threshold proportion of median count to consider
            a timestamp valid.
        filter_outliers: If True, filters timestamps with few data points.
        group_by_intervals: If True, groups data into regular intervals.
        interval_minutes: Interval size in minutes.

    Returns:
        Dictionary with SDbhh:mm value and related statistics.
    """
    df = pd.DataFrame({"time": timestamps, "glucose": glucose})
    df["hour"] = df["time"].dt.hour
    df["minute"] = df["time"].dt.minute
    df["day"] = df["time"].dt.date
    if group_by_intervals:
        minutes_of_day = df["hour"] * 60 + df["minute"]
        df["interval"] = (minutes_of_day // interval_minutes) * interval_minutes
        df["hour"] = df["interval"] // 60
        df["minute"] = df["interval"] % 60
    df["time_key"] = (
        df["hour"].astype(str).str.zfill(2) + ":" + df["minute"].astype(str).str.zfill(2)
    )
    grouped = df.groupby(["day", "time_key"])
    daily_means = grouped["glucose"].mean().reset_index()
    timepoint_stats = daily_means.groupby("time_key")
    sd_by_timepoint = timepoint_stats["glucose"].std()
    values_by_timepoint = timepoint_stats["glucose"].mean()
    count_by_timepoint = timepoint_stats["glucose"].count()
    median_count = count_by_timepoint.median()
    threshold = median_count * min_count_threshold
    total_timepoints = len(sd_by_timepoint)
    if filter_outliers:
        valid_mask = count_by_timepoint >= threshold
        sd_by_timepoint = sd_by_timepoint[valid_mask]
        values_by_timepoint = values_by_timepoint[valid_mask]
        count_by_timepoint = count_by_timepoint[valid_mask]
    if len(sd_by_timepoint) == 0:
        return {
            "sd": 0.0,
            "mean": 0.0,
            "threshold": threshold,
            "total_timepoints": total_timepoints,
        }
    weights = count_by_timepoint / count_by_timepoint.sum()
    sd_value = (sd_by_timepoint * weights).sum()
    mean_value = values_by_timepoint.mean()
    result = {
        "sd": sd_value,
        "mean": mean_value,
        "sd_by_timepoint": sd_by_timepoint.to_dict(),
        "values_by_timepoint": values_by_timepoint.to_dict(),
        "count_by_timepoint": count_by_timepoint.to_dict(),
        "threshold": threshold,
        "total_timepoints": total_timepoints,
    }
    return result


def sd_same_timepoint_adjusted(
    glucose: pd.Series,
    timestamps: pd.Series,
) -> dict:
    """Calculates between-day SD for each timepoint, adjusted for daily means.

    The process is:
    1. Adjust glucose values: Adjusted_Glucose = Glucose - Daily_Mean + Total_Mean
    2. Calculate between-day SD for each timepoint using adjusted values
    3. Average the resulting SDs

    Args:
        glucose: Glucose values.
        timestamps: Corresponding timestamps.

    Returns:
        Dictionary with adjusted SDbhh:mm value.
    """
    grand_mean = glucose.mean()
    df = pd.DataFrame({"time": timestamps, "glucose": glucose})
    daily_means = df.groupby(df["time"].dt.date)["glucose"].transform("mean")
    adjusted_glucose = glucose - daily_means + grand_mean
    adjusted_data = pd.DataFrame({"time": timestamps, "glucose": adjusted_glucose})
    time_groups = adjusted_data.groupby(adjusted_data["time"].dt.strftime("%H:%M"))
    time_sds = time_groups["glucose"].std()
    return {
        "sd": time_sds.mean() if not time_sds.empty else 0.0,
        "mean": grand_mean if not time_sds.empty else 0.0,
    }


def _get_segment_mask(
    timestamps: pd.Series,
    start_time: str,
    duration_hours: int,
) -> pd.Series:
    """Returns a boolean mask for data within the specified time segment.

    Args:
        timestamps: Timestamps to filter.
        start_time: Start time in "HH:MM" format.
        duration_hours: Duration of the segment in hours.

    Returns:
        Boolean series mask.
    """
    from datetime import datetime, time, timedelta

    start_hour, start_minute = map(int, start_time.split(":"))
    start = time(hour=start_hour, minute=start_minute)
    base = datetime(2000, 1, 1, start.hour, start.minute)
    end_dt = base + timedelta(hours=duration_hours)
    end = end_dt.time()
    minutes_from_midnight = timestamps.dt.hour * 60 + timestamps.dt.minute
    start_min = start.hour * 60 + start.minute
    end_min = end.hour * 60 + end.minute
    if end_min == 0 and duration_hours > 0:
        end_min = 1440
    if start_min < end_min:
        mask = (minutes_from_midnight >= start_min) & (minutes_from_midnight < end_min)
    else:
        mask = (minutes_from_midnight >= start_min) | (minutes_from_midnight < end_min)
    return mask


def sd_segment(
    glucose: pd.Series,
    timestamps: pd.Series,
    start_time: str,
    duration_hours: int,
) -> dict:
    """Calculates standard deviation within a specific segment.

    Useful for analyzing segments like night, day, afternoon, etc.

    Args:
        glucose: Glucose values.
        timestamps: Corresponding timestamps.
        start_time: Start time in "HH:MM".
        duration_hours: Duration of the segment in hours.

    Returns:
        Dictionary with SD and mean of readings within the segment.
    """
    mask = _get_segment_mask(timestamps, start_time, duration_hours)
    segment_glucose = glucose[mask]
    return {
        "sd": segment_glucose.std() if not segment_glucose.empty else 0.0,
        "mean": segment_glucose.mean() if not segment_glucose.empty else 0.0,
    }


def sd_within_series(
    glucose: pd.Series,
    timestamps: pd.Series,
    hours: int = 1,
) -> dict:
    """Calculates SDws and mean of time series.

    The fewer hours, the smaller the SDws value because
    it gives less time for glucose to vary.

    .. note::

        **Approximation.** For performance on large datasets this method
        samples at most ~1000 sliding windows (``step = max(1, len(df) // 1000)``).
        The result is an **estimate** of the true value, not an exact
        computation. If you need exact results on a small dataset, use a
        different metric (e.g. :func:`sd_global`) or reduce the dataset size
        before calling this function.

    Args:
        glucose: Glucose values.
        timestamps: Corresponding timestamps.
        hours: Window size in hours.

    Returns:
        Dictionary with average SD and mean of time series.
    """
    df = pd.DataFrame({"time": timestamps, "glucose": glucose})
    df = df.sort_values("time")
    sd_values = []
    mean_values = []
    step = max(1, len(df) // 1000)
    for i in range(0, len(df), step):
        start_time = df.iloc[i]["time"]
        end_time = start_time + pd.Timedelta(hours=hours)
        window_data = df[(df["time"] >= start_time) & (df["time"] < end_time)]
        if len(window_data) > 2:
            sd_values.append(window_data["glucose"].std())
            mean_values.append(window_data["glucose"].mean())
    result = {
        "sd": np.mean(sd_values) if sd_values else 0.0,
        "mean": np.mean(mean_values) if mean_values else 0.0,
        "windows_analyzed": len(sd_values),
    }
    return result


def sd_interaction(
    glucose: pd.Series,
    timestamps: pd.Series,
) -> dict:
    """Calculates standard deviation of interaction (SDI).

    SDI quantifies daily variability in the glycemic pattern,
    considering interactions between time of day and specific day.

    Args:
        glucose: Glucose values.
        timestamps: Corresponding timestamps.

    Returns:
        Dictionary with SDI value and global mean.
    """
    df = pd.DataFrame({"time": timestamps, "glucose": glucose})
    df["hour"] = df["time"].dt.hour
    df["minute"] = df["time"].dt.minute
    df["day"] = df["time"].dt.date
    df["time_key"] = (
        df["hour"].astype(str).str.zfill(2) + ":" + df["minute"].astype(str).str.zfill(2)
    )
    global_mean = df["glucose"].mean()
    daily_means = df.groupby("day")["glucose"].mean()
    timepoint_means = df.groupby("time_key")["glucose"].mean()
    df_temp = df.copy()
    day_mean_df = pd.DataFrame(daily_means).reset_index()
    day_mean_df.columns = ["day", "day_mean"]
    timepoint_mean_df = pd.DataFrame(timepoint_means).reset_index()
    timepoint_mean_df.columns = ["time_key", "timepoint_mean"]
    df_temp = df_temp.merge(day_mean_df, on="day")
    df_temp = df_temp.merge(timepoint_mean_df, on="time_key")
    df_temp["expected"] = (
        global_mean
        + (df_temp["day_mean"] - global_mean)
        + (df_temp["timepoint_mean"] - global_mean)
    )
    df_temp["interaction"] = df_temp["glucose"] - df_temp["expected"]
    sdi = df_temp["interaction"].std()
    return {"sd": sdi, "mean": global_mean}


def cv_global(glucose: pd.Series) -> float:
    """Calculates total coefficient of variation (CVT).

    Args:
        glucose: Glucose values.

    Returns:
        CV as percentage (sd / mean * 100), 0 if mean is 0.
    """
    return cv_from_sd_mean(sd_global(glucose), mean_global(glucose))


def cv_from_sd_mean(sd_val: float, mean_val: float) -> float:
    """Calculates CV from SD and mean values.

    Args:
        sd_val: Standard deviation.
        mean_val: Mean value.

    Returns:
        CV as percentage (sd / mean * 100), 0 if mean is 0.
    """
    if mean_val == 0:
        return 0.0
    return sd_val / mean_val * 100
