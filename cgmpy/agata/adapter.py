"""Adapter to prepare cgmpy data for the py_agata library."""

from __future__ import annotations

import pandas as pd

from ..data.core import ModularGlucoseData
from ..errors import EmptyDataError


def prepare_data_for_agata(
    glucose_data: ModularGlucoseData, resample_freq: str = "5min"
) -> pd.DataFrame:
    """Prepare a :class:`~cgmpy.data.core.ModularGlucoseData` for py_agata.

    The function aligns the timestamp grid by flooring every measurement to
    ``resample_freq`` (default ``"5min"``), then re-emits the data on a
    homogeneous, gap-aware time grid that py_agata can consume.

    Args:
        glucose_data: The cgmpy data object to adapt.
        resample_freq: Pandas frequency string for the target grid
            (e.g. ``"5min"``, ``"15min"``). Must be a pandas-recognised
            frequency.

    Returns:
        pd.DataFrame: A two-column DataFrame with columns:

            * ``t`` (datetime64[ns]) -- the regular time index.
            * ``glucose`` (float, mg/dL) -- the glucose values, with NaN for
              timestamps that had no original measurement.

    Raises:
        EmptyDataError: If the input is empty, if the timestamp
            normalization drops every row, if the cleaned data covers a
            zero-duration interval, or if every glucose value on the
            regular grid is NaN.

    Notes:
        The function does **not** convert mmol/L to mg/dL; the input is
        assumed to already be in mg/dL. Unit conversion is a deferred
        feature targeted at v0.9.0 (see ``ROADMAP.md``).
    """
    df = glucose_data.data.copy()
    date_col = glucose_data.date_col
    glucose_col = glucose_data.glucose_col

    # Guard 1: nothing to do if the input has zero rows.
    if len(df) == 0:
        raise EmptyDataError(context="input ModularGlucoseData")

    # Ensure the date column is of datetime type
    df[date_col] = pd.to_datetime(df[date_col])

    # Sort the data chronologically first
    df_clean = df.sort_values(date_col)

    # -- KEY STEP: STANDARDIZE THE TIME COLUMN --
    # Round each time down to the nearest 5-minute interval.
    df_clean[date_col] = df_clean[date_col].dt.floor(resample_freq)

    # Now, remove duplicates. If 00:01 and 00:03 rounded to 00:00,
    # we keep the first one that appeared.
    df_clean = df_clean.drop_duplicates(subset=[date_col], keep="first")

    # Guard 2: the timestamp normalization step may have dropped every row
    # (e.g. all timestamps were NaT and floor() left them as NaT, or
    # duplicate removal wiped the dataset).
    if df_clean.empty:
        raise EmptyDataError(context="after timestamp normalization")

    # The rest of the process already works perfectly
    start_time = df_clean[date_col].min()
    end_time = df_clean[date_col].max()

    # Guard 3: a zero-duration interval (start == end is allowed and yields
    # a single-point grid, but a NaT boundary or otherwise invalid range
    # must be surfaced as EmptyDataError, not a pandas ValueError).
    if pd.isna(start_time) or pd.isna(end_time):
        raise EmptyDataError(context="the cleaned data covers a zero-duration interval")
    time_range = pd.date_range(start=start_time, end=end_time, freq=resample_freq)
    if len(time_range) == 0:
        raise EmptyDataError(context="the cleaned data covers a zero-duration interval")

    df_homogeneous = pd.DataFrame({date_col: time_range})

    # Merge with the already standardized data
    df_final = df_homogeneous.merge(df_clean, on=date_col, how="left")

    # Rename columns
    df_final = df_final.rename(columns={date_col: "t", glucose_col: "glucose"})

    # Guard 4: py_agata cannot work with a fully empty glucose series.
    # Reaching this point with all-NaN glucose means every original sample
    # was dropped by the merge (e.g. they fell outside the resample grid,
    # or the original glucose column was empty after processing).
    if df_final["glucose"].isna().all():
        raise EmptyDataError(
            context="all glucose values became NaN after merging to the regular grid"
        )

    return df_final[["t", "glucose"]]
