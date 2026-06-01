"""Adapter to prepare cgmpy data for the py_agata library."""

import pandas as pd

from ..data.core import ModularGlucoseData


def prepare_data_for_agata(
    glucose_data: ModularGlucoseData, resample_freq: str = "5min"
) -> pd.DataFrame:
    """
    Prepares the data of a cgmpy object to be analyzed by py_agata,
    handling unaligned start times.
    """
    df = glucose_data.data.copy()
    date_col = glucose_data.date_col
    glucose_col = glucose_data.glucose_col

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

    # The rest of the process already works perfectly
    start_time = df_clean[date_col].min()
    end_time = df_clean[date_col].max()
    time_range = pd.date_range(start=start_time, end=end_time, freq=resample_freq)
    df_homogeneous = pd.DataFrame({date_col: time_range})

    # Merge with the already standardized data
    df_final = df_homogeneous.merge(df_clean, on=date_col, how="left")

    # Rename columns
    df_final = df_final.rename(columns={date_col: "t", glucose_col: "glucose"})

    return df_final[["t", "glucose"]]
