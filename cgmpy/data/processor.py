"""
Module for glucose data processing and validation.
"""

import logging
import time

import numpy as np
import pandas as pd


class DataProcessor:
    """
    Class responsible for glucose data processing, validation, and cleaning.
    """

    def __init__(self, logger: logging.Logger | None = None):
        """
        Initializes the DataProcessor.

        :param logger: Logger to record operations
        """
        self.logger = logger or logging.getLogger(__name__)

    def process_data(
        self,
        data: pd.DataFrame,
        date_col: str,
        glucose_col: str,
        log_performance: bool = False,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Processes glucose data in an optimized way.

        :param data: DataFrame with data
        :param date_col: Name of the date column
        :param glucose_col: Name of the glucose column
        :param log_performance: If True, records performance metrics
        :return: Tuple with processed DataFrame and time differences Series
        """
        t_start = time.time()

        if log_performance:
            self.logger.debug("\n--- DETAILED PERFORMANCE ANALYSIS ---")

        # Validate columns
        self._validate_columns(data, date_col, glucose_col)

        # Determine if data comes from optimized Parquet
        from_parquet = self._is_optimized_parquet(data, date_col, glucose_col)

        if from_parquet:
            processed_data, time_diffs = self._process_parquet_optimized(
                data, date_col, glucose_col, log_performance
            )
        else:
            processed_data, time_diffs = self._process_standard(
                data, date_col, glucose_col, log_performance
            )

        # Final validation
        if not pd.api.types.is_datetime64_any_dtype(processed_data[date_col]):
            raise ValueError("Error in date conversion")

        if log_performance:
            t_end = time.time()
            memoria_bytes = processed_data.memory_usage(deep=True).sum()
            memoria_mb = memoria_bytes / (1024 * 1024)
            self.logger.debug(f"DataFrame memory usage: {memoria_mb:.2f} MB")
            self.logger.debug(f"Total processing time: {t_end - t_start:.3f}s")
            self.logger.debug("--- END OF ANALYSIS ---\n")

        return processed_data, time_diffs

    def _validate_columns(self, data: pd.DataFrame, date_col: str, glucose_col: str):
        """
        Validates that the specified columns exist in the DataFrame.

        :param data: DataFrame to validate
        :param date_col: Name of the date column
        :param glucose_col: Name of the glucose column
        """
        if date_col not in data.columns or glucose_col not in data.columns:
            raise ValueError(
                f"Columns '{date_col}' or '{glucose_col}' not found in the DataFrame. "
                f"Available columns: {data.columns.tolist()}."
            )

    def _is_optimized_parquet(self, data: pd.DataFrame, date_col: str, glucose_col: str) -> bool:
        """
        Determines if the data comes from an optimized Parquet file.

        :param data: DataFrame with data
        :param date_col: Name of the date column
        :param glucose_col: Name of the glucose column
        :return: True if it is optimized Parquet
        """
        return (
            pd.api.types.is_datetime64_any_dtype(data[date_col])
            and data[glucose_col].dtype == "int16"
        )

    def _process_parquet_optimized(
        self,
        data: pd.DataFrame,
        date_col: str,
        glucose_col: str,
        log_performance: bool = False,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Processes optimized Parquet data with a fast path.

        :param data: DataFrame with data
        :param date_col: Name of the date column
        :param glucose_col: Name of the glucose column
        :param log_performance: If True, records performance metrics
        :return: Tuple with processed DataFrame and time differences Series
        """
        if log_performance:
            self.logger.debug("Detected Parquet source data with optimized types.")
            self.logger.debug("Applying fast path for Parquet data...")

        processed_data = data.copy()

        # Check and remove nulls
        t_nulos = time.time()
        if processed_data.isna().any().any():
            processed_data = processed_data.dropna(subset=[date_col, glucose_col])
            if log_performance:
                self.logger.debug(f"  - Removed null values: {time.time() - t_nulos:.3f}s")
        elif log_performance:
            self.logger.debug(f"  - No null values found: {time.time() - t_nulos:.3f}s")

        # Check and sort
        t_orden = time.time()
        if not processed_data[date_col].is_monotonic_increasing:
            if log_performance:
                self.logger.debug("  - Sorting data...")
            processed_data = processed_data.sort_values(date_col, ignore_index=True)
        elif log_performance:
            self.logger.debug("  - Data already sorted")

        if log_performance:
            self.logger.debug(f"  - Order verification: {time.time() - t_orden:.3f}s")

        # Optimized calculation of time differences
        t_diff = time.time()
        time_values = processed_data[date_col].values
        time_diffs_ns = np.diff(time_values.astype("datetime64[ns]"))
        time_diffs_ns = np.insert(time_diffs_ns, 0, np.timedelta64(0, "ns"))
        time_diffs = pd.Series(pd.TimedeltaIndex(time_diffs_ns), index=processed_data.index)

        if log_performance:
            self.logger.debug(f"  - Optimized difference calculation: {time.time() - t_diff:.3f}s")

        return processed_data, time_diffs

    def _process_standard(
        self,
        data: pd.DataFrame,
        date_col: str,
        glucose_col: str,
        log_performance: bool = False,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Processes data with full validations for CSV and other formats.

        :param data: DataFrame with data
        :param date_col: Name of the date column
        :param glucose_col: Name of the glucose column
        :param log_performance: If True, records performance metrics
        :return: Tuple with processed DataFrame and time differences Series
        """
        if log_performance:
            self.logger.debug("Processing data with type conversion and full validations.")

        processed_data = data.copy()

        # Handle nulls
        t_nulos = time.time()
        processed_data = self._handle_nulls(processed_data, date_col, glucose_col)
        if log_performance:
            self.logger.debug(f"2. Null removal: {time.time() - t_nulos:.3f}s")

        # Type conversion
        t_tipos = time.time()
        processed_data = self._convert_data_types(processed_data, date_col, glucose_col)
        if log_performance:
            self.logger.debug(f"3. Type conversion: {time.time() - t_tipos:.3f}s")

        # Handle duplicates
        t_dups = time.time()
        processed_data = self._handle_duplicates(processed_data, date_col, glucose_col)
        if log_performance:
            self.logger.debug(f"4. Duplicate processing: {time.time() - t_dups:.3f}s")

        # Sorting
        t_orden = time.time()
        processed_data = self._sort_data(processed_data, date_col)
        if log_performance:
            self.logger.debug(f"5. Sorting: {time.time() - t_orden:.3f}s")

        # Calculate differences
        t_diff = time.time()
        time_diffs = processed_data[date_col].diff()
        if log_performance:
            self.logger.debug(f"6. Difference calculation: {time.time() - t_diff:.3f}s")

        return processed_data, time_diffs

    def _handle_nulls(self, data: pd.DataFrame, date_col: str, glucose_col: str) -> pd.DataFrame:
        """
        Removes rows with null values in key columns.

        :param data: DataFrame with data
        :param date_col: Name of the date column
        :param glucose_col: Name of the glucose column
        :return: DataFrame without nulls
        """
        filas_antes = len(data)
        data_cleaned = data.dropna(subset=[date_col, glucose_col])
        filas_despues = len(data_cleaned)

        if filas_antes > filas_despues:
            self.logger.debug(f"  - Removed {filas_antes - filas_despues} rows with null values.")

        return data_cleaned

    def _convert_data_types(
        self, data: pd.DataFrame, date_col: str, glucose_col: str
    ) -> pd.DataFrame:
        """
        Convert date and glucose columns to correct types.

        :param data: DataFrame with data
        :param date_col: Name of the date column
        :param glucose_col: Name of the glucose column
        :return: DataFrame with correct types
        """
        data_converted = data.copy()

        # Convert date column
        if not pd.api.types.is_datetime64_any_dtype(data_converted[date_col]):
            self.logger.debug(f"  - Converting column '{date_col}' to datetime...")
            if pd.api.types.is_numeric_dtype(data_converted[date_col]):
                unit = "ms" if data_converted[date_col].iloc[0] > 1e10 else "s"
                data_converted[date_col] = pd.to_datetime(data_converted[date_col], unit=unit)
            else:
                data_converted[date_col] = pd.to_datetime(
                    data_converted[date_col], errors="coerce", format="mixed"
                )

        # Convert glucose column
        if not pd.api.types.is_numeric_dtype(data_converted[glucose_col]):
            self.logger.debug(f"  - Converting column '{glucose_col}' to numeric...")

            # Si es tipo objeto (string), manejamos posibles comas decimales
            if data_converted[glucose_col].dtype == "object":
                # Reemplazamos comas por puntos para soportar formatos europeos
                data_converted[glucose_col] = (
                    data_converted[glucose_col].astype(str).str.replace(",", ".", regex=False)
                )

            data_converted[glucose_col] = pd.to_numeric(
                data_converted[glucose_col], errors="coerce"
            )
            data_converted = data_converted.dropna(subset=[glucose_col])

        # Optimize glucose type to int16 if possible
        if data_converted[glucose_col].dtype != "int16":
            self.logger.debug(f"  - Optimizing column '{glucose_col}'...")
            min_val, max_val = (
                data_converted[glucose_col].min(),
                data_converted[glucose_col].max(),
            )
            if pd.notna(min_val) and pd.notna(max_val) and min_val >= -32768 and max_val <= 32767:
                # Only convert to int16 if values are already integers to avoid precision loss
                is_integer = np.all(np.mod(data_converted[glucose_col].dropna(), 1) == 0)
                if is_integer:
                    data_converted[glucose_col] = data_converted[glucose_col].astype("int16")
                else:
                    data_converted[glucose_col] = data_converted[glucose_col].astype("float32")
            else:
                data_converted[glucose_col] = pd.to_numeric(
                    data_converted[glucose_col], errors="coerce", downcast="float"
                )

        return data_converted

    def _handle_duplicates(
        self, data: pd.DataFrame, date_col: str, glucose_col: str
    ) -> pd.DataFrame:
        """
        Finds and resolves duplicates in the date column.

        :param data: DataFrame with data
        :param date_col: Name of the date column
        :param glucose_col: Name of the glucose column
        :return: DataFrame without duplicates
        """
        mask_duplicados = data.duplicated(subset=[date_col], keep=False)
        num_duplicados = mask_duplicados.sum()

        if num_duplicados > 0:
            self.logger.debug(f"  - Found {num_duplicados // 2} duplicate timestamps. Resolving...")

            df_dups = data[mask_duplicados].copy()
            df_dups["diff"] = df_dups.groupby(date_col)[glucose_col].transform(
                lambda x: (x - x.mean()).abs()
            )

            idx_to_keep = df_dups.groupby(date_col)["diff"].idxmin()

            return pd.concat(
                [data[~mask_duplicados], df_dups.loc[idx_to_keep].drop(columns="diff")]
            )

        return data

    def _sort_data(self, data: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """
        Sorts the DataFrame by the date column if it is not already sorted.

        :param data: DataFrame with data
        :param date_col: Name of the date column
        :return: Sorted DataFrame
        """
        if not data[date_col].is_monotonic_increasing:
            self.logger.debug("  - Sorting data by timestamp...")
            return data.sort_values(date_col, ignore_index=True)

        self.logger.debug("  - Data already sorted, skipping sorting.")
        return data

    def rename_columns(self, data: pd.DataFrame, date_col: str, glucose_col: str) -> pd.DataFrame:
        """
        Renames columns to standard names.

        :param data: DataFrame with data
        :param date_col: Current name of the date column
        :param glucose_col: Current name of the glucose column
        :return: DataFrame with renamed columns
        """
        renamed_data = data.copy()

        if date_col != "time":
            renamed_data = renamed_data.rename(columns={date_col: "time"})
        if glucose_col != "glucose":
            renamed_data = renamed_data.rename(columns={glucose_col: "glucose"})

        return renamed_data

    def filter_by_dates(
        self,
        data: pd.DataFrame,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        date_col: str = "time",
    ) -> pd.DataFrame:
        """
        Filters data by date range.

        :param data: DataFrame with data
        :param start_date: Start date
        :param end_date: End date
        :param date_col: Name of the date column
        :return: Filtered DataFrame
        """
        filtered_data = data.copy()

        if start_date is not None:
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            filtered_data = filtered_data[filtered_data[date_col] >= start_date]

        if end_date is not None:
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            filtered_data = filtered_data[filtered_data[date_col] <= end_date]

        if len(filtered_data) == 0:
            raise ValueError("No data available in the specified date range.")

        return filtered_data

    def regularize(
        self,
        data: pd.DataFrame,
        interval_mins: int = 5,
        max_gap_mins: int = 30,
        date_col: str = "time",
        glucose_col: str = "glucose",
    ) -> pd.DataFrame:
        """
        Regularizes glucose data to a constant sampling frequency.

        This method is critical when combining data from sensors with different
        frequencies (e.g., 1 min vs 15 min) to avoid statistical bias.

        Args:
            data: The original glucose DataFrame.
            interval_mins: Target interval in minutes (default 5).
            max_gap_mins: Maximum gap to interpolate (default 30).
                          Gaps larger than this are kept as NaNs to mark disconnections.
            date_col: Name of the time column.
            glucose_col: Name of the glucose column.

        Returns:
            pd.DataFrame: A new DataFrame with uniform frequency.
        """
        if len(data) < 2:
            return data.copy()

        # 1. Create a copy and set the index to the time column for resampling
        df = data.copy()
        df = df.set_index(date_col).sort_index()

        # 2. Resample to the target frequency
        freq = f"{interval_mins}min"
        # We use mean() for points that collapse into the same bin (if any)
        resampled = df[glucose_col].resample(freq).mean()

        # 3. Calculate the interpolation limit
        # If interval is 5min and max_gap is 30min, we can interpolate up to 6 points (30/5)
        limit = max_gap_mins // interval_mins

        # 4. Interpolate with limit
        # This fills small gaps but keeps large ones as NaN (disconnections)
        regularized_glucose = resampled.interpolate(
            method="linear", limit=limit, limit_area="inside"
        )

        # 5. Reconstruct the DataFrame
        regularized_df = pd.DataFrame(
            {date_col: regularized_glucose.index, glucose_col: regularized_glucose.values}
        )

        # 6. Drop NaNs created by resampling that weren't interpolated
        # (These are the actual long disconnections)
        regularized_df = regularized_df.dropna(subset=[glucose_col])

        # 7. Ensure correct types (glucose as int16 if appropriate)
        regularized_df = self._convert_data_types(regularized_df, date_col, glucose_col)

        self.logger.info(
            f"Data regularized to {interval_mins}min. Rows changed from {len(data)} to {len(regularized_df)}."
        )

        return regularized_df
