"""
Module for glucose data export.
"""

import logging
import time
from pathlib import Path

import pandas as pd


class DataExporter:
    """
    Class responsible for exporting glucose data in different formats.
    """

    def __init__(self, logger: logging.Logger | None = None):
        """
        Initializes the DataExporter.

        Args:
            logger: Logger to record operations
        """
        self.logger = logger or logging.getLogger(__name__)

    def to_parquet(
        self,
        data: pd.DataFrame,
        file_path: str,
        compression: str = "snappy",
        sort: bool = True,
    ):
        """
        Saves data in optimized Parquet format.

        Args:
            data: DataFrame with data
            file_path: Path where to save the file
            compression: Compression algorithm
            sort: Whether to sort data before saving
        """
        self.logger.info("Preparing data to save in optimized Parquet format...")

        # Prepare data for saving
        df_to_save = data.copy()

        # Sort by time if requested
        if sort and not df_to_save["time"].is_monotonic_increasing:
            self.logger.info("  - Sorting data by timestamp...")
            df_to_save = df_to_save.sort_values("time", ignore_index=True)

        # Optimize data types
        df_to_save = self._optimize_data_types(df_to_save)

        # Remove duplicates if they exist
        df_to_save = self._remove_duplicates(df_to_save)

        # Save in Parquet format
        t_start = time.time()
        df_to_save.to_parquet(file_path, compression=compression, index=False, engine="pyarrow")
        t_end = time.time()

        # Final information
        self._log_save_info(file_path, df_to_save, t_end - t_start)

    def append_to_parquet(
        self,
        data: pd.DataFrame,
        file_path: str,
        compression: str = "snappy",
        handle_duplicates: str = "keep_new",
    ) -> int:
        """
        Appends data to an existing Parquet file.

        Args:
            data: DataFrame with new data
            file_path: Path to the Parquet file
            compression: Compression algorithm
            handle_duplicates: Strategy for handling duplicates

        Returns:
            Number of records added
        """
        if not Path(file_path).exists():
            self.logger.info(f"File {file_path} does not exist. Creating new file...")
            self.to_parquet(data, file_path, compression=compression)
            return len(data)

        self.logger.info(f"Appending data to existing Parquet file: {file_path}")

        # Load existing data
        t_start = time.time()
        existing_data = pd.read_parquet(file_path)
        t_load = time.time()
        self.logger.info(f"  - Existing file loaded in {t_load - t_start:.3f}s")
        self.logger.info(f"  - Existing records: {len(existing_data):,}")

        # Prepare new data
        new_data = self._prepare_new_data(data)

        # Handle duplicates
        existing_data, new_data = self._handle_duplicates(
            existing_data, new_data, handle_duplicates
        )

        # Combine and sort data
        t_combine = time.time()
        final_data = pd.concat([existing_data, new_data])
        final_data = final_data.sort_values("time", ignore_index=True)
        t_sort = time.time()
        self.logger.info(f"  - Data combined and sorted in {t_sort - t_combine:.3f}s")

        # Save result
        final_data.to_parquet(file_path, compression=compression, index=False, engine="pyarrow")
        t_save = time.time()

        # Final information
        records_added = len(final_data) - len(existing_data)
        file_size = Path(file_path).stat().st_size / 1024 / 1024
        self.logger.info("Data added successfully:")
        self.logger.info(f"  - Records added: {records_added:,}")
        self.logger.info(f"  - Total records: {len(final_data):,}")
        self.logger.info(f"  - File size: {file_size:.2f} MB")
        self.logger.info(f"  - Total time: {t_save - t_start:.3f}s")

        return records_added

    def _optimize_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Optimizes data types for efficient storage.

        Args:
            data: DataFrame with data

        Returns:
            DataFrame with optimized types
        """
        df_optimized = data.copy()

        # Optimize glucose column
        if not pd.api.types.is_integer_dtype(df_optimized["glucose"]):
            self.logger.info("  - Converting 'glucose' to numeric format...")
            df_optimized["glucose"] = pd.to_numeric(df_optimized["glucose"], errors="coerce")

        # Try to convert to int16 if possible
        min_val = df_optimized["glucose"].min()
        max_val = df_optimized["glucose"].max()

        if pd.notna(min_val) and pd.notna(max_val) and min_val >= -32768 and max_val <= 32767:
            self.logger.info(
                f"  - Optimizing 'glucose' to int16 (range: {min_val} to {max_val})..."
            )
            df_optimized["glucose"] = df_optimized["glucose"].astype("int16")
        else:
            self.logger.warning(
                f"  - Glucose values outside int16 range ({min_val} to {max_val}). Using int32."
            )
            df_optimized["glucose"] = df_optimized["glucose"].astype("int32")

        # Verify that time is datetime
        if not pd.api.types.is_datetime64_any_dtype(df_optimized["time"]):
            self.logger.info("  - Converting 'time' to datetime...")
            df_optimized["time"] = pd.to_datetime(df_optimized["time"], errors="coerce")

        return df_optimized

    def _remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Removes duplicates based on the time column.

        Args:
            data: DataFrame with data

        Returns:
            DataFrame without duplicates
        """
        duplicates = data.duplicated(subset=["time"], keep="first")
        if duplicates.any():
            n_duplicates = duplicates.sum()
            self.logger.info(f"  - Removing {n_duplicates} duplicate timestamps...")
            return data.drop_duplicates(subset=["time"], keep="first")
        return data

    def _prepare_new_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares new data for insertion.

        Args:
            data: DataFrame with new data

        Returns:
            Prepared DataFrame
        """
        new_data = data.copy()

        # Ensure correct types
        if not pd.api.types.is_datetime64_any_dtype(new_data["time"]):
            new_data["time"] = pd.to_datetime(new_data["time"], errors="coerce")

        if not pd.api.types.is_integer_dtype(new_data["glucose"]):
            new_data["glucose"] = pd.to_numeric(new_data["glucose"], errors="coerce")

            # Convert to int16 if possible
            min_val = new_data["glucose"].min()
            max_val = new_data["glucose"].max()
            if pd.notna(min_val) and pd.notna(max_val) and min_val >= -32768 and max_val <= 32767:
                new_data["glucose"] = new_data["glucose"].astype("int16")

        return new_data

    def _handle_duplicates(
        self, existing_data: pd.DataFrame, new_data: pd.DataFrame, strategy: str
    ) -> tuple:
        """
        Handles duplicates between existing and new data.

        Args:
            existing_data: DataFrame with existing data
            new_data: DataFrame with new data
            strategy: Strategy for handling duplicates

        Returns:
            Tuple with processed DataFrames
        """
        # Identify duplicates
        combined = pd.concat([existing_data, new_data])
        duplicated_times = combined["time"].duplicated(keep=False)
        num_duplicates = duplicated_times.sum() // 2

        if num_duplicates > 0:
            self.logger.info(f"  - Found {num_duplicates} duplicate timestamps")

            if strategy == "keep_new":
                self.logger.info("  - Strategy: Keep new data in case of duplicates")
                existing_times = set(existing_data["time"])
                new_times = set(new_data["time"])
                common_times = existing_times.intersection(new_times)

                if common_times:
                    existing_data = existing_data[~existing_data["time"].isin(common_times)]

            elif strategy == "keep_old":
                self.logger.info("  - Strategy: Keep existing data in case of duplicates")
                existing_times = set(existing_data["time"])
                new_data = new_data[~new_data["time"].isin(existing_times)]

        return existing_data, new_data

    def _log_save_info(self, file_path: str, data: pd.DataFrame, save_time: float):
        """
        Logs information about the save operation.

        Args:
            file_path: Path of the saved file
            data: Saved DataFrame
            save_time: Save time
        """
        file_size = Path(file_path).stat().st_size / 1024 / 1024
        self.logger.info(f"Data saved in Parquet format at: {file_path}")
        self.logger.info(f"  - File size: {file_size:.2f} MB")
        self.logger.info(f"  - Save time: {save_time:.3f}s")
        self.logger.info(f"  - Records saved: {len(data):,}")
        self.logger.info(f"  - Date range: {data['time'].min()} to {data['time'].max()}")
        self.logger.info("  - Format ready for fast loading")

    def to_csv(
        self,
        data: pd.DataFrame,
        file_path: str,
        separator: str = ",",
        include_index: bool = False,
    ):
        """
        Saves data in CSV format.

        Args:
            data: DataFrame with data
            file_path: Path where to save the file
            separator: Field separator
            include_index: Whether to include the index
        """
        self.logger.info(f"Saving data in CSV format: {file_path}")

        t_start = time.time()
        data.to_csv(file_path, sep=separator, index=include_index)
        t_end = time.time()

        file_size = Path(file_path).stat().st_size / 1024 / 1024
        self.logger.info(f"Data saved in CSV format at: {file_path}")
        self.logger.info(f"  - File size: {file_size:.2f} MB")
        self.logger.info(f"  - Save time: {t_end - t_start:.3f}s")
        self.logger.info(f"  - Records saved: {len(data):,}")

    def to_excel(self, data: pd.DataFrame, file_path: str, sheet_name: str = "glucose_data"):
        """
        Saves data in Excel format.

        Args:
            data: DataFrame with data
            file_path: Path where to save the file
            sheet_name: Sheet name
        """
        self.logger.info(f"Saving data in Excel format: {file_path}")

        t_start = time.time()
        data.to_excel(file_path, sheet_name=sheet_name, index=False)
        t_end = time.time()

        file_size = Path(file_path).stat().st_size / 1024 / 1024
        self.logger.info(f"Data saved in Excel format at: {file_path}")
        self.logger.info(f"  - File size: {file_size:.2f} MB")
        self.logger.info(f"  - Save time: {t_end - t_start:.3f}s")
        self.logger.info(f"  - Records saved: {len(data):,}")
