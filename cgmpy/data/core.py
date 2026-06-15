"""
Main module for modular glucose data management.
"""

from __future__ import annotations

import copy
import datetime
import logging
import time
from typing import Any

import pandas as pd

from ..metrics.targets import GlucoseTargets, get_targets
from ..metrics.units import GlucoseUnit, convert
from .analyzer import DataAnalyzer
from .exporter import DataExporter
from .loader import DataLoader
from .processor import DataProcessor


class GlucoseData:
    """
    Refactored main class for modular glucose data management.

    This class integrates all specialized modules:
    - DataLoader: Data loading from different sources
    - DataProcessor: Data processing and validation
    - DataAnalyzer: Basic data analysis
    - DataExporter: Data export to different formats
    """

    @classmethod
    def from_sources(
        cls,
        sources: list[str | pd.DataFrame | GlucoseData],
        date_col: str = "time",
        glucose_col: str = "glucose",
        delimiter: str | None = None,
        header: int = 0,
        log: bool = False,
        target_type: str = "diabetes",
        unit: str | GlucoseUnit = GlucoseUnit.MG_DL,
        **kwargs,
    ) -> GlucoseData:
        """
        Creates an instance by merging multiple data sources.

        This method handles loading, column renaming, and initial merging.
        The constructor will then handle duplicate removal and sorting.

        Args:
            sources: List of sources (paths, DataFrames, or GlucoseData instances)
            date_col: Default date column name for new sources
            glucose_col: Default glucose column name for new sources
            delimiter: Delimiter for CSV files
            header: Header row for CSV files
            log: Whether to enable logging
            target_type: Type of glucose targets ('diabetes' or 'pregnancy')
            unit: Glucose unit of the data ('mg/dL' or 'mmol/L')
            kwargs: Additional arguments for the constructor

        Returns:
            A new GlucoseData instance containing all merged data
        """
        temp_loader = DataLoader()
        temp_processor = DataProcessor()
        all_dfs = []

        for source in sources:
            if isinstance(source, GlucoseData):
                # If it's another instance, we take its already processed data
                df = source.get_raw_data()
                # These are already named 'time' and 'glucose'
                all_dfs.append(df)

            elif isinstance(source, str | pd.DataFrame):
                # Load the source (if string) or take the DataFrame
                raw_df = temp_loader.load_from_source(
                    source, date_col, glucose_col, delimiter, header
                )
                # Standardize column names before concatenation
                standard_df = temp_processor.rename_columns(raw_df, date_col, glucose_col)
                all_dfs.append(standard_df)

            else:
                raise TypeError(f"Unsupported source type: {type(source)}")

        if not all_dfs:
            raise ValueError("No valid sources provided.")

        # Concatenate all standardized DataFrames
        combined_df = pd.concat(all_dfs, ignore_index=True)

        # Create the final instance.
        # Note: We pass date_col="time" and glucose_col="glucose" because
        # we've already standardized them above.
        return cls(
            combined_df,
            date_col="time",
            glucose_col="glucose",
            log=log,
            target_type=target_type,
            unit=unit,
            **kwargs,
        )

    def __init__(
        self,
        data_source: str | pd.DataFrame,
        date_col: str = "time",
        glucose_col: str = "glucose",
        delimiter: str | None = None,
        header: int = 0,
        start_date: str | datetime.datetime | None = None,
        end_date: str | datetime.datetime | None = None,
        log: bool = False,
        target_type: str = "diabetes",
        unit: str | GlucoseUnit = GlucoseUnit.MG_DL,
        regularize: bool = False,
        regularize_interval: int = 5,
        regularize_max_gap: int = 30,
    ):
        """
        Initializes glucose data with modular architecture.

        Args:
            data_source: CSV/Parquet file or DataFrame with data
            date_col: Name of the date/time column
            glucose_col: Name of the glucose values column
            delimiter: Delimiter for CSV files
            header: Header row for CSV files
            start_date: Start date for filtering data (optional)
            end_date: End date for filtering data (optional)
            log: If True, enables detailed performance logs
            target_type: Type of glucose targets ('diabetes' or 'pregnancy')
            unit: Glucose unit of the data ('mg/dL' or 'mmol/L')
        """
        # Logging configuration
        self.log = log
        self.logger = self._setup_logger()

        # Store parameters
        self.date_col = date_col
        self.glucose_col = glucose_col
        self.target_type = target_type
        if isinstance(unit, str):
            self._unit = GlucoseUnit(unit)
        elif isinstance(unit, GlucoseUnit):
            self._unit = unit
        else:
            self._unit = GlucoseUnit.MG_DL

        # Initialize targets
        self.targets: GlucoseTargets
        if isinstance(target_type, GlucoseTargets):
            self.targets = target_type
        else:
            self.targets = get_targets(target_type)

        # Initialize modules
        self.loader = DataLoader(self.logger)
        self.processor = DataProcessor(self.logger)
        self.analyzer = DataAnalyzer(self.logger)
        self.exporter = DataExporter(self.logger)

        # Process data
        self.data, self.time_diffs, self.typical_interval = self._initialize_data(
            data_source,
            date_col,
            glucose_col,
            delimiter,
            header,
            start_date,
            end_date,
            regularize,
            regularize_interval,
            regularize_max_gap,
        )

        # Dictionary to store operation logs
        self.logs: dict[str, Any] = {}

    @property
    def glucose(self) -> pd.Series:
        """Glucose values as a pandas Series.

        Convenience accessor for ``self.data["glucose"]``.

        Returns:
            pd.Series: Glucose values in mg/dL.
        """
        return self.data["glucose"]

    @property
    def timestamps(self) -> pd.Series:
        """Timestamps as a pandas Series.

        Convenience accessor for ``self.data["time"]``.

        Returns:
            pd.Series: Timestamps of glucose readings.
        """
        return self.data["time"]

    @property
    def unit(self) -> GlucoseUnit:
        """Glucose unit of the data.

        Returns:
            GlucoseUnit: The unit of the glucose values.
        """
        return self._unit

    @unit.setter
    def unit(self, value: GlucoseUnit | str):
        if isinstance(value, str):
            self._unit = GlucoseUnit(value)
        elif isinstance(value, GlucoseUnit):
            self._unit = value

    def glucose_in_unit(self, unit: GlucoseUnit | str = GlucoseUnit.MG_DL) -> pd.Series:
        """Return glucose values converted to the specified unit.

        Args:
            unit: Target unit.

        Returns:
            Glucose values in the requested unit.
        """
        target = GlucoseUnit(unit) if isinstance(unit, str) else unit
        if target == self._unit:
            return self.glucose
        return convert(self.glucose, self._unit, target)

    def _setup_logger(self) -> logging.Logger:
        """
        Configures the logger for the class.

        Returns:
            Configured logger
        """
        logger = logging.getLogger(__name__)

        if self.log and not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)

        return logger

    def _initialize_data(
        self,
        data_source: str | pd.DataFrame,
        date_col: str,
        glucose_col: str,
        delimiter: str | None,
        header: int,
        start_date: str | datetime.datetime | None,
        end_date: str | datetime.datetime | None,
        regularize: bool = False,
        regularize_interval: int = 5,
        regularize_max_gap: int = 30,
    ) -> tuple[pd.DataFrame, pd.Series, float]:
        """
        Initializes and processes data using modules.

        Returns:
            Tuple with (data, time_diffs, typical_interval)
        """
        t_start = time.time()

        # Step 1: Load data
        t0 = time.time()
        raw_data = self.loader.load_from_source(
            data_source, date_col, glucose_col, delimiter, header
        )
        t1 = time.time()

        # Step 2: Rename columns
        t2 = time.time()
        data = self.processor.rename_columns(raw_data, date_col, glucose_col)
        # After renaming, the columns are always "time" and "glucose".
        # Update the instance attributes so downstream consumers (e.g.
        # the AGATA adapter) reference the correct column names.
        self.date_col = "time"
        self.glucose_col = "glucose"
        t3 = time.time()

        # Step 3: Filter by dates
        t4 = time.time()
        if start_date is not None or end_date is not None:
            data = self.processor.filter_by_dates(data, start_date, end_date)
        t5 = time.time()

        # Step 4: Process data
        t6 = time.time()
        processed_data, time_diffs = self.processor.process_data(data, "time", "glucose", self.log)
        t7 = time.time()

        # Step 5: Calculate typical interval
        t8 = time.time()
        typical_interval = self.analyzer.calculate_typical_interval(time_diffs, self.log)
        t9 = time.time()

        # Step 6: Optional Regularization
        if regularize:
            t10 = time.time()
            processed_data = self.processor.regularize(
                processed_data,
                interval_mins=regularize_interval,
                max_gap_mins=regularize_max_gap,
                date_col="time",
                glucose_col="glucose",
            )
            time_diffs = processed_data["time"].diff()
            typical_interval = self.analyzer.calculate_typical_interval(time_diffs, self.log)
            t11 = time.time()
            if self.log:
                self.logger.debug(f"Regularization applied: {t11 - t10:.3f}s")

        t_end = time.time()

        # Performance logging
        if self.log:
            self.logger.debug(f"""
            Modular initialization timing:
            Data loading: {t1 - t0:.3f}s
            Column renaming: {t3 - t2:.3f}s
            Date filtering: {t5 - t4:.3f}s
            Data processing: {t7 - t6:.3f}s
            Typical interval calculation: {t9 - t8:.3f}s
            Total time: {t_end - t_start:.3f}s
            """)

        if self.log:
            self.logger.info(f"Data source loaded and processed in {t_end - t_start:.3f} seconds.")

        return processed_data, time_diffs, typical_interval

    def get_typical_interval(self) -> float:
        """
        Returns the typical interval between measurements in minutes.

        Returns:
            Typical interval in minutes
        """
        return self.typical_interval

    def info(self, include_disconnections: bool = False) -> dict[str, Any]:
        """
        Shows basic file information.

        Args:
            include_disconnections: Whether to include disconnection details

        Returns:
            Dictionary with basic information
        """
        return self.analyzer.get_basic_info(
            self.data, self.time_diffs, self.typical_interval, include_disconnections
        )

    def __str__(self) -> str:
        """
        String representation of the object with basic information.

        Returns:
            String with information summary
        """
        info = self.info()
        return self.analyzer.get_summary_string(info)

    def get_data_quality_metrics(self) -> dict[str, Any]:
        """
        Calculates data quality metrics.

        Returns:
            Dictionary with quality metrics
        """
        return self.analyzer.get_data_quality_metrics(
            self.data, self.time_diffs, self.typical_interval
        )

    def get_logs(self) -> dict[str, Any]:
        """
        Returns all stored logs.

        Returns:
            Dictionary with generated logs
        """
        if not self.log:
            self.logger.warning(
                "Logging is not enabled. Initialize the class with log=True to generate logs."
            )
            return {}
        return self.logs

    # Export methods - Delegate to DataExporter
    def to_parquet(self, file_path: str, compression: str = "snappy", sort: bool = True):
        """Delegates to DataExporter to save in Parquet format."""
        self.exporter.to_parquet(self.data, file_path, compression, sort)

    def append_to_parquet(
        self,
        file_path: str,
        compression: str = "snappy",
        handle_duplicates: str = "keep_new",
    ) -> int:
        """Delegates to DataExporter to append to an existing Parquet file."""
        return self.exporter.append_to_parquet(self.data, file_path, compression, handle_duplicates)

    def to_csv(self, file_path: str, separator: str = ",", include_index: bool = False):
        """Delegates to DataExporter to save in CSV format."""
        self.exporter.to_csv(self.data, file_path, separator, include_index)

    def to_excel(self, file_path: str, sheet_name: str = "glucose_data"):
        """Delegates to DataExporter to save in Excel format."""
        self.exporter.to_excel(self.data, file_path, sheet_name)

    # Data access methods
    def get_raw_data(self) -> pd.DataFrame:
        """
        Returns the DataFrame with processed data.

        Returns:
            DataFrame with glucose data
        """
        return self.data.copy()

    def get_glucose_values(self) -> pd.Series:
        """
        Returns only glucose values.

        Returns:
            Series with glucose values
        """
        return self.data["glucose"].copy()

    def get_timestamps(self) -> pd.Series:
        """
        Returns only timestamps.

        Returns:
            Series with timestamps
        """
        return self.data["time"].copy()

    def get_time_differences(self) -> pd.Series:
        """
        Returns time differences between measurements.

        Returns:
            Series with time differences
        """
        return self.time_diffs.copy()

    # Filtering methods
    def filter_by_date_range(
        self,
        start_date: str | datetime.datetime,
        end_date: str | datetime.datetime,
    ) -> GlucoseData:
        """
        Creates a new instance filtered by date range.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            New instance with filtered data
        """
        # Use the processor to filter data
        filtered_data = self.processor.filter_by_dates(self.data, start_date, end_date)

        # Create new instance using the constructor
        return self._create_filtered_instance(filtered_data)

    def filter_by_glucose_range(self, min_glucose: float, max_glucose: float) -> GlucoseData:
        """
        Creates a new instance filtered by glucose range.

        Args:
            min_glucose: Minimum glucose value
            max_glucose: Maximum glucose value

        Returns:
            New instance with filtered data
        """
        mask = (self.data["glucose"] >= min_glucose) & (self.data["glucose"] <= max_glucose)
        filtered_data = self.data[mask].copy()

        if len(filtered_data) == 0:
            raise ValueError("No data in the specified glucose range.")

        return self._create_filtered_instance(filtered_data)

    def regularize(self, interval_mins: int = 5, max_gap_mins: int = 30) -> GlucoseData:
        """
        Regularizes the current instance's data in-place.
        Returns self for method chaining.

        Args:
            interval_mins: Target interval in minutes.
            max_gap_mins: Maximum gap to interpolate.

        Returns:
            GlucoseData: The current instance with uniform frequency.
        """
        # 1. Use processor to regularize data
        self.data = self.processor.regularize(
            self.data,
            interval_mins=interval_mins,
            max_gap_mins=max_gap_mins,
            date_col="time",
            glucose_col="glucose",
        )

        # 2. Update time differences and internal metrics
        self.time_diffs = self.data["time"].diff()
        self.typical_interval = self.analyzer.calculate_typical_interval(self.time_diffs)

        # 3. Refresh trimesters if specialized for pregnancy
        if hasattr(self, "_split_trimesters"):
            self.trimesters = self._split_trimesters()

        if self.log:
            self.logger.info(f"Data regularized in-place to {interval_mins}min.")

        return self

    def _create_filtered_instance(self, filtered_data: pd.DataFrame) -> GlucoseData:
        """
        Creates a new instance (clone) with modified data.

        Args:
            filtered_data: DataFrame with filtered or modified data

        Returns:
            New GlucoseData instance
        """
        new_instance = copy.copy(self)
        new_instance.logs = {}

        # Refresh dependent data
        new_instance.data = filtered_data.copy()
        new_instance.time_diffs = new_instance.data["time"].diff()
        new_instance.typical_interval = self.analyzer.calculate_typical_interval(
            new_instance.time_diffs
        )

        # Refresh trimesters if PregnancyData
        if hasattr(self, "_split_trimesters"):
            new_instance.trimesters = new_instance._split_trimesters()  # type: ignore[attr-defined]

        return new_instance
