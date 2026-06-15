"""
Module for loading glucose data from different sources.

Errors raised by this module are CGMPy-specific subclasses of
:class:`ValueError` (see :mod:`cgmpy.errors`), so legacy callers that
catch :class:`ValueError` continue to work.
"""

import logging
from pathlib import Path

import pandas as pd

from ..errors import InvalidCSVFormatError


class DataLoader:
    """
    Class responsible for loading data from different sources (CSV, Parquet, DataFrame).
    """

    def __init__(self, logger: logging.Logger | None = None):
        """
        Initializes the DataLoader.

        Args:
            logger: Logger to record operations
        """
        self.logger = logger or logging.getLogger(__name__)

    def load_from_source(
        self,
        data_source: str | pd.DataFrame,
        date_col: str,
        glucose_col: str,
        delimiter: str | None = None,
        header: int = 0,
    ) -> pd.DataFrame:
        """
        Loads data from different sources.

        Args:
            data_source: CSV/Parquet file or DataFrame
            date_col: Name of the date column
            glucose_col: Name of the glucose column
            delimiter: Delimiter for CSV files
            header: Header row

        Returns:
            DataFrame with loaded data
        """
        if isinstance(data_source, str):
            return self._load_from_file(data_source, date_col, glucose_col, delimiter, header)
        elif isinstance(data_source, pd.DataFrame):
            return self._load_from_dataframe(data_source)
        else:
            raise ValueError("data_source must be a CSV file, Parquet file, or a DataFrame")

    def _load_from_file(
        self,
        file_path: str,
        date_col: str,
        glucose_col: str,
        delimiter: str | None,
        header: int,
    ) -> pd.DataFrame:
        """
        Loads data from a file.

        Args:
            file_path: Path to the file
            date_col: Name of the date column
            glucose_col: Name of the glucose column
            delimiter: Delimiter for CSV
            header: Header row

        Returns:
            DataFrame with the data
        """
        if not Path(file_path).is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        is_parquet = file_path.lower().endswith(".parquet")

        if is_parquet:
            return self._load_parquet(file_path, date_col, glucose_col)
        else:
            return self._load_csv(file_path, date_col, glucose_col, delimiter, header)

    def _load_parquet(self, file_path: str, date_col: str, glucose_col: str) -> pd.DataFrame:
        """
        Loads data from a Parquet file.

        Args:
            file_path: Path to the Parquet file
            date_col: Name of the date column
            glucose_col: Name of the glucose column

        Returns:
            DataFrame with the data

        Raises:
            InvalidCSVFormatError: If pyarrow/pandas fails to read the
                file (corrupt, missing columns, schema mismatch, ...).
        """
        try:
            return pd.read_parquet(file_path, columns=[date_col, glucose_col])
        except Exception as e:
            raise InvalidCSVFormatError(file_path, reason=str(e)) from e

    def _load_csv(
        self,
        file_path: str,
        date_col: str,
        glucose_col: str,
        delimiter: str | None,
        header: int,
    ) -> pd.DataFrame:
        """
        Loads data from a CSV file.

        Args:
            file_path: Path to the CSV file
            date_col: Name of the date column
            glucose_col: Name of the glucose column
            delimiter: Delimiter
            header: Header row

        Returns:
            DataFrame with the data

        Raises:
            InvalidCSVFormatError: If the file cannot be parsed by pandas
                using either ``,`` or ``;`` as the delimiter.
        """
        if delimiter is None:
            delimiter = ","

        try:
            return pd.read_csv(
                file_path,
                delimiter=delimiter,
                header=header,
                usecols=[date_col, glucose_col],
            )
        except Exception as e:
            # Try with alternative delimiter if it fails
            if delimiter == ",":
                try:
                    return pd.read_csv(
                        file_path,
                        delimiter=";",
                        header=header,
                        usecols=[date_col, glucose_col],
                    )
                except Exception as inner_e:
                    raise InvalidCSVFormatError(
                        file_path,
                        reason=str(inner_e),
                        hint=("Try specifying the delimiter with the 'delimiter' parameter."),
                    ) from inner_e
            else:
                raise InvalidCSVFormatError(
                    file_path,
                    reason=(f"Error reading CSV file with delimiter '{delimiter}': {e!s}"),
                ) from e

    def _load_from_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Loads data from an existing DataFrame.

        Args:
            dataframe: DataFrame with data

        Returns:
            Copy of the DataFrame
        """
        return dataframe.copy()
