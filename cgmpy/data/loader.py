"""
Module for loading glucose data from different sources.
"""

import logging
from pathlib import Path

import pandas as pd


class DataLoader:
    """
    Class responsible for loading data from different sources (CSV, Parquet, DataFrame).
    """

    def __init__(self, logger: logging.Logger | None = None):
        """
        Initializes the DataLoader.

        :param logger: Logger to record operations
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

        :param data_source: CSV/Parquet file or DataFrame
        :param date_col: Name of the date column
        :param glucose_col: Name of the glucose column
        :param delimiter: Delimiter for CSV files
        :param header: Header row
        :return: DataFrame with loaded data
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

        :param file_path: Path to the file
        :param date_col: Name of the date column
        :param glucose_col: Name of the glucose column
        :param delimiter: Delimiter for CSV
        :param header: Header row
        :return: DataFrame with the data
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

        :param file_path: Path to the Parquet file
        :param date_col: Name of the date column
        :param glucose_col: Name of the glucose column
        :return: DataFrame with the data
        """
        try:
            return pd.read_parquet(file_path, columns=[date_col, glucose_col])
        except Exception as e:
            raise ValueError(f"Error reading Parquet file: {e!s}") from e

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

        :param file_path: Path to the CSV file
        :param date_col: Name of the date column
        :param glucose_col: Name of the glucose column
        :param delimiter: Delimiter
        :param header: Header row
        :return: DataFrame with the data
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
                    raise ValueError(
                        f"Error reading CSV file: {e!s}. "
                        "Try manually specifying the delimiter with the 'delimiter' parameter."
                    ) from inner_e
            else:
                raise ValueError(
                    f"Error reading CSV file with delimiter '{delimiter}': {e!s}"
                ) from e

    def _load_from_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Loads data from an existing DataFrame.

        :param dataframe: DataFrame with data
        :return: Copy of the DataFrame
        """
        return dataframe.copy()
