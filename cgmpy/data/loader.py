"""
Módulo para carga de datos de glucosa desde diferentes fuentes.
"""
import pandas as pd
import os
from typing import Union
import logging


class DataLoader:
    """
    Clase responsable de cargar datos desde diferentes fuentes (CSV, Parquet, DataFrame).
    """
    
    def __init__(self, logger: logging.Logger = None):
        """
        Inicializa el DataLoader.
        
        :param logger: Logger para registrar operaciones
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def load_from_source(
        self,
        data_source: Union[str, pd.DataFrame],
        date_col: str,
        glucose_col: str,
        delimiter: Union[str, None] = None,
        header: int = 0,
    ) -> pd.DataFrame:
        """
        Carga datos desde diferentes fuentes.
        
        :param data_source: Archivo CSV/Parquet o DataFrame
        :param date_col: Nombre de la columna de fecha
        :param glucose_col: Nombre de la columna de glucosa
        :param delimiter: Delimitador para archivos CSV
        :param header: Fila de encabezado
        :return: DataFrame con los datos cargados
        """
        if isinstance(data_source, str):
            return self._load_from_file(data_source, date_col, glucose_col, delimiter, header)
        elif isinstance(data_source, pd.DataFrame):
            return self._load_from_dataframe(data_source)
        else:
            raise ValueError("data_source debe ser un archivo CSV, Parquet o un DataFrame")
    
    def _load_from_file(
        self,
        file_path: str,
        date_col: str,
        glucose_col: str,
        delimiter: Union[str, None],
        header: int,
    ) -> pd.DataFrame:
        """
        Carga datos desde un archivo.
        
        :param file_path: Ruta al archivo
        :param date_col: Nombre de la columna de fecha
        :param glucose_col: Nombre de la columna de glucosa
        :param delimiter: Delimitador para CSV
        :param header: Fila de encabezado
        :return: DataFrame con los datos
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        
        is_parquet = file_path.lower().endswith(".parquet")
        
        if is_parquet:
            return self._load_parquet(file_path, date_col, glucose_col)
        else:
            return self._load_csv(file_path, date_col, glucose_col, delimiter, header)
    
    def _load_parquet(self, file_path: str, date_col: str, glucose_col: str) -> pd.DataFrame:
        """
        Carga datos desde un archivo Parquet.
        
        :param file_path: Ruta al archivo Parquet
        :param date_col: Nombre de la columna de fecha
        :param glucose_col: Nombre de la columna de glucosa
        :return: DataFrame con los datos
        """
        try:
            return pd.read_parquet(file_path, columns=[date_col, glucose_col])
        except Exception as e:
            raise ValueError(f"Error al leer el archivo Parquet: {str(e)}") from e
    
    def _load_csv(
        self,
        file_path: str,
        date_col: str,
        glucose_col: str,
        delimiter: Union[str, None],
        header: int,
    ) -> pd.DataFrame:
        """
        Carga datos desde un archivo CSV.
        
        :param file_path: Ruta al archivo CSV
        :param date_col: Nombre de la columna de fecha
        :param glucose_col: Nombre de la columna de glucosa
        :param delimiter: Delimitador
        :param header: Fila de encabezado
        :return: DataFrame con los datos
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
            # Intentar con delimitador alternativo si falla
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
                        f"Error al leer el archivo CSV: {str(e)}. "
                        "Intente especificar manualmente el delimitador con el parámetro 'delimiter'."
                    ) from inner_e
            else:
                raise ValueError(
                    f"Error al leer el archivo CSV con delimitador '{delimiter}': {str(e)}"
                ) from e
    
    def _load_from_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Carga datos desde un DataFrame existente.
        
        :param dataframe: DataFrame con los datos
        :return: Copia del DataFrame
        """
        return dataframe.copy() 