"""
Módulo para procesamiento y validación de datos de glucosa.
"""

import logging
import time
from typing import Tuple, Union

import numpy as np
import pandas as pd


class DataProcessor:
    """
    Clase responsable del procesamiento, validación y limpieza de datos de glucosa.
    """

    def __init__(self, logger: logging.Logger = None):
        """
        Inicializa el DataProcessor.

        :param logger: Logger para registrar operaciones
        """
        self.logger = logger or logging.getLogger(__name__)

    def process_data(
        self,
        data: pd.DataFrame,
        date_col: str,
        glucose_col: str,
        log_performance: bool = False,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Procesa los datos de glucosa de forma optimizada.

        :param data: DataFrame con los datos
        :param date_col: Nombre de la columna de fecha
        :param glucose_col: Nombre de la columna de glucosa
        :param log_performance: Si True, registra métricas de rendimiento
        :return: Tupla con DataFrame procesado y Series de diferencias de tiempo
        """
        t_start = time.time()

        if log_performance:
            self.logger.debug("\n--- ANÁLISIS DETALLADO DE RENDIMIENTO ---")

        # Validar columnas
        self._validate_columns(data, date_col, glucose_col)

        # Determinar si los datos vienen de Parquet optimizado
        from_parquet = self._is_optimized_parquet(data, date_col, glucose_col)

        if from_parquet:
            processed_data, time_diffs = self._process_parquet_optimized(data, date_col, glucose_col, log_performance)
        else:
            processed_data, time_diffs = self._process_standard(data, date_col, glucose_col, log_performance)

        # Validación final
        if not pd.api.types.is_datetime64_any_dtype(processed_data[date_col]):
            raise ValueError("Error en conversión de fechas")

        if log_performance:
            t_end = time.time()
            memoria_bytes = processed_data.memory_usage(deep=True).sum()
            memoria_mb = memoria_bytes / (1024 * 1024)
            self.logger.debug(f"Uso de memoria del DataFrame: {memoria_mb:.2f} MB")
            self.logger.debug(f"Tiempo total de procesamiento: {t_end - t_start:.3f}s")
            self.logger.debug("--- FIN DEL ANÁLISIS ---\n")

        return processed_data, time_diffs

    def _validate_columns(self, data: pd.DataFrame, date_col: str, glucose_col: str):
        """
        Valida que las columnas especificadas existan en el DataFrame.

        :param data: DataFrame a validar
        :param date_col: Nombre de la columna de fecha
        :param glucose_col: Nombre de la columna de glucosa
        """
        if date_col not in data.columns or glucose_col not in data.columns:
            raise ValueError(
                f"Las columnas '{date_col}' o '{glucose_col}' no se encuentran en el DataFrame. "
                f"Columnas disponibles: {data.columns.tolist()}."
            )

    def _is_optimized_parquet(self, data: pd.DataFrame, date_col: str, glucose_col: str) -> bool:
        """
        Determina si los datos vienen de un archivo Parquet optimizado.

        :param data: DataFrame con los datos
        :param date_col: Nombre de la columna de fecha
        :param glucose_col: Nombre de la columna de glucosa
        :return: True si es Parquet optimizado
        """
        return pd.api.types.is_datetime64_any_dtype(data[date_col]) and data[glucose_col].dtype == "int16"

    def _process_parquet_optimized(
        self,
        data: pd.DataFrame,
        date_col: str,
        glucose_col: str,
        log_performance: bool = False,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Procesa datos optimizados de Parquet con ruta rápida.

        :param data: DataFrame con los datos
        :param date_col: Nombre de la columna de fecha
        :param glucose_col: Nombre de la columna de glucosa
        :param log_performance: Si True, registra métricas de rendimiento
        :return: Tupla con DataFrame procesado y Series de diferencias de tiempo
        """
        if log_performance:
            self.logger.debug("Detectados datos de origen Parquet con tipos optimizados.")
            self.logger.debug("Aplicando ruta rápida para datos Parquet...")

        processed_data = data.copy()

        # Verificar y eliminar nulos
        t_nulos = time.time()
        if processed_data.isna().any().any():
            processed_data = processed_data.dropna(subset=[date_col, glucose_col])
            if log_performance:
                self.logger.debug(f"  - Eliminados valores nulos: {time.time() - t_nulos:.3f}s")
        elif log_performance:
            self.logger.debug(f"  - No hay valores nulos: {time.time() - t_nulos:.3f}s")

        # Verificar y ordenar
        t_orden = time.time()
        if not processed_data[date_col].is_monotonic_increasing:
            if log_performance:
                self.logger.debug("  - Ordenando datos...")
            processed_data = processed_data.sort_values(date_col, ignore_index=True)
        elif log_performance:
            self.logger.debug("  - Datos ya ordenados")

        if log_performance:
            self.logger.debug(f"  - Verificación de orden: {time.time() - t_orden:.3f}s")

        # Cálculo optimizado de diferencias de tiempo
        t_diff = time.time()
        time_values = processed_data[date_col].values
        time_diffs_ns = np.diff(time_values.astype("datetime64[ns]"))
        time_diffs_ns = np.insert(time_diffs_ns, 0, np.timedelta64(0, "ns"))
        time_diffs = pd.Series(pd.TimedeltaIndex(time_diffs_ns), index=processed_data.index)

        if log_performance:
            self.logger.debug(f"  - Cálculo optimizado de diferencias: {time.time() - t_diff:.3f}s")

        return processed_data, time_diffs

    def _process_standard(
        self,
        data: pd.DataFrame,
        date_col: str,
        glucose_col: str,
        log_performance: bool = False,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Procesa datos con validaciones completas para CSV y otros formatos.

        :param data: DataFrame con los datos
        :param date_col: Nombre de la columna de fecha
        :param glucose_col: Nombre de la columna de glucosa
        :param log_performance: Si True, registra métricas de rendimiento
        :return: Tupla con DataFrame procesado y Series de diferencias de tiempo
        """
        if log_performance:
            self.logger.debug("Procesando datos con conversión de tipos y validaciones completas.")

        processed_data = data.copy()

        # Manejo de nulos
        t_nulos = time.time()
        processed_data = self._handle_nulls(processed_data, date_col, glucose_col)
        if log_performance:
            self.logger.debug(f"2. Eliminación de nulos: {time.time() - t_nulos:.3f}s")

        # Conversión de tipos
        t_tipos = time.time()
        processed_data = self._convert_data_types(processed_data, date_col, glucose_col)
        if log_performance:
            self.logger.debug(f"3. Conversión de tipos: {time.time() - t_tipos:.3f}s")

        # Manejo de duplicados
        t_dups = time.time()
        processed_data = self._handle_duplicates(processed_data, date_col, glucose_col)
        if log_performance:
            self.logger.debug(f"4. Procesamiento de duplicados: {time.time() - t_dups:.3f}s")

        # Ordenación
        t_orden = time.time()
        processed_data = self._sort_data(processed_data, date_col)
        if log_performance:
            self.logger.debug(f"5. Ordenación: {time.time() - t_orden:.3f}s")

        # Cálculo de diferencias
        t_diff = time.time()
        time_diffs = processed_data[date_col].diff()
        if log_performance:
            self.logger.debug(f"6. Cálculo de diferencias: {time.time() - t_diff:.3f}s")

        return processed_data, time_diffs

    def _handle_nulls(self, data: pd.DataFrame, date_col: str, glucose_col: str) -> pd.DataFrame:
        """
        Elimina filas con valores nulos en columnas clave.

        :param data: DataFrame con los datos
        :param date_col: Nombre de la columna de fecha
        :param glucose_col: Nombre de la columna de glucosa
        :return: DataFrame sin nulos
        """
        filas_antes = len(data)
        data_cleaned = data.dropna(subset=[date_col, glucose_col])
        filas_despues = len(data_cleaned)

        if filas_antes > filas_despues:
            self.logger.debug(f"  - Eliminadas {filas_antes - filas_despues} filas con valores nulos.")

        return data_cleaned

    def _convert_data_types(self, data: pd.DataFrame, date_col: str, glucose_col: str) -> pd.DataFrame:
        """
        Convierte las columnas de fecha y glucosa a los tipos correctos.

        :param data: DataFrame con los datos
        :param date_col: Nombre de la columna de fecha
        :param glucose_col: Nombre de la columna de glucosa
        :return: DataFrame con tipos correctos
        """
        data_converted = data.copy()

        # Convertir columna de fecha
        if not pd.api.types.is_datetime64_any_dtype(data_converted[date_col]):
            self.logger.debug(f"  - Convirtiendo columna '{date_col}' a datetime...")
            if pd.api.types.is_numeric_dtype(data_converted[date_col]):
                unit = "ms" if data_converted[date_col].iloc[0] > 1e10 else "s"
                data_converted[date_col] = pd.to_datetime(data_converted[date_col], unit=unit)
            else:
                data_converted[date_col] = pd.to_datetime(data_converted[date_col], errors="coerce", format="mixed")

        # Convertir columna de glucosa
        if not pd.api.types.is_numeric_dtype(data_converted[glucose_col]):
            self.logger.debug(f"  - Convirtiendo columna '{glucose_col}' a numérica...")
            data_converted[glucose_col] = pd.to_numeric(data_converted[glucose_col], errors="coerce")
            data_converted = data_converted.dropna(subset=[glucose_col])

        # Optimizar tipo de glucosa a int16 si es posible
        if data_converted[glucose_col].dtype != "int16":
            self.logger.debug(f"  - Optimizando columna '{glucose_col}'...")
            min_val, max_val = (
                data_converted[glucose_col].min(),
                data_converted[glucose_col].max(),
            )
            if pd.notna(min_val) and pd.notna(max_val) and min_val >= -32768 and max_val <= 32767:
                data_converted[glucose_col] = data_converted[glucose_col].astype("int16")
            else:
                data_converted[glucose_col] = pd.to_numeric(
                    data_converted[glucose_col], errors="coerce", downcast="float"
                )

        return data_converted

    def _handle_duplicates(self, data: pd.DataFrame, date_col: str, glucose_col: str) -> pd.DataFrame:
        """
        Encuentra y resuelve duplicados en la columna de fecha.

        :param data: DataFrame con los datos
        :param date_col: Nombre de la columna de fecha
        :param glucose_col: Nombre de la columna de glucosa
        :return: DataFrame sin duplicados
        """
        mask_duplicados = data.duplicated(subset=[date_col], keep=False)
        num_duplicados = mask_duplicados.sum()

        if num_duplicados > 0:
            self.logger.debug(f"  - Encontrados {num_duplicados // 2} timestamps duplicados. Resolviendo...")

            df_dups = data[mask_duplicados].copy()
            df_dups["diff"] = df_dups.groupby(date_col)[glucose_col].transform(lambda x: (x - x.mean()).abs())

            idx_to_keep = df_dups.groupby(date_col)["diff"].idxmin()

            return pd.concat([data[~mask_duplicados], df_dups.loc[idx_to_keep].drop(columns="diff")])

        return data

    def _sort_data(self, data: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """
        Ordena el DataFrame por la columna de fecha si no está ya ordenado.

        :param data: DataFrame con los datos
        :param date_col: Nombre de la columna de fecha
        :return: DataFrame ordenado
        """
        if not data[date_col].is_monotonic_increasing:
            self.logger.debug("  - Ordenando datos por timestamp...")
            return data.sort_values(date_col, ignore_index=True)

        self.logger.debug("  - Datos ya ordenados, omitiendo ordenación.")
        return data

    def rename_columns(self, data: pd.DataFrame, date_col: str, glucose_col: str) -> pd.DataFrame:
        """
        Renombra las columnas a nombres estándar.

        :param data: DataFrame con los datos
        :param date_col: Nombre actual de la columna de fecha
        :param glucose_col: Nombre actual de la columna de glucosa
        :return: DataFrame con columnas renombradas
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
        start_date: Union[str, pd.Timestamp, None] = None,
        end_date: Union[str, pd.Timestamp, None] = None,
        date_col: str = "time",
    ) -> pd.DataFrame:
        """
        Filtra los datos por rango de fechas.

        :param data: DataFrame con los datos
        :param start_date: Fecha de inicio
        :param end_date: Fecha de fin
        :param date_col: Nombre de la columna de fecha
        :return: DataFrame filtrado
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
            raise ValueError("No hay datos disponibles en el rango de fechas especificado.")

        return filtered_data
