"""
Módulo para exportación de datos de glucosa.
"""

import logging
import os
import time

import pandas as pd


class DataExporter:
    """
    Clase responsable de exportar datos de glucosa en diferentes formatos.
    """

    def __init__(self, logger: logging.Logger = None):
        """
        Inicializa el DataExporter.

        :param logger: Logger para registrar operaciones
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
        Guarda los datos en formato Parquet optimizado.

        :param data: DataFrame con los datos
        :param file_path: Ruta donde guardar el archivo
        :param compression: Algoritmo de compresión
        :param sort: Si ordenar los datos antes de guardar
        """
        self.logger.info("Preparando datos para guardar en formato Parquet optimizado...")

        # Preparar datos para guardar
        df_to_save = data.copy()

        # Ordenar por tiempo si se solicita
        if sort and not df_to_save["time"].is_monotonic_increasing:
            self.logger.info("  - Ordenando datos por timestamp...")
            df_to_save = df_to_save.sort_values("time", ignore_index=True)

        # Optimizar tipos de datos
        df_to_save = self._optimize_data_types(df_to_save)

        # Eliminar duplicados si existen
        df_to_save = self._remove_duplicates(df_to_save)

        # Guardar en formato Parquet
        t_start = time.time()
        df_to_save.to_parquet(file_path, compression=compression, index=False, engine="pyarrow")
        t_end = time.time()

        # Información final
        self._log_save_info(file_path, df_to_save, t_end - t_start)

    def append_to_parquet(
        self,
        data: pd.DataFrame,
        file_path: str,
        compression: str = "snappy",
        handle_duplicates: str = "keep_new",
    ) -> int:
        """
        Añade datos a un archivo Parquet existente.

        :param data: DataFrame con los nuevos datos
        :param file_path: Ruta al archivo Parquet
        :param compression: Algoritmo de compresión
        :param handle_duplicates: Estrategia para manejar duplicados
        :return: Número de registros añadidos
        """
        if not os.path.exists(file_path):
            self.logger.info(f"El archivo {file_path} no existe. Creando nuevo archivo...")
            self.to_parquet(data, file_path, compression=compression)
            return len(data)

        self.logger.info(f"Añadiendo datos a archivo Parquet existente: {file_path}")

        # Cargar datos existentes
        t_start = time.time()
        existing_data = pd.read_parquet(file_path)
        t_load = time.time()
        self.logger.info(f"  - Archivo existente cargado en {t_load - t_start:.3f}s")
        self.logger.info(f"  - Registros existentes: {len(existing_data):,}")

        # Preparar nuevos datos
        new_data = self._prepare_new_data(data)

        # Manejar duplicados
        existing_data, new_data = self._handle_duplicates(existing_data, new_data, handle_duplicates)

        # Combinar y ordenar datos
        t_combine = time.time()
        final_data = pd.concat([existing_data, new_data])
        final_data = final_data.sort_values("time", ignore_index=True)
        t_sort = time.time()
        self.logger.info(f"  - Datos combinados y ordenados en {t_sort - t_combine:.3f}s")

        # Guardar resultado
        final_data.to_parquet(file_path, compression=compression, index=False, engine="pyarrow")
        t_save = time.time()

        # Información final
        records_added = len(final_data) - len(existing_data)
        file_size = os.path.getsize(file_path) / 1024 / 1024
        print("Datos añadidos correctamente:")
        print(f"  - Registros añadidos: {records_added:,}")
        print(f"  - Total registros: {len(final_data):,}")
        print(f"  - Tamaño del archivo: {file_size:.2f} MB")
        print(f"  - Tiempo total: {t_save - t_start:.3f}s")

        return records_added

    def _optimize_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Optimiza los tipos de datos para almacenamiento eficiente.

        :param data: DataFrame con los datos
        :return: DataFrame con tipos optimizados
        """
        df_optimized = data.copy()

        # Optimizar columna de glucosa
        if not pd.api.types.is_integer_dtype(df_optimized["glucose"]):
            self.logger.info("  - Convirtiendo 'glucose' a formato numérico...")
            df_optimized["glucose"] = pd.to_numeric(df_optimized["glucose"], errors="coerce")

        # Intentar convertir a int16 si es posible
        min_val = df_optimized["glucose"].min()
        max_val = df_optimized["glucose"].max()

        if pd.notna(min_val) and pd.notna(max_val) and min_val >= -32768 and max_val <= 32767:
            self.logger.info(f"  - Optimizando 'glucose' a int16 (rango: {min_val} a {max_val})...")
            df_optimized["glucose"] = df_optimized["glucose"].astype("int16")
        else:
            self.logger.warning(
                f"  - Valores de glucosa fuera del rango de int16 ({min_val} a {max_val}). Usando int32."
            )
            df_optimized["glucose"] = df_optimized["glucose"].astype("int32")

        # Verificar que time sea datetime
        if not pd.api.types.is_datetime64_any_dtype(df_optimized["time"]):
            self.logger.info("  - Convirtiendo 'time' a datetime...")
            df_optimized["time"] = pd.to_datetime(df_optimized["time"], errors="coerce")

        return df_optimized

    def _remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Elimina duplicados basados en la columna time.

        :param data: DataFrame con los datos
        :return: DataFrame sin duplicados
        """
        duplicados = data.duplicated(subset=["time"], keep="first")
        if duplicados.any():
            num_duplicados = duplicados.sum()
            self.logger.info(f"  - Eliminando {num_duplicados} timestamps duplicados...")
            return data.drop_duplicates(subset=["time"], keep="first")
        return data

    def _prepare_new_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara los nuevos datos para la inserción.

        :param data: DataFrame con los nuevos datos
        :return: DataFrame preparado
        """
        new_data = data.copy()

        # Asegurar tipos correctos
        if not pd.api.types.is_datetime64_any_dtype(new_data["time"]):
            new_data["time"] = pd.to_datetime(new_data["time"], errors="coerce")

        if not pd.api.types.is_integer_dtype(new_data["glucose"]):
            new_data["glucose"] = pd.to_numeric(new_data["glucose"], errors="coerce")

            # Convertir a int16 si es posible
            min_val = new_data["glucose"].min()
            max_val = new_data["glucose"].max()
            if pd.notna(min_val) and pd.notna(max_val) and min_val >= -32768 and max_val <= 32767:
                new_data["glucose"] = new_data["glucose"].astype("int16")

        return new_data

    def _handle_duplicates(self, existing_data: pd.DataFrame, new_data: pd.DataFrame, strategy: str) -> tuple:
        """
        Maneja duplicados entre datos existentes y nuevos.

        :param existing_data: DataFrame con datos existentes
        :param new_data: DataFrame con datos nuevos
        :param strategy: Estrategia para manejar duplicados
        :return: Tupla con DataFrames procesados
        """
        # Identificar duplicados
        combined = pd.concat([existing_data, new_data])
        duplicated_times = combined["time"].duplicated(keep=False)
        num_duplicates = duplicated_times.sum() // 2

        if num_duplicates > 0:
            self.logger.info(f"  - Encontrados {num_duplicates} timestamps duplicados")

            if strategy == "keep_new":
                self.logger.info("  - Estrategia: Mantener nuevos datos en caso de duplicados")
                existing_times = set(existing_data["time"])
                new_times = set(new_data["time"])
                common_times = existing_times.intersection(new_times)

                if common_times:
                    existing_data = existing_data[~existing_data["time"].isin(common_times)]

            elif strategy == "keep_old":
                self.logger.info("  - Estrategia: Mantener datos existentes en caso de duplicados")
                existing_times = set(existing_data["time"])
                new_data = new_data[~new_data["time"].isin(existing_times)]

        return existing_data, new_data

    def _log_save_info(self, file_path: str, data: pd.DataFrame, save_time: float):
        """
        Registra información sobre el guardado.

        :param file_path: Ruta del archivo guardado
        :param data: DataFrame guardado
        :param save_time: Tiempo de guardado
        """
        file_size = os.path.getsize(file_path) / 1024 / 1024
        print(f"Datos guardados en formato Parquet en: {file_path}")
        print(f"  - Tamaño del archivo: {file_size:.2f} MB")
        print(f"  - Tiempo de guardado: {save_time:.3f}s")
        print(f"  - Registros guardados: {len(data):,}")
        print(f"  - Rango de fechas: {data['time'].min()} a {data['time'].max()}")
        print("  - Formato listo para carga rápida")

    def to_csv(
        self,
        data: pd.DataFrame,
        file_path: str,
        separator: str = ",",
        include_index: bool = False,
    ):
        """
        Guarda los datos en formato CSV.

        :param data: DataFrame con los datos
        :param file_path: Ruta donde guardar el archivo
        :param separator: Separador de campos
        :param include_index: Si incluir el índice
        """
        self.logger.info(f"Guardando datos en formato CSV: {file_path}")

        t_start = time.time()
        data.to_csv(file_path, sep=separator, index=include_index)
        t_end = time.time()

        file_size = os.path.getsize(file_path) / 1024 / 1024
        print(f"Datos guardados en formato CSV en: {file_path}")
        print(f"  - Tamaño del archivo: {file_size:.2f} MB")
        print(f"  - Tiempo de guardado: {t_end - t_start:.3f}s")
        print(f"  - Registros guardados: {len(data):,}")

    def to_excel(self, data: pd.DataFrame, file_path: str, sheet_name: str = "glucose_data"):
        """
        Guarda los datos en formato Excel.

        :param data: DataFrame con los datos
        :param file_path: Ruta donde guardar el archivo
        :param sheet_name: Nombre de la hoja
        """
        self.logger.info(f"Guardando datos en formato Excel: {file_path}")

        t_start = time.time()
        data.to_excel(file_path, sheet_name=sheet_name, index=False)
        t_end = time.time()

        file_size = os.path.getsize(file_path) / 1024 / 1024
        print(f"Datos guardados en formato Excel en: {file_path}")
        print(f"  - Tamaño del archivo: {file_size:.2f} MB")
        print(f"  - Tiempo de guardado: {t_end - t_start:.3f}s")
        print(f"  - Registros guardados: {len(data):,}")
