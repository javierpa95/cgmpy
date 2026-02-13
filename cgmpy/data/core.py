"""
Módulo principal para el manejo modular de datos de glucosa.
"""

import datetime
import logging
import time
from typing import Any, Dict, Union

import pandas as pd

from .analyzer import DataAnalyzer
from .exporter import DataExporter
from .loader import DataLoader
from .processor import DataProcessor


class ModularGlucoseData:
    """
    Clase principal refactorizada para manejo modular de datos de glucosa.

    Esta clase integra todos los módulos especializados:
    - DataLoader: Carga de datos desde diferentes fuentes
    - DataProcessor: Procesamiento y validación de datos
    - DataAnalyzer: Análisis básico de datos
    - DataExporter: Exportación a diferentes formatos
    """

    def __init__(
        self,
        data_source: Union[str, pd.DataFrame],
        date_col: str = "time",
        glucose_col: str = "glucose",
        delimiter: Union[str, None] = None,
        header: int = 0,
        start_date: Union[str, datetime.datetime, None] = None,
        end_date: Union[str, datetime.datetime, None] = None,
        log: bool = False,
    ):
        """
        Inicializa los datos de glucosa con arquitectura modular.

        :param data_source: Archivo CSV/Parquet o DataFrame con los datos
        :param date_col: Nombre de la columna de fecha/hora
        :param glucose_col: Nombre de la columna de valores de glucosa
        :param delimiter: Delimitador para archivos CSV
        :param header: Fila de encabezado para archivos CSV
        :param start_date: Fecha de inicio para filtrar datos (opcional)
        :param end_date: Fecha de fin para filtrar datos (opcional)
        :param log: Si True, activa logs detallados de rendimiento
        """
        # Configuración del logging
        self.log = log
        self.logger = self._setup_logger()

        # Almacenar parámetros
        self.date_col = date_col
        self.glucose_col = glucose_col

        # Inicializar módulos
        self.loader = DataLoader(self.logger)
        self.processor = DataProcessor(self.logger)
        self.analyzer = DataAnalyzer(self.logger)
        self.exporter = DataExporter(self.logger)

        # Procesar datos
        self.data, self.time_diffs, self.typical_interval = self._initialize_data(
            data_source, date_col, glucose_col, delimiter, header, start_date, end_date
        )

        # Diccionario para almacenar logs de operaciones
        self.logs = {}

    def _setup_logger(self) -> logging.Logger:
        """
        Configura el logger para la clase.

        :return: Logger configurado
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
        data_source: Union[str, pd.DataFrame],
        date_col: str,
        glucose_col: str,
        delimiter: Union[str, None],
        header: int,
        start_date: Union[str, datetime.datetime, None],
        end_date: Union[str, datetime.datetime, None],
    ) -> tuple:
        """
        Inicializa y procesa los datos usando los módulos.

        :return: Tupla con (data, time_diffs, typical_interval)
        """
        t_start = time.time()

        # Paso 1: Cargar datos
        t0 = time.time()
        raw_data = self.loader.load_from_source(data_source, date_col, glucose_col, delimiter, header)
        t1 = time.time()

        # Paso 2: Renombrar columnas
        t2 = time.time()
        data = self.processor.rename_columns(raw_data, date_col, glucose_col)
        t3 = time.time()

        # Paso 3: Filtrar por fechas
        t4 = time.time()
        if start_date is not None or end_date is not None:
            data = self.processor.filter_by_dates(data, start_date, end_date)
        t5 = time.time()

        # Paso 4: Procesar datos
        t6 = time.time()
        processed_data, time_diffs = self.processor.process_data(data, "time", "glucose", self.log)
        t7 = time.time()

        # Paso 5: Calcular intervalo típico
        t8 = time.time()
        typical_interval = self.analyzer.calculate_typical_interval(time_diffs, self.log)
        t9 = time.time()

        t_end = time.time()

        # Logging de tiempos
        if self.log:
            self.logger.debug(f"""
            Tiempos de inicialización modular:
            Carga de datos: {t1 - t0:.3f}s
            Renombrado de columnas: {t3 - t2:.3f}s
            Filtrado por fechas: {t5 - t4:.3f}s
            Procesamiento de datos: {t7 - t6:.3f}s
            Cálculo de intervalo típico: {t9 - t8:.3f}s
            Tiempo total: {t_end - t_start:.3f}s
            """)

        if self.log:
            self.logger.info(f"Data source loaded and processed in {t_end - t_start:.3f} seconds.")

        return processed_data, time_diffs, typical_interval

    def get_typical_interval(self) -> float:
        """
        Devuelve el intervalo típico entre mediciones en minutos.

        :return: Intervalo típico en minutos
        """
        return self.typical_interval

    def info(self, include_disconnections: bool = False) -> Dict[str, Any]:
        """
        Muestra información básica del archivo en formato JSON.

        :param include_disconnections: Si incluir detalles de desconexiones
        :return: Diccionario con información básica
        """
        return self.analyzer.get_basic_info(self.data, self.time_diffs, self.typical_interval, include_disconnections)

    def __str__(self) -> str:
        """
        Representación en string del objeto con información básica.

        :return: String con resumen de información
        """
        info = self.info()
        return self.analyzer.get_summary_string(info)

    def get_data_quality_metrics(self) -> Dict[str, Any]:
        """
        Calcula métricas de calidad de los datos.

        :return: Diccionario con métricas de calidad
        """
        return self.analyzer.get_data_quality_metrics(self.data, self.time_diffs, self.typical_interval)

    def get_logs(self) -> Dict[str, Any]:
        """
        Devuelve todos los logs almacenados.

        :return: Diccionario con logs generados
        """
        if not self.log:
            self.logger.warning("Logging no está activado. Inicialice la clase con log=True para generar logs.")
            return {}
        return self.logs

    # Métodos de exportación - Delegan al DataExporter
    def to_parquet(self, file_path: str, compression: str = "snappy", sort: bool = True):
        """Delega al DataExporter para guardar en formato Parquet."""
        self.exporter.to_parquet(self.data, file_path, compression, sort)

    def append_to_parquet(
        self,
        file_path: str,
        compression: str = "snappy",
        handle_duplicates: str = "keep_new",
    ) -> int:
        """Delega al DataExporter para añadir a archivo Parquet existente."""
        return self.exporter.append_to_parquet(self.data, file_path, compression, handle_duplicates)

    def to_csv(self, file_path: str, separator: str = ",", include_index: bool = False):
        """Delega al DataExporter para guardar en formato CSV."""
        self.exporter.to_csv(self.data, file_path, separator, include_index)

    def to_excel(self, file_path: str, sheet_name: str = "glucose_data"):
        """Delega al DataExporter para guardar en formato Excel."""
        self.exporter.to_excel(self.data, file_path, sheet_name)

    # Métodos de acceso a datos
    def get_raw_data(self) -> pd.DataFrame:
        """
        Devuelve el DataFrame con los datos procesados.

        :return: DataFrame con los datos de glucosa
        """
        return self.data.copy()

    def get_glucose_values(self) -> pd.Series:
        """
        Devuelve solo los valores de glucosa.

        :return: Series con valores de glucosa
        """
        return self.data["glucose"].copy()

    def get_timestamps(self) -> pd.Series:
        """
        Devuelve solo los timestamps.

        :return: Series con timestamps
        """
        return self.data["time"].copy()

    def get_time_differences(self) -> pd.Series:
        """
        Devuelve las diferencias de tiempo entre mediciones.

        :return: Series con diferencias de tiempo
        """
        return self.time_diffs.copy()

    # Métodos de filtrado
    def filter_by_date_range(
        self,
        start_date: Union[str, datetime.datetime],
        end_date: Union[str, datetime.datetime],
    ) -> "ModularGlucoseData":
        """
        Crea una nueva instancia filtrada por rango de fechas.

        :param start_date: Fecha de inicio
        :param end_date: Fecha de fin
        :return: Nueva instancia con datos filtrados
        """
        # Usar el processor para filtrar los datos
        filtered_data = self.processor.filter_by_dates(self.data, start_date, end_date)

        # Crear nueva instancia usando el constructor
        return self._create_filtered_instance(filtered_data)

    def filter_by_glucose_range(self, min_glucose: float, max_glucose: float) -> "ModularGlucoseData":
        """
        Crea una nueva instancia filtrada por rango de glucosa.

        :param min_glucose: Valor mínimo de glucosa
        :param max_glucose: Valor máximo de glucosa
        :return: Nueva instancia con datos filtrados
        """
        mask = (self.data["glucose"] >= min_glucose) & (self.data["glucose"] <= max_glucose)
        filtered_data = self.data[mask].copy()

        if len(filtered_data) == 0:
            raise ValueError("No hay datos en el rango de glucosa especificado.")

        return self._create_filtered_instance(filtered_data)

    def _create_filtered_instance(self, filtered_data: pd.DataFrame) -> "ModularGlucoseData":
        """
        Crea una nueva instancia con datos filtrados.

        :param filtered_data: DataFrame con los datos filtrados
        :return: Nueva instancia de ModularGlucoseData
        """
        # Crear nueva instancia
        new_instance = ModularGlucoseData.__new__(ModularGlucoseData)

        # Copiar configuración
        new_instance.log = self.log
        new_instance.logger = self.logger
        new_instance.date_col = self.date_col
        new_instance.glucose_col = self.glucose_col
        new_instance.loader = self.loader
        new_instance.processor = self.processor
        new_instance.analyzer = self.analyzer
        new_instance.exporter = self.exporter
        new_instance.logs = {}

        # Asignar datos filtrados y recalcular métricas
        new_instance.data = filtered_data
        new_instance.time_diffs = filtered_data["time"].diff()
        new_instance.typical_interval = self.analyzer.calculate_typical_interval(new_instance.time_diffs)

        return new_instance
