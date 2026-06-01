"""
Módulo para análisis básico de datos de glucosa.
"""

import logging
import time
from typing import Any

import numpy as np
import pandas as pd


class DataAnalyzer:
    """
    Clase responsable del análisis básico de datos de glucosa.
    """

    def __init__(self, logger: logging.Logger | None = None):
        """
        Inicializa el DataAnalyzer.

        :param logger: Logger para registrar operaciones
        """
        self.logger = logger or logging.getLogger(__name__)

    def calculate_typical_interval(
        self, time_diffs: pd.Series, log_performance: bool = False
    ) -> float:
        """
        Calcula el intervalo típico entre mediciones en minutos.

        :param time_diffs: Series con las diferencias de tiempo
        :param log_performance: Si True, registra métricas de rendimiento
        :return: Intervalo típico en minutos
        """
        if log_performance:
            t_start = time.time()
            self.logger.debug("\n--- ANÁLISIS DE CÁLCULO DE INTERVALO TÍPICO ---")

        # Convertir a array de NumPy para operaciones más rápidas
        time_diffs_seconds = time_diffs.dt.total_seconds().values
        # Filtrar valores válidos (mayores que 0)
        valid_diffs = time_diffs_seconds[time_diffs_seconds > 0]

        if len(valid_diffs) > 0:
            # Usar NumPy para calcular la mediana (más rápido)
            intervalo = np.median(valid_diffs) / 60
        else:
            # Valor predeterminado si no hay diferencias válidas
            intervalo = 5.0

        if log_performance:
            t_end = time.time()
            self.logger.debug(f"Cálculo optimizado de mediana: {t_end - t_start:.3f}s")
            self.logger.debug(f"Tiempo total de cálculo de intervalo: {t_end - t_start:.3f}s")
            self.logger.debug("--- FIN DEL ANÁLISIS ---\n")

        return abs(intervalo)

    def get_basic_info(
        self,
        data: pd.DataFrame,
        time_diffs: pd.Series,
        typical_interval: float,
        include_disconnections: bool = False,
    ) -> dict[str, Any]:
        """
        Genera información básica sobre los datos de glucosa.

        :param data: DataFrame con los datos de glucosa
        :param time_diffs: Series con las diferencias de tiempo
        :param typical_interval: Intervalo típico entre mediciones
        :param include_disconnections: Si incluir detalles de desconexiones
        :return: Diccionario con información básica
        """
        # Información básica
        num_datos = len(data)
        fecha_inicio = data["time"].min()
        fecha_fin = data["time"].max()

        # Análisis de desconexiones
        umbral_desconexion = pd.Timedelta(minutes=typical_interval + 10)
        desconexiones = time_diffs[time_diffs > umbral_desconexion]
        num_desconexiones = len(desconexiones)

        # Tiempo total de desconexión
        tiempo_total_desconexion = desconexiones.sum()
        horas_desconexion = tiempo_total_desconexion.total_seconds() / 3600

        # Uso de memoria
        memoria_bytes = data.memory_usage(deep=True).sum()
        memoria_mb = memoria_bytes / (1024 * 1024)

        # Datos teóricos esperados
        tiempo_total = (data["time"].max() - data["time"].min()).total_seconds() / 60

        # Evitar errores si tiempo_total o typical_interval son inválidos
        if pd.isna(tiempo_total) or typical_interval <= 0:
            datos_teoricos = 0
        else:
            datos_teoricos = int(tiempo_total / typical_interval)

        porcentaje_disponibilidad = (
            (num_datos / datos_teoricos * 100) if datos_teoricos > 0 else 0.0
        )

        # Create summary dictionary
        summary = {
            "n_records": num_datos,
            "start_date": fecha_inicio,
            "end_date": fecha_fin,
            "typical_interval": typical_interval,
            "expected_data": datos_teoricos,
            "completeness": porcentaje_disponibilidad,
            "n_disconnections": (
                f"{num_desconexiones} disconnections (For more info, "
                "use info(include_disconnections=True))"
            ),
            "total_disconnection_time": horas_desconexion,
            "memory_usage_mb": memoria_mb,
        }

        if include_disconnections:
            summary["disconnection_list"] = self._get_disconnection_details(data, desconexiones)

        return summary

    def _get_disconnection_details(self, data: pd.DataFrame, disconnections: pd.Series) -> list:
        """
        Gets details of disconnections.

        :param data: DataFrame with data
        :param disconnections: Series with disconnections
        :return: List with disconnection details
        """
        disconnection_list = []

        if len(disconnections) > 0:
            for idx, index in enumerate(disconnections.index, 1):
                try:
                    current_pos = data.index.get_loc(index)
                    if current_pos > 0:
                        disconnection_end = data.iloc[current_pos]["time"]
                        disconnection_start = data.iloc[current_pos - 1]["time"]
                        duration_minutes = (
                            disconnection_end - disconnection_start
                        ).total_seconds() / 60
                        hours = int(duration_minutes // 60)
                        minutes = int(duration_minutes % 60)
                        disconnection_list.append(
                            {
                                "start": disconnection_start.strftime("%d/%m/%Y %H:%M"),
                                "end": disconnection_end.strftime("%d/%m/%Y %H:%M"),
                                "duration": f"{hours:02d} hours and {minutes:02d} minutes",
                            }
                        )
                except Exception as e:
                    self.logger.warning(f"Error processing disconnection {idx}: {e}")

        return disconnection_list

    def get_summary_string(self, info: dict[str, Any]) -> str:
        """
        Generates a string representation of basic information.

        :param info: Dictionary with basic information
        :return: String with summary
        """
        return (
            f"File contains {info['n_records']} records between {info['start_date']} and {info['end_date']}.\n"
            f"Typical interval between measurements: {info['typical_interval']:.1f} minutes.\n"
            f"Expected theoretical data: {info['expected_data']}\n"
            f"Data availability percentage: {info['completeness']:.1f}%\n"
            f"Detected {info['n_disconnections']}\n"
            f"Total disconnection time: {info['total_disconnection_time']:.1f} hours.\n"
            f"DataFrame memory usage: {info['memory_usage_mb']:.2f} MB"
        )

    def get_data_quality_metrics(
        self, data: pd.DataFrame, time_diffs: pd.Series, typical_interval: float
    ) -> dict[str, Any]:
        """
        Calcula métricas de calidad de los datos.

        :param data: DataFrame con los datos
        :param time_diffs: Series con las diferencias de tiempo
        :param typical_interval: Intervalo típico entre mediciones
        :return: Diccionario con métricas de calidad
        """
        # Calcular gaps en los datos
        umbral_gap = pd.Timedelta(minutes=typical_interval * 2)
        gaps = time_diffs[time_diffs > umbral_gap]

        # Calcular estadísticas de intervalos
        valid_intervals = time_diffs[time_diffs > pd.Timedelta(0)].dt.total_seconds() / 60

        return {
            "total_gaps": len(gaps),
            "max_gap_hours": gaps.max().total_seconds() / 3600 if len(gaps) > 0 else 0,
            "mean_interval": valid_intervals.mean() if len(valid_intervals) > 0 else 0,
            "std_interval": valid_intervals.std() if len(valid_intervals) > 0 else 0,
            "min_glucose": data["glucose"].min(),
            "max_glucose": data["glucose"].max(),
            "mean_glucose": data["glucose"].mean(),
            "std_glucose": data["glucose"].std(),
        }
