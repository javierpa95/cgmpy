"""
Módulo para análisis básico de datos de glucosa.
"""

import logging
import time
from typing import Any, Dict

import numpy as np
import pandas as pd


class DataAnalyzer:
    """
    Clase responsable del análisis básico de datos de glucosa.
    """

    def __init__(self, logger: logging.Logger = None):
        """
        Inicializa el DataAnalyzer.

        :param logger: Logger para registrar operaciones
        """
        self.logger = logger or logging.getLogger(__name__)

    def calculate_typical_interval(self, time_diffs: pd.Series, log_performance: bool = False) -> float:
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
    ) -> Dict[str, Any]:
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
        datos_teoricos = int(tiempo_total / typical_interval)
        porcentaje_disponibilidad = (num_datos / datos_teoricos) * 100

        # Crear diccionario de resumen
        resumen = {
            "num_datos": num_datos,
            "fecha_inicio": fecha_inicio,
            "fecha_fin": fecha_fin,
            "intervalo_tipico": typical_interval,
            "datos_teoricos": datos_teoricos,
            "porcentaje_disponibilidad": porcentaje_disponibilidad,
            "num_desconexiones": (
                f"{num_desconexiones} desconexiones (Para más información, "
                "use el método info(include_disconnections=True))"
            ),
            "tiempo_total_desconexion": horas_desconexion,
            "uso_memoria_mb": memoria_mb,
        }

        if include_disconnections:
            resumen["lista_desconexiones"] = self._get_disconnection_details(data, desconexiones)

        return resumen

    def _get_disconnection_details(self, data: pd.DataFrame, desconexiones: pd.Series) -> list:
        """
        Obtiene detalles de las desconexiones.

        :param data: DataFrame con los datos
        :param desconexiones: Series con las desconexiones
        :return: Lista con detalles de desconexiones
        """
        lista_desconexiones = []

        if len(desconexiones) > 0:
            for idx, indice in enumerate(desconexiones.index, 1):
                try:
                    posicion_actual = data.index.get_loc(indice)
                    if posicion_actual > 0:
                        fecha_fin_desconexion = data.iloc[posicion_actual]["time"]
                        fecha_inicio_desconexion = data.iloc[posicion_actual - 1]["time"]
                        duracion_minutos = (fecha_fin_desconexion - fecha_inicio_desconexion).total_seconds() / 60
                        horas = int(duracion_minutos // 60)
                        minutos = int(duracion_minutos % 60)
                        lista_desconexiones.append(
                            {
                                "inicio": fecha_inicio_desconexion.strftime("%d/%m/%Y %H:%M"),
                                "fin": fecha_fin_desconexion.strftime("%d/%m/%Y %H:%M"),
                                "duracion": f"{horas:02d} horas y {minutos:02d} minutos",
                            }
                        )
                except Exception as e:
                    self.logger.warning(f"Error procesando desconexión {idx}: {e}")

        return lista_desconexiones

    def get_summary_string(self, info: Dict[str, Any]) -> str:
        """
        Genera una representación en string de la información básica.

        :param info: Diccionario con información básica
        :return: String con resumen
        """
        return (
            f"El archivo contiene {info['num_datos']} datos entre {info['fecha_inicio']} y {info['fecha_fin']}.\n"
            f"Intervalo típico entre mediciones: {info['intervalo_tipico']:.1f} minutos.\n"
            f"Datos teóricos esperados: {info['datos_teoricos']}\n"
            f"Porcentaje de datos disponibles: {info['porcentaje_disponibilidad']:.1f}%\n"
            f"Se detectaron {info['num_desconexiones']} desconexiones.\n"
            f"Tiempo total de desconexión: {info['tiempo_total_desconexion']:.1f} horas.\n"
            f"Uso de memoria del DataFrame: {info['uso_memoria_mb']:.2f} MB"
        )

    def get_data_quality_metrics(
        self, data: pd.DataFrame, time_diffs: pd.Series, typical_interval: float
    ) -> Dict[str, Any]:
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
