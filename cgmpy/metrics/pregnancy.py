"""
Módulo de métricas específicas para diabetes gestacional.

Este módulo contiene las métricas y análisis específicos para el seguimiento
de la diabetes durante el embarazo, incluyendo análisis por trimestres.
"""

import datetime
from typing import Any, Dict, Union

import pandas as pd

from ..data.pregnancy_data import PregnancyData
from . import ModularGlucoseMetrics


class PregnancyMetrics:
    """
    Mixin para métricas específicas de diabetes gestacional.

    Esta clase debe ser utilizada como mixin con una clase que herede de PregnancyData.
    """

    def _init_pregnancy_metrics(self):
        """
        Inicializa las métricas de embarazo delegando la gestión de datos a la clase de datos base.
        """
        # Importación local para evitar circulares
        from .. import GlucoseMetrics

        # Crear instancias de análisis por trimestre
        # Usamos los dataframes ya segregados en la clase de datos (PregnancyData)
        self.primer_trimestre = self._create_trimester_metrics(self.trimesters["primer_trimestre"], GlucoseMetrics)
        self.segundo_trimestre = self._create_trimester_metrics(self.trimesters["segundo_trimestre"], GlucoseMetrics)
        self.tercer_trimestre = self._create_trimester_metrics(self.trimesters["tercer_trimestre"], GlucoseMetrics)

    def _create_trimester_metrics(self, df: pd.DataFrame, cls) -> Union[None, Any]:
        """
        Crea una instancia de la clase de métricas para un dataframe de trimestre.
        """
        if len(df) > 0:
            return cls(
                data_source=df,
                date_col="time",
                glucose_col="glucose",
            )
        return None

    def info_gestational(self) -> Dict[str, Any]:
        """
        Obtiene información detallada por trimestre.
        """
        info = {}
        periods = [
            ("primer_trimestre", self.fecha_concepcion, self.primer_trimestre_fin, self.primer_trimestre),
            ("segundo_trimestre", self.primer_trimestre_fin, self.segundo_trimestre_fin, self.segundo_trimestre),
            ("tercer_trimestre", self.segundo_trimestre_fin, self.fecha_parto, self.tercer_trimestre),
        ]

        for name, start, end, obj in periods:
            if obj:
                trimester_info = obj.info()
                intervalo = trimester_info["intervalo_tipico"]
                dias = (end - start).days
                esperados = int((60 / intervalo) * 24 * dias)

                trimester_info.update(
                    {
                        "datos_esperados": esperados,
                        "porcentaje_datos": (trimester_info["num_datos"] / esperados * 100) if esperados > 0 else 0,
                    }
                )
                info[name] = trimester_info
            else:
                info[name] = "No hay datos disponibles"

        return info

    def time_statistics_trimestres(self) -> Dict[str, Any]:
        """
        Calcula estadísticas de tiempo en rango por trimestre.
        """
        stats = {}
        for name, obj in [
            ("primer_trimestre", self.primer_trimestre),
            ("segundo_trimestre", self.segundo_trimestre),
            ("tercer_trimestre", self.tercer_trimestre),
        ]:
            if obj:
                stats[name] = {
                    "TIR": obj.TIR(),
                    "TIR_tight": obj.TIR_tight(),
                    "TBR70": obj.TBR70(),
                    "TBR55": obj.TBR55(),
                    "TAR140": obj.TAR140(),
                    "TAR180": obj.TAR180(),
                    "TAR250": obj.TAR250(),
                }
        return stats

    def calculate_all_metrics(self) -> Dict[str, Any]:
        """
        Calcula todas las métricas del embarazo.
        """
        semanas, dias = self.get_semanas_dias()

        metrics = {
            "informacion_gestacional": {
                "semanas_gestacion": semanas,
                "dias_gestacion": dias,
                "fecha_parto": self.fecha_parto.strftime("%Y-%m-%d"),
                "fecha_concepcion": self.fecha_concepcion.strftime("%Y-%m-%d"),
            },
            "info_por_trimestre": self.info_gestational(),
            "estadisticas_tiempo_trimestres": self.time_statistics_trimestres(),
        }

        for name, obj in [
            ("primer_trimestre", self.primer_trimestre),
            ("segundo_trimestre", self.segundo_trimestre),
            ("tercer_trimestre", self.tercer_trimestre),
        ]:
            if obj:
                metrics[f"metricas_basicas_{name}"] = {
                    "GMI": obj.gmi(),
                    "Media": obj.mean(),
                    "Mediana": obj.median(),
                    "Desviacion_estandar": obj.sd(),
                    "CV": obj.cv(),
                }

        return metrics

    def __str__(self) -> str:
        """
        Representación en string del objeto mostrando las semanas de gestación.
        """
        semanas, dias = self.get_semanas_dias()
        info_gest = self.info_gestational()

        output = [f"Gestión: {semanas}+{dias} semanas\n"]

        num_datos_total = len(self.data)
        info_total = self.info()
        intervalo_tipico = info_total["intervalo_tipico"]
        duracion_embarazo = (self.fecha_parto - self.fecha_concepcion).total_seconds() / 60
        datos_teoricos_embarazo = int(duracion_embarazo / intervalo_tipico)
        disponibilidad_real = (num_datos_total / datos_teoricos_embarazo) * 100

        output.append(f"GMI del embarazo: {self.gmi():.1f}%")
        output.append("Información básica del CGM:")
        output.append(f"  - Número de datos: {num_datos_total:,}")
        output.append(
            f"  - Período teórico completo: {self.fecha_concepcion.strftime('%d/%m/%Y')} - {self.fecha_parto.strftime('%d/%m/%Y')}"
        )
        output.append(
            f"  - Período real con datos: {info_total['fecha_inicio'].strftime('%d/%m/%Y')} - {info_total['fecha_fin'].strftime('%d/%m/%Y')}"
        )
        output.append(f"  - Intervalo típico: {intervalo_tipico:.1f} minutos")
        output.append(f"  - Datos esperados (embarazo completo): {datos_teoricos_embarazo:,}")
        output.append(f"  - Disponibilidad real: {disponibilidad_real:.1f}%")
        output.append(f"  - Desconexiones: {info_total['num_desconexiones'].split(' ')[0]}")
        output.append(f"  - Tiempo total sin datos: {info_total['tiempo_total_desconexion']:.1f} horas\n")

        for trimestre, datos in info_gest.items():
            output.append(f"\n=== {trimestre.upper().replace('_', ' ')} ===")
            if isinstance(datos, dict):
                output.append(f"Número de datos: {datos.get('num_datos', 'No disponible'):,}")
                output.append(f"Datos esperados: {datos.get('datos_esperados', 'No disponible'):,}")
                output.append(f"Porcentaje de datos: {datos.get('porcentaje_datos', 'No disponible'):.1f}%")
                output.append(f"Intervalo típico: {datos.get('intervalo_tipico', 'No disponible')} minutos")

                if trimestre == "primer_trimestre":
                    periodo_inicio, periodo_fin = self.fecha_concepcion, self.primer_trimestre_fin
                elif trimestre == "segundo_trimestre":
                    periodo_inicio, periodo_fin = self.primer_trimestre_fin, self.segundo_trimestre_fin
                else:
                    periodo_inicio, periodo_fin = self.segundo_trimestre_fin, self.fecha_parto

                output.append(f"Período: {periodo_inicio.strftime('%d/%m/%Y')} - {periodo_fin.strftime('%d/%m/%Y')}")
            else:
                output.append(str(datos))
            output.append("-" * 50)

        return "\n".join(output)


class GestationalDiabetes(PregnancyMetrics, PregnancyData, ModularGlucoseMetrics):
    """
    Clase principal para análisis de diabetes gestacional.
    Combina la lógica de datos de PregnancyData con las métricas de PregnancyMetrics.
    """

    def __init__(
        self,
        data_source: Union[str, pd.DataFrame],
        fecha_parto: str,
        week: int,
        day: int = 0,
        date_col: str = "time",
        glucose_col: str = "glucose",
        delimiter: Union[str, None] = None,
        header: int = 0,
        start_date: Union[str, datetime.datetime, None] = None,
        end_date: Union[str, datetime.datetime, None] = None,
        log: bool = False,
    ):
        # Inicializar PregnancyData (que a su vez inicializa ModularGlucoseData y gestiona trimestres)
        super().__init__(
            data_source=data_source,
            fecha_parto=fecha_parto,
            week=week,
            day=day,
            date_col=date_col,
            glucose_col=glucose_col,
            delimiter=delimiter,
            header=header,
            start_date=start_date,
            end_date=end_date,
            log=log,
        )

        # Inicializar PregnancyMetrics (mixin) para configurar los objetos de métricas por trimestre
        # Estos objetos usarán los dataframes ya creados por el constructor de PregnancyData
        self._init_pregnancy_metrics()
