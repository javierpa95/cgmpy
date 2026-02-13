"""
Módulo para el manejo de datos específicos de embarazo.
"""

import datetime
from typing import Dict, Tuple, Union

import pandas as pd

from .core import ModularGlucoseData


class PregnancyData(ModularGlucoseData):
    """
    Clase para la gestión de datos y fechas específicas de embarazo.
    Hereda de ModularGlucoseData para integrar la carga y procesamiento base,
    añadiendo la segmentación por trimestres.
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
        """
        Inicializa los datos de embarazo.
        """
        # Inicialización base de ModularGlucoseData
        super().__init__(
            data_source=data_source,
            date_col=date_col,
            glucose_col=glucose_col,
            delimiter=delimiter,
            header=header,
            start_date=start_date,
            end_date=end_date,
            log=log,
        )

        # Configuración específica de embarazo
        self.gestational_info = self.calculate_dates(fecha_parto, week, day)

        # Mapeo de atributos para fácil acceso
        self.fecha_parto = self.gestational_info["fecha_parto"]
        self.fecha_concepcion = self.gestational_info["fecha_concepcion"]
        self.semana_gestacion = self.gestational_info["semana_gestacion_decimal"]
        self.primer_trimestre_fin = self.gestational_info["primer_trimestre_fin"]
        self.segundo_trimestre_fin = self.gestational_info["segundo_trimestre_fin"]

        # Creación de los dataframes por trimestre
        # Esto permite que si el usuario quiere usar otro framework para analizar
        # los trimestres, ya los tiene disponibles como dataframes.
        self.trimesters = self._split_trimesters()

    @staticmethod
    def calculate_dates(fecha_parto: str, week: int, day: int = 0) -> dict:
        """
        Calcula las fechas clave del embarazo.
        """
        fecha_parto_dt = pd.to_datetime(fecha_parto)
        if pd.isna(fecha_parto_dt):
            raise ValueError("Fecha de parto inválida")

        semana_gestacion = week + (day / 7)
        fecha_concepcion = fecha_parto_dt - pd.Timedelta(weeks=semana_gestacion)

        if pd.isna(fecha_concepcion):
            raise ValueError("Fecha de concepción inválida")

        primer_trimestre_fin = fecha_concepcion + pd.Timedelta(weeks=13, days=6)
        segundo_trimestre_fin = fecha_concepcion + pd.Timedelta(weeks=27, days=6)

        return {
            "fecha_parto": fecha_parto_dt,
            "fecha_concepcion": fecha_concepcion,
            "primer_trimestre_fin": primer_trimestre_fin,
            "segundo_trimestre_fin": segundo_trimestre_fin,
            "semana_gestacion_decimal": semana_gestacion,
        }

    def _split_trimesters(self) -> Dict[str, pd.DataFrame]:
        """
        Divide el dataframe principal en tres dataframes, uno por trimestre.
        """
        return {
            "primer_trimestre": self.get_trimester_data(self.fecha_concepcion, self.primer_trimestre_fin),
            "segundo_trimestre": self.get_trimester_data(self.primer_trimestre_fin, self.segundo_trimestre_fin),
            "tercer_trimestre": self.get_trimester_data(self.segundo_trimestre_fin, self.fecha_parto),
        }

    def get_trimester_data(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        """
        Filtra el DataFrame para un rango de fechas.
        """
        return self.data[(self.data["time"] >= start_date) & (self.data["time"] < end_date)].copy()

    @staticmethod
    def decimal_to_weeks_days(semanas_decimal: float) -> Tuple[int, int]:
        """
        Convierte semanas decimales a (semanas, días).
        """
        semanas = int(semanas_decimal)
        dias = round((semanas_decimal - semanas) * 7)
        return semanas, dias

    def get_semanas_dias(self) -> Tuple[int, int]:
        """
        Retorna las semanas y días de gestación en formato tradicional.
        """
        return self.decimal_to_weeks_days(self.semana_gestacion)


# Alias para mantener compatibilidad si fuera necesario (opcional)
PregnancyDataHandler = PregnancyData
