"""
Módulo de métricas específicas para diabetes gestacional.

Este módulo contiene las métricas y análisis específicos para el seguimiento
de la diabetes durante el embarazo, incluyendo análisis por trimestres.
"""

import datetime
import pandas as pd
from typing import Union, Dict, Any, Tuple
import numpy as np

from ..data.core import ModularGlucoseData


class PregnancyMetrics:
    """
    Clase para métricas específicas de diabetes gestacional.
    
    Esta clase debe ser utilizada como mixin con ModularGlucoseData.
    """
    
    def __init__(self, 
                 fecha_parto: str,
                 week: int,
                 day: int = 0):
        """
        Inicializa las métricas de embarazo.
        
        Args:
            fecha_parto: Fecha esperada del parto en formato 'YYYY-MM-DD'
            week: Número de semanas de gestación (ej: 38)
            day: Número de días adicionales (0-6) (ej: para 38+4, day=4)
        """
        # Convertir y validar la fecha de parto
        self.fecha_parto = pd.to_datetime(fecha_parto)
        if pd.isna(self.fecha_parto):
            raise ValueError("Fecha de parto inválida")

        self.semana_gestacion = week + (day / 7)
        
        # Calcular y validar la fecha de concepción
        self.fecha_concepcion = self.fecha_parto - pd.Timedelta(weeks=self.semana_gestacion)
        if pd.isna(self.fecha_concepcion):
            raise ValueError("Fecha de concepción inválida")
        
        # Calcular y validar las fechas de los trimestres
        self.primer_trimestre_fin = self.fecha_concepcion + pd.Timedelta(weeks=13)
        self.segundo_trimestre_fin = self.fecha_concepcion + pd.Timedelta(weeks=26)
        
        # Crear los DataFrames por trimestre
        self.primer_trimestre_df = self._create_trimester_df(self.fecha_concepcion, self.primer_trimestre_fin)
        self.segundo_trimestre_df = self._create_trimester_df(self.primer_trimestre_fin, self.segundo_trimestre_fin)
        self.tercer_trimestre_df = self._create_trimester_df(self.segundo_trimestre_fin, self.fecha_parto)
        
        # Crear instancias de análisis por trimestre
        self.primer_trimestre = None
        self.segundo_trimestre = None
        self.tercer_trimestre = None
        
        if len(self.primer_trimestre_df) > 0:
            self.primer_trimestre = ModularGlucoseData(
                data_source=self.primer_trimestre_df,
                date_col="time",
                glucose_col="glucose"
            )
        
        if len(self.segundo_trimestre_df) > 0:
            self.segundo_trimestre = ModularGlucoseData(
                data_source=self.segundo_trimestre_df,
                date_col="time",
                glucose_col="glucose"
            )
        
        if len(self.tercer_trimestre_df) > 0:
            self.tercer_trimestre = ModularGlucoseData(
                data_source=self.tercer_trimestre_df,
                date_col="time",
                glucose_col="glucose"
            )

    def _create_trimester_df(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        """
        Crea un DataFrame para un trimestre dado.
        
        Args:
            start_date: Fecha de inicio del trimestre
            end_date: Fecha de fin del trimestre
            
        Returns:
            DataFrame con los datos del trimestre
        """
        return self.data[
            (self.data['time'] >= start_date) & 
            (self.data['time'] < end_date)
        ].copy()
    
    @staticmethod
    def decimal_a_semanas_dias(semanas_decimal: float) -> Tuple[int, int]:
        """
        Convierte semanas en formato decimal a tupla de (semanas, días).
        
        Args:
            semanas_decimal: Número de semanas en formato decimal (ej: 38.57)
            
        Returns:
            tuple: (semanas, días) (ej: (38, 4))
        """
        semanas = int(semanas_decimal)
        dias = round((semanas_decimal - semanas) * 7)
        return semanas, dias
    
    def get_semanas_dias(self) -> Tuple[int, int]:
        """
        Retorna las semanas y días de gestación en formato tradicional.
        
        Returns:
            tuple: (semanas, días) (ej: (38, 4))
        """
        return self.decimal_a_semanas_dias(self.semana_gestacion)
    
    def info_gestational(self) -> Dict[str, Any]:
        """
        Obtiene información detallada por trimestre.
        
        Returns:
            dict: Información de cada trimestre
        """
        info_primer_trimestre = "No hay datos disponibles"
        info_segundo_trimestre = "No hay datos disponibles"
        info_tercer_trimestre = "No hay datos disponibles"
        
        # Primer trimestre
        if len(self.primer_trimestre_df) > 0:
            info_primer_trimestre = self.primer_trimestre.info()
            intervalo_datos_1T = info_primer_trimestre['intervalo_tipico']
            dias_1T = (self.primer_trimestre_fin - self.fecha_concepcion).days
            datos_esperados_1T = int((60 / intervalo_datos_1T) * 24 * dias_1T)
            info_primer_trimestre['datos_esperados'] = datos_esperados_1T
            porcentaje_datos_1T = info_primer_trimestre['num_datos'] / datos_esperados_1T * 100
            info_primer_trimestre['porcentaje_datos'] = porcentaje_datos_1T
        
        # Segundo trimestre
        if len(self.segundo_trimestre_df) > 0:
            info_segundo_trimestre = self.segundo_trimestre.info()
            intervalo_datos_2T = info_segundo_trimestre['intervalo_tipico']
            dias_2T = (self.segundo_trimestre_fin - self.primer_trimestre_fin).days
            datos_esperados_2T = int((60 / intervalo_datos_2T) * 24 * dias_2T)
            info_segundo_trimestre['datos_esperados'] = datos_esperados_2T
            porcentaje_datos_2T = info_segundo_trimestre['num_datos'] / datos_esperados_2T * 100
            info_segundo_trimestre['porcentaje_datos'] = porcentaje_datos_2T
        
        # Tercer trimestre
        if len(self.tercer_trimestre_df) > 0:
            info_tercer_trimestre = self.tercer_trimestre.info()
            intervalo_datos_3T = info_tercer_trimestre['intervalo_tipico']
            dias_3T = (self.fecha_parto - self.segundo_trimestre_fin).days
            datos_esperados_3T = int((60 / intervalo_datos_3T) * 24 * dias_3T)
            info_tercer_trimestre['datos_esperados'] = datos_esperados_3T
            porcentaje_datos_3T = info_tercer_trimestre['num_datos'] / datos_esperados_3T * 100
            info_tercer_trimestre['porcentaje_datos'] = porcentaje_datos_3T
        
        return {
            'primer_trimestre': info_primer_trimestre,
            'segundo_trimestre': info_segundo_trimestre,
            'tercer_trimestre': info_tercer_trimestre
        }
    
    def time_statistics_trimestres(self) -> Dict[str, Any]:
        """
        Calcula estadísticas de tiempo en rango por trimestre.
        
        Returns:
            dict: Estadísticas de tiempo por trimestre
        """
        stats = {}
        
        if self.primer_trimestre:
            stats['primer_trimestre'] = {
                'TIR': self.primer_trimestre.TIR(),
                'TIR_tight': self.primer_trimestre.TIR_tight(),
                'TBR70': self.primer_trimestre.TBR70(),
                'TBR55': self.primer_trimestre.TBR55(),
                'TAR140': self.primer_trimestre.TAR140(),
                'TAR180': self.primer_trimestre.TAR180(),
                'TAR250': self.primer_trimestre.TAR250()
            }
        
        if self.segundo_trimestre:
            stats['segundo_trimestre'] = {
                'TIR': self.segundo_trimestre.TIR(),
                'TIR_tight': self.segundo_trimestre.TIR_tight(),
                'TBR70': self.segundo_trimestre.TBR70(),
                'TBR55': self.segundo_trimestre.TBR55(),
                'TAR140': self.segundo_trimestre.TAR140(),
                'TAR180': self.segundo_trimestre.TAR180(),
                'TAR250': self.segundo_trimestre.TAR250()
            }
        
        if self.tercer_trimestre:
            stats['tercer_trimestre'] = {
                'TIR': self.tercer_trimestre.TIR(),
                'TIR_tight': self.tercer_trimestre.TIR_tight(),
                'TBR70': self.tercer_trimestre.TBR70(),
                'TBR55': self.tercer_trimestre.TBR55(),
                'TAR140': self.tercer_trimestre.TAR140(),
                'TAR180': self.tercer_trimestre.TAR180(),
                'TAR250': self.tercer_trimestre.TAR250()
            }
        
        return stats
    
    def calculate_all_metrics(self) -> Dict[str, Any]:
        """
        Calcula todas las métricas del embarazo.
        
        Returns:
            dict: Todas las métricas organizadas por trimestre
        """
        semanas, dias = self.get_semanas_dias()
        
        metrics = {
            'informacion_gestacional': {
                'semanas_gestacion': semanas,
                'dias_gestacion': dias,
                'fecha_parto': self.fecha_parto.strftime('%Y-%m-%d'),
                'fecha_concepcion': self.fecha_concepcion.strftime('%Y-%m-%d')
            },
            'info_por_trimestre': self.info_gestational(),
            'estadisticas_tiempo_trimestres': self.time_statistics_trimestres()
        }
        
        # Agregar métricas básicas por trimestre si están disponibles
        for trimestre_name, trimestre_obj in [
            ('primer_trimestre', self.primer_trimestre),
            ('segundo_trimestre', self.segundo_trimestre),
            ('tercer_trimestre', self.tercer_trimestre)
        ]:
            if trimestre_obj:
                metrics[f'metricas_basicas_{trimestre_name}'] = {
                    'GMI': trimestre_obj.gmi(),
                    'Media': trimestre_obj.mean(),
                    'Mediana': trimestre_obj.median(),
                    'Desviacion_estandar': trimestre_obj.sd(),
                    'CV': trimestre_obj.cv()
                }
        
        return metrics
    
    def __str__(self) -> str:
        """
        Representación en string del objeto mostrando las semanas de gestación.
        """
        semanas, dias = self.get_semanas_dias()
        info_gest = self.info_gestational()
        
        output = [f"Gestión: {semanas}+{dias} semanas\n"]
        
        # Calcular la disponibilidad real considerando todo el embarazo
        num_datos_total = len(self.data)
        intervalo_tipico = self.info()['intervalo_tipico']
        duracion_embarazo = (self.fecha_parto - self.fecha_concepcion).total_seconds() / 60  # en minutos
        datos_teoricos_embarazo = int(duracion_embarazo / intervalo_tipico)
        disponibilidad_real = (num_datos_total / datos_teoricos_embarazo) * 100

        # Formatear la información básica
        output.append(f"GMI del embarazo: {self.gmi():.1f}%")
        output.append("Información básica del CGM:")
        output.append(f"  - Número de datos: {num_datos_total:,}")
        output.append(f"  - Período teórico completo: {self.fecha_concepcion.strftime('%d/%m/%Y')} - {self.fecha_parto.strftime('%d/%m/%Y')}")
        output.append(f"  - Período real con datos: {self.info()['fecha_inicio'].strftime('%d/%m/%Y')} - {self.info()['fecha_fin'].strftime('%d/%m/%Y')}")
        output.append(f"  - Intervalo típico: {intervalo_tipico:.1f} minutos")
        output.append(f"  - Datos esperados (embarazo completo): {datos_teoricos_embarazo:,}")
        output.append(f"  - Disponibilidad real: {disponibilidad_real:.1f}%")
        output.append(f"  - Desconexiones: {self.info()['num_desconexiones'].split(' ')[0]}")
        output.append(f"  - Tiempo total sin datos: {self.info()['tiempo_total_desconexion']:.1f} horas\n")

        for trimestre, datos in info_gest.items():
            output.append(f"\n=== {trimestre.upper().replace('_', ' ')} ===")
            if isinstance(datos, dict):
                output.append(f"Número de datos: {datos.get('num_datos', 'No disponible'):,}")
                output.append(f"Datos esperados: {datos.get('datos_esperados', 'No disponible'):,}")
                output.append(f"Porcentaje de datos: {datos.get('porcentaje_datos', 'No disponible'):.1f}%")
                output.append(f"Intervalo típico: {datos.get('intervalo_tipico', 'No disponible')} minutos")
                
                # Añadir los períodos correctos según el trimestre
                if trimestre == "primer_trimestre":
                    periodo_inicio = self.fecha_concepcion
                    periodo_fin = self.primer_trimestre_fin
                elif trimestre == "segundo_trimestre":
                    periodo_inicio = self.primer_trimestre_fin
                    periodo_fin = self.segundo_trimestre_fin
                else:  # tercer_trimestre
                    periodo_inicio = self.segundo_trimestre_fin
                    periodo_fin = self.fecha_parto
                    
                output.append(f"Período: {periodo_inicio.strftime('%d/%m/%Y')} - {periodo_fin.strftime('%d/%m/%Y')}")
            else:
                output.append(str(datos))
            output.append("-" * 50)
        
        return "\n".join(output)


class GestationalDiabetes(ModularGlucoseData, PregnancyMetrics):
    """
    Clase principal para análisis de diabetes gestacional.
    
    Combina la funcionalidad de ModularGlucoseData con las métricas específicas
    de embarazo.
    """
    
    def __init__(self, 
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
                 log: bool = False):
        """
        Inicializa el análisis de diabetes gestacional.
        
        Args:
            data_source: Fuente de datos (archivo o DataFrame)
            fecha_parto: Fecha esperada del parto en formato 'YYYY-MM-DD'
            week: Número de semanas de gestación (ej: 38)
            day: Número de días adicionales (0-6) (ej: para 38+4, day=4)
            date_col: Nombre de la columna de fecha
            glucose_col: Nombre de la columna de glucosa
            delimiter: Delimitador del archivo
            header: Número de fila que contiene los encabezados
            start_date: Fecha de inicio para filtrar datos
            end_date: Fecha de fin para filtrar datos
            log: Si activar logs detallados
        """
        # Inicializar ModularGlucoseData
        super().__init__(
            data_source=data_source,
            date_col=date_col,
            glucose_col=glucose_col,
            delimiter=delimiter,
            header=header,
            start_date=start_date,
            end_date=end_date,
            log=log
        )
        
        # Inicializar PregnancyMetrics
        PregnancyMetrics.__init__(self, fecha_parto, week, day) 