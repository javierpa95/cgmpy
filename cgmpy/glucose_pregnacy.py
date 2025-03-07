import datetime
from .glucose_analysis import GlucoseAnalysis
from typing import Union
import pandas as pd


class GestationalDiabetes(GlucoseAnalysis):
    def __init__(self, 
                 file_path: str, 
                 fecha_parto: str,
                 week: int,
                 day: int = 0,
                 date_col: str="time", 
                 glucose_col: str="glucose", 
                 delimiter: Union[str, None] = None, 
                 header: int = 0,
                 start_date: Union[str, datetime.datetime, None] = None,
                 end_date: Union[str, datetime.datetime, None] = None, log: bool = False):
        """
        Inicializa el objeto GestationalDiabetes.
        
        Args:
            fecha_parto: Fecha esperada del parto en formato 'YYYY-MM-DD'
            week: Número de semanas de gestación (ej: 38)
            day: Número de días adicionales (0-6) (ej: para 38+4, day=4)
            date_col: Nombre de la columna de fecha
            glucose_col: Nombre de la columna de glucosa
            delimiter: Delimitador del archivo
            header: Número de fila que contiene los encabezados
        """
        super().__init__(file_path, date_col, glucose_col, delimiter, header, start_date, end_date, log)

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
        
        # Asegurarse de que las fechas en self.data son datetime
        self.data[self.date_col] = pd.to_datetime(self.data[self.date_col])
        
        # Crear los DataFrames por trimestre con validación
        self.primer_trimestre_df = self._create_trimester_df(self.fecha_concepcion, self.primer_trimestre_fin)
        self.segundo_trimestre_df = self._create_trimester_df(self.primer_trimestre_fin, self.segundo_trimestre_fin)
        self.tercer_trimestre_df = self._create_trimester_df(self.segundo_trimestre_fin, self.fecha_parto)
        
        # Solo crear instancias si hay datos
        self.primer_trimestre = None
        self.segundo_trimestre = None
        self.tercer_trimestre = None
        
        if len(self.primer_trimestre_df) > 0:
            self.primer_trimestre = GlucoseAnalysis(
                data_source=self.primer_trimestre_df,
                date_col=self.date_col,
                glucose_col=self.glucose_col
            )
        
        if len(self.segundo_trimestre_df) > 0:
            self.segundo_trimestre = GlucoseAnalysis(
                data_source=self.segundo_trimestre_df,
                date_col=self.date_col,
                glucose_col=self.glucose_col
            )
        
        if len(self.tercer_trimestre_df) > 0:
            self.tercer_trimestre = GlucoseAnalysis(
                data_source=self.tercer_trimestre_df,
                date_col=self.date_col,
                glucose_col=self.glucose_col
            )

    def _create_trimester_df(self, start_date, end_date):
        """Crea un DataFrame para un trimestre dado, incluyendo fechas sin datos."""
        df = self.data[
            (self.data[self.date_col] >= start_date) & 
            (self.data[self.date_col] < end_date)
        ].copy()  # Usar .copy() para evitar SettingWithCopyWarning
        
        
        return df

    def __str__(self) -> str:
        """
        Representación en string del objeto mostrando las semanas de gestación y la información por trimestre
        """
        semanas, dias = self.get_semanas_dias()
        info_gest = self.info_gestational()
        
        output = [f"Gestación: {semanas}+{dias} semanas\n"]
        
        gmi = self.gmi()
        info_gestacional = self.info()

        output.append(f"GMI del embarazo: {gmi}%")
        output.append(f"Información básica del CGM: {info_gestacional}")

        for trimestre, datos in info_gest.items():
            output.append(f"\n=== {trimestre.upper().replace('_', ' ')} ===")
            output.append(f"Número de datos: {datos['datos']}")
            output.append(f"{datos['info']}")
            output.append("-" * 50)
        
        return "\n".join(output)
      
    @staticmethod
    def decimal_a_semanas_dias(semanas_decimal: float) -> tuple[int, int]:
        """
        Convierte semanas en formato decimal a tupla de (semanas, días)
        
        Args:
            semanas_decimal: Número de semanas en formato decimal (ej: 38.57)
            
        Returns:
            tuple: (semanas, días) (ej: (38, 4))
        """
        semanas = int(semanas_decimal)
        dias = round((semanas_decimal - semanas) * 7)
        return semanas, dias
    
    def get_semanas_dias(self) -> tuple[int, int]:
        """
        Retorna las semanas y días de gestación en formato tradicional
        
        Returns:
            tuple: (semanas, días) (ej: (38, 4))
        """
        return self.decimal_a_semanas_dias(self.semana_gestacion)


    # INFORMACIÓN BÁSICAAAAAAA
       
    def info_gestational(self):
        # Validar si hay datos en cada trimestre antes de crear las instancias
        info_primer_trimestre = "No hay datos disponibles"
        info_segundo_trimestre = "No hay datos disponibles"
        info_tercer_trimestre = "No hay datos disponibles"
        
        # Solo obtener info si hay datos
        if len(self.primer_trimestre_df) > 0:
            info_primer_trimestre = self.primer_trimestre.info()
            intervalo_datos_1T = info_primer_trimestre['intervalo_tipico']
            # Calcular días del primer trimestre y datos esperados
            dias_1T = (self.primer_trimestre_fin - self.fecha_concepcion).days
            datos_esperados_1T = int((60 / intervalo_datos_1T) * 24 * dias_1T)
            info_primer_trimestre['datos_esperados'] = datos_esperados_1T
            porcentaje_datos_1T = info_primer_trimestre['num_datos'] / datos_esperados_1T*100
            info_primer_trimestre['porcentaje_datos'] = porcentaje_datos_1T
        
        if len(self.segundo_trimestre_df) > 0:
            info_segundo_trimestre = self.segundo_trimestre.info()
            intervalo_datos_2T = info_segundo_trimestre['intervalo_tipico']
            # Calcular días del segundo trimestre y datos esperados
            dias_2T = (self.segundo_trimestre_fin - self.primer_trimestre_fin).days
            datos_esperados_2T = int((60 / intervalo_datos_2T) * 24 * dias_2T)
            info_segundo_trimestre['datos_esperados'] = datos_esperados_2T
            porcentaje_datos_2T = info_segundo_trimestre['num_datos'] / datos_esperados_2T*100
            info_segundo_trimestre['porcentaje_datos'] = porcentaje_datos_2T


        if len(self.tercer_trimestre_df) > 0:
            info_tercer_trimestre = self.tercer_trimestre.info()
            intervalo_datos_3T = info_tercer_trimestre['intervalo_tipico']
            # Calcular días del tercer trimestre y datos esperados
            dias_3T = (self.fecha_parto - self.segundo_trimestre_fin).days
            datos_esperados_3T = int((60 / intervalo_datos_3T) * 24 * dias_3T)
            info_tercer_trimestre['datos_esperados'] = datos_esperados_3T
            porcentaje_datos_3T = info_tercer_trimestre['num_datos'] / datos_esperados_3T*100
            info_tercer_trimestre['porcentaje_datos'] = porcentaje_datos_3T


        return {
            "primer_trimestre": info_primer_trimestre,
            "segundo_trimestre": info_segundo_trimestre,
            "tercer_trimestre": info_tercer_trimestre
        }

    def before_and_after_partum(self):
        """
        Analiza las métricas del día anterior y posterior al parto.
        
        Returns:
            dict: Diccionario con las métricas de ambos períodos
        """
        # Definir los rangos de tiempo
        dia_antes = pd.Timedelta(days=1)
        dia_despues = pd.Timedelta(days=1)
        
        # Filtrar los datos para el día anterior
        datos_preparto = self.data[
            (self.data[self.date_col] >= self.fecha_parto - dia_antes) & 
            (self.data[self.date_col] < self.fecha_parto)
        ].copy()
        
        # Filtrar los datos para el día posterior
        datos_postparto = self.data[
            (self.data[self.date_col] > self.fecha_parto) & 
            (self.data[self.date_col] <= self.fecha_parto + dia_despues)
        ].copy()
        
        # Crear instancias de GlucoseAnalysis para cada período si hay datos
        resultados = {
            'preparto': 'No hay datos disponibles',
            'postparto': 'No hay datos disponibles'
        }
        
        if len(datos_preparto) > 0:
            analisis_preparto = GlucoseAnalysis(
                data_source=datos_preparto,
                date_col=self.date_col,
                glucose_col=self.glucose_col
            )
            resultados['preparto'] = {
                'datos': len(datos_preparto),
                'metricas': analisis_preparto.time_statistics_pregnancy(),
                'distribucion': analisis_preparto.distribution_analysis()
            }
        
        if len(datos_postparto) > 0:
            analisis_postparto = GlucoseAnalysis(
                data_source=datos_postparto,
                date_col=self.date_col,
                glucose_col=self.glucose_col
            )
            resultados['postparto'] = {
                'datos': len(datos_postparto),
                'metricas': analisis_postparto.time_statistics_pregnancy(),
                'distribucion': analisis_postparto.distribution_analysis()
            }
        
        return resultados


    def time_statistics_trimestres(self):
        resultados = {}
        
        if len(self.primer_trimestre_df) > 0:
            resultados['primer_trimestre'] = self.primer_trimestre.time_statistics_pregnancy()
        else:
            resultados['primer_trimestre'] = "No hay datos disponibles"
        
        if len(self.segundo_trimestre_df) > 0:
            resultados['segundo_trimestre'] = self.segundo_trimestre.time_statistics_pregnancy()
        else:
            resultados['segundo_trimestre'] = "No hay datos disponibles"
        
        if len(self.tercer_trimestre_df) > 0:
            resultados['tercer_trimestre'] = self.tercer_trimestre.time_statistics_pregnancy()
        else:
            resultados['tercer_trimestre'] = "No hay datos disponibles"
        
        return resultados
    
    def distribution_analysis_trimestres(self):
        """
        Analiza la distribución de los valores de glucosa por trimestre utilizando
        instancias independientes de GlucoseAnalysis
        """
        return {
            'primer_trimestre': self.primer_trimestre.distribution_analysis(),
            'segundo_trimestre': self.segundo_trimestre.distribution_analysis(),
            'tercer_trimestre': self.tercer_trimestre.distribution_analysis()
        }
    
    def histogram_trimestres(self):
        return {
            'primer_trimestre': self.primer_trimestre.histogram(),
            'segundo_trimestre': self.segundo_trimestre.histogram(),
            'tercer_trimestre': self.tercer_trimestre.histogram()
        }

    def time_statistics_weeks(self):
        """
        Calcula las estadísticas de tiempo para cada semana de embarazo.
        
        Returns:
            dict: Diccionario con las métricas para cada semana de embarazo
            {
                'semana_X': {
                    'datos': número de mediciones,
                    'metricas': resultado de time_statistics_pregnancy()
                }
            }
        """
        resultados = {}
        
        # Calcular la fecha de inicio de cada semana
        fecha_inicial = self.fecha_concepcion
        semana_actual = 1
        
        while fecha_inicial < self.fecha_parto:
            fecha_final = fecha_inicial + pd.Timedelta(weeks=1)
            
            # Filtrar datos para la semana actual
            datos_semana = self.data[
                (self.data[self.date_col] >= fecha_inicial) & 
                (self.data[self.date_col] < fecha_final)
            ].copy()
            
            if len(datos_semana) > 0:
                # Crear instancia de GlucoseAnalysis para la semana
                analisis_semana = GlucoseAnalysis(
                    data_source=datos_semana,
                    date_col=self.date_col,
                    glucose_col=self.glucose_col
                )
                
                resultados[f'semana_{semana_actual}'] = {
                    'datos': len(datos_semana),
                    'metricas': analisis_semana.time_statistics_pregnancy()
                }
            else:
                resultados[f'semana_{semana_actual}'] = {
                    'datos': 0,
                    'metricas': 'No hay datos disponibles'
                }
            
            fecha_inicial = fecha_final
            semana_actual += 1
        
        return resultados
