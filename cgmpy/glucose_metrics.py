import datetime
from .glucose_data import GlucoseData
from typing import Union
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

class GlucoseMetrics(GlucoseData):
    def __init__(self, data_source: Union[str, pd.DataFrame], date_col: str="time", glucose_col: str="glucose", delimiter: Union[str, None] = None, header: int = 0, start_date: Union[str, datetime.datetime, None] = None,
                 end_date: Union[str, datetime.datetime, None] = None):

        super().__init__(data_source, date_col, glucose_col, delimiter, header, start_date, end_date)

    def _calculate_data_completeness(self, interval_minutes: Union[float, None] = None) -> dict:
        """
        Calcula el porcentaje de datos disponibles para el DataFrame actual.
        
        :param interval_minutes: Intervalo esperado entre mediciones en minutos. 
                               Si es None, se calcula automáticamente.
        :return: Diccionario con información sobre la completitud de datos
        """
        # Si no se especifica el intervalo, calcularlo como la mediana de las diferencias
        if interval_minutes is None:
            interval_minutes = self.typical_interval

        # Crear una copia de los datos y ordenarlos
        data = self.data.sort_values('time').copy()
        
        # Análisis para todo el período
        tiempo_total = (data['time'].max() - data['time'].min()).total_seconds() / 60
        datos_esperados = int(tiempo_total / interval_minutes)
        datos_reales = len(data)
        
        return {
            'inicio': data['time'].min(),
            'fin': data['time'].max(),
            'intervalo': interval_minutes,
            'datos_esperados': datos_esperados,
            'datos_reales': datos_reales,
            'porcentaje': (datos_reales / datos_esperados) * 100 if datos_esperados > 0 else 0
        }
    
    def _get_segment_data(self, start_time: str, duration_hours: int) -> pd.DataFrame:
        """
        Función auxiliar para obtener datos de un segmento del día.
        """
        from datetime import time, datetime, timedelta
        
        start_hour, start_minute = map(int, start_time.split(':'))
        start = time(hour=start_hour, minute=start_minute)
        
        base = datetime(2000, 1, 1, start.hour, start.minute)
        end_dt = base + timedelta(hours=duration_hours)
        end = end_dt.time()
        
        if start <= end:
            return self.data[self.data['time'].apply(lambda dt: start <= dt.time() < end)]
        else:
            return self.data[self.data['time'].apply(lambda dt: dt.time() >= start or dt.time() < end)]
    
    ## ESTADÍSTICAS BÁSICAS

    def data_completeness(self, interval_minutes: Union[float, None] = None) -> int:
        """
        Calcula el porcentaje de datos disponibles para el DataFrame actual.
        :return: Porcentaje de completitud como número entero
        """
        return int(self._calculate_data_completeness(interval_minutes)['porcentaje'])
    
    def mean(self) -> float:
        """Calcula la glucemia media."""
        return self.data['glucose'].mean()
    
    def median(self) -> float:
        """Calcula la mediana de la glucemia."""
        return self.data['glucose'].median()

    def percentile(self, percentile: float) -> float:
        """Calcula el percentil de la glucemia."""
        return self.data['glucose'].quantile(percentile / 100)
    
    def sd(self) -> float:
        """Calcula la desviación estándar de la glucemia."""
        return self.data['glucose'].std()
    
    def cv(self) -> float:
        """Calcula el coeficiente de variación."""
        return (self.sd() / self.mean()) * 100
    
    def gmi(self) -> float:
        """
        Calcula el Glucose Management Index (GMI).
        :return: GMI (estimación de HbA1c)
        :reference: DOI: 10.2337/dc18-1581
        """
        return round(3.31 + (0.02392 * self.mean()), 2)
    
    def calculate_time_in_range(self, low_threshold: float, high_threshold: float) -> float:
        """
        Calcula el tiempo en rango (TIR) de glucemia.
        :param low_threshold: Umbral inferior del rango.
        :param high_threshold: Umbral superior del rango.
        :return: Porcentaje de tiempo en rango.
        """
        in_range = self.data[(self.data['glucose'] >= low_threshold) & (self.data['glucose'] <= high_threshold)]
        return (len(in_range) / len(self.data)) * 100
    
    def TAR(self, threshold: float) -> float:
        """
        Calcula el tiempo por encima del rango (TAR).
        :param threshold: Umbral de hiperglucemia.
        :return: Porcentaje de lecturas por encima del umbral.
        """
        return (len(self.data[self.data['glucose'] > threshold]) / len(self.data)) * 100
    
    def TBR(self, threshold: float) -> float:
        """
        Calcula el tiempo por debajo del rango (TBR).
        :param threshold: Umbral de hipoglucemia.
        :return: Porcentaje de lecturas por debajo del umbral.
        """
        return (len(self.data[self.data['glucose'] < threshold]) / len(self.data)) * 100

    def TAR250(self) -> float:
        """Calcula el tiempo por encima de 250 mg/dL."""
        return self.TAR(250)
    
    def TAR180(self) -> float:
        """Calcula el tiempo en rango entre 180 y 250 mg/dL."""
        return self.calculate_time_in_range(180, 250)
    
    def TAR140(self) -> float:
        """Calcula el tiempo por encima de 140 mg/dL."""
        return self.calculate_time_in_range(140, 250)
    
    def TIR(self) -> float:
        """Calcula el tiempo en rango entre 70 y 180 mg/dL."""
        return self.calculate_time_in_range(70, 180)
    
    def TIR_tight(self) -> float:
        """Calcula el tiempo en rango entre 70 y 180 mg/dL."""
        return self.calculate_time_in_range(70, 140)
    
    def TIR_pregnancy(self) -> float:
        """Calcula el tiempo en rango entre 70 y 180 mg/dL."""
        return self.calculate_time_in_range(63, 140)
    
    def TBR70(self) -> float:
        """Calcula el tiempo por debajo de 70 mg/dL."""
        return self.calculate_time_in_range(55, 70)
    
    def TBR63(self) -> float:
        """Calcula el tiempo por debajo de 70 mg/dL."""
        return self.TBR(63)
    
    def TBR55(self) -> float:
        """Calcula el tiempo por debajo de 55 mg/dL."""
        return self.TBR(55)

    def time_statistics(self):
        """Calcula las estadísticas de tiempo de glucosa"""
        return {
            '%Data': int(self._calculate_data_completeness()['porcentaje']),
            'TIR': self.TIR(),
            'TIR_tight': self.TIR_tight(),
            'TBR70': self.TBR70(),
            'TBR55': self.TBR55(),
            'TAR250': self.TAR250(),
            'TAR180': self.TAR180(),
            'TAR140': self.TAR140(),
            'GMI': self.gmi(),
            'CV': self.cv(),
            'Media': self.mean(),
            'Mediana': self.median(),
            'P5': self.percentile(5), 
            'P25': self.percentile(25),
            'P75': self.percentile(75),
            'P95': self.percentile(95),
            'Desviacion_estandar': self.sd(),
            'Asimetria': self.data['glucose'].skew(),
            'Curtosis': self.data['glucose'].kurtosis()
        }
    def time_statistics_pregnancy(self):
        """
        Calcula las estadísticas de tiempo de glucosa específicas para embarazo
        siguiendo las guías internacionales para diabetes gestacional
        """
        return {
            '%Data': int(self._calculate_data_completeness()['porcentaje']),
            'TIR_pregnancy': self.TIR_pregnancy(),  # 63-140 mg/dL
            'TBR63': self.TBR63(),    # < 63 mg/dL
            'TAR140': self.TAR140(),  # > 140 mg/dL 
            'TAR250': self.TAR250(),
            'GMI': self.gmi(),
            'CV': self.cv(),
            'Media': self.mean(),
            'Mediana': self.median(),
            'P5': self.percentile(5), 
            'P25': self.percentile(25),
            'P75': self.percentile(75),
            'P95': self.percentile(95),
            'Desviacion_estandar': self.sd(),
            'Asimetria': self.data['glucose'].skew(),
            'Curtosis': self.data['glucose'].kurtosis()
        }

    def distribution_analysis(self):
        """Analiza la distribución de los valores de glucosa"""
        stats = {
            '%Data': int(self._calculate_data_completeness()['porcentaje']),
            'media': self.mean(),
            'mediana': self.median(),
            'desviacion_estandar': self.sd(),
            'coef_variacion': self.cv(),
            'asimetria': self.data['glucose'].skew(),
            'curtosis': self.data['glucose'].kurtosis(),
            'percentiles': {
                'p25': self.percentile(25),
                'p75': self.percentile(75),
                'IQR': self.percentile(75) - self.percentile(25)
            }
        }
        return stats
        
    ## VARIABILIDAD

    ## DISTINTAS MEDIDAS DE LA DESVIACIÓN ESTÁNDAR

    """DOI: 10.1089/dia.2009/0015 """

    def sd_total(self) -> dict:
        """
        Calcula la desviación estándar total (SDT) y media global.
        Devuelve: {'sd': float, 'mean': float}
        """
        return {'sd': self.sd(), 'mean': self.mean()}
    
    def sd_within_day(self) -> dict:
        """
        Calcula la desviación estándar dentro del día (SDw) como el promedio de las 
        desviaciones estándar de cada día individual  y media de medias diarias.
        Devuelve: {'sd': float, 'mean': float}
        """
        daily_stats = self.data.groupby(self.data['time'].dt.date)['glucose'].agg(['std', 'mean'])
        return {
            'sd': daily_stats['std'].mean() if not daily_stats.empty else 0.0,
            'mean': daily_stats['mean'].mean() if not daily_stats.empty else 0.0
        }
    
    def sd_within_day_segment(self, start_time: str, duration_hours: int) -> dict:
        """
        Calcula la desviación estándar within-day para un segmento específico del día.
        Para cada día, calcula la SD del segmento horario especificado y luego
        promedia estas SD diarias.
        
        :param start_time: Hora de inicio en formato "HH:MM"
        :param duration_hours: Duración del segmento en horas
        :return: Promedio de las SD del segmento de cada día
        
        Ejemplo:
            sd_within_day_segment("00:00", 8)  # SD promedio del segmento nocturno (00:00-08:00)
            sd_within_day_segment("08:00", 8)  # SD promedio del segmento diurno (08:00-16:00)
        """
        # Obtener los datos del segmento
        segment_data = self._get_segment_data(start_time, duration_hours)
        
        if segment_data.empty:
            return {'sd': 0.0, 'mean': 0.0}
        
        # Calcular SD para el segmento de cada día
        daily_segment_sds = segment_data.groupby(segment_data['time'].dt.date)['glucose'].std()
        
        return {
            'sd': daily_segment_sds.mean() if not daily_segment_sds.empty else 0.0,
            'mean': segment_data['glucose'].mean() if not segment_data.empty else 0.0
        }

    def sd_timepoint_pattern(self) -> dict:
        """
        Calcula la desviación estándar del patrón promedio por tiempo del día (SDhh:mm).
        Agrupa los datos por la marca de tiempo exacta en formato "HH:MM" y calcula la 
        desviación estándar de los promedios de glucosa correspondientes a esos momentos a 
        lo largo de los días.
        
        Calcula SDhh:mm y media de los promedios temporales.
        Devuelve: {'sd': float, 'mean': float}
        """
        # Agrupar por la marca de tiempo "HH:MM"
        time_avg = self.data.groupby(self.data['time'].dt.strftime("%H:%M"))['glucose'].mean()
        return {
            'sd': time_avg.std() if not time_avg.empty else 0.0,
            'mean': time_avg.mean() if not time_avg.empty else 0.0
        }
    
    def sd_timepoint_pattern_segment(self, start_time: str, duration_hours: int) -> dict:
        """
        Calcula SDhh:mm para un segmento específico del día.
        
        :param start_time: Hora de inicio en formato "HH:MM"
        :param duration_hours: Duración del segmento en horas
        :return: SD del patrón promedio por tiempo del día en el segmento
        """
        # Filtrar el segmento primero
        segment_data = self._get_segment_data(start_time, duration_hours)
        
        # Agrupar por marca de tiempo "HH:MM" dentro del segmento
        time_avg = segment_data.groupby(segment_data['time'].dt.strftime("%H:%M"))['glucose'].mean()
        return {
            'sd': time_avg.std() if not time_avg.empty else 0.0,
            'mean': time_avg.mean() if not time_avg.empty else 0.0
        }

    def sd_within_series(self, hours: int = 1) -> dict:
        """
        Calcula SDws y media de las series temporales.
        Devuelve: {'sd': float, 'mean': float}
        """
        # Convertir horas a intervalos según la frecuencia de los datos
        data = self.data.sort_values('time')
        series_stats = []
        
        # Crear ventanas móviles del tamaño especificado
        for start_time in data['time']:
            end_time = start_time + pd.Timedelta(hours=hours)
            series = data[(data['time'] >= start_time) & (data['time'] < end_time)]['glucose']
            if len(series) > 1:
                series_stats.append({'sd': series.std(), 'mean': series.mean()})
        
        return {
            'sd': np.mean([s['sd'] for s in series_stats]) if series_stats else 0.0,
            'mean': np.mean([s['mean'] for s in series_stats]) if series_stats else 0.0
        }
    
    def sd_daily_mean(self) -> dict:
        """
        Calcula la desviación estándar de los promedios diarios (SDdm).
        """
        daily_means = self.data.groupby(self.data['time'].dt.date)['glucose'].mean()
        return {
            'sd': daily_means.std() if not daily_means.empty else 0.0,
            'mean': daily_means.mean() if not daily_means.empty else 0.0
        }

    def sd_same_timepoint(self) -> dict:
        """
        Calcula la SD entre días para cada punto temporal ($SD_{b,hh:mm}$).
        Para cada tiempo específico del día (HH:MM), calcula la SD entre días,
        luego promedia todas estas SD.
        
        Returns:
            dict: Promedio de las desviaciones estándar calculadas para cada tiempo específico del día.
        """
        # Agrupar por tiempo específico del día (HH:MM)
        time_groups = self.data.groupby(self.data['time'].dt.strftime("%H:%M"))
        
        # Calcular SD para cada tiempo específico
        time_sds = time_groups['glucose'].std()
        
        # Promediar todas las SD
        return {
            'sd': time_sds.mean() if not time_sds.empty else 0.0,
            'mean': self.data['glucose'].mean() if not time_sds.empty else 0.0
        }
       
    def sd_same_timepoint_adjusted(self) -> dict:
        """
        Calcula la SD entre días para cada punto temporal, después de corregir por cambios en las medias diarias.
        
        El proceso es:
        1. Ajustar los valores de glucosa: Glucosa_ajustada = Glucosa - Media_diaria + Media_total
        2. Calcular la SD entre días para cada punto temporal usando los valores ajustados
        3. Promediar las SD resultantes
        
        Returns:
            dict: SD entre días ajustada por medias diarias
        """
        # Calcular la media total (Grand Mean)
        grand_mean = self.data['glucose'].mean()
        
        # Calcular medias diarias
        daily_means = self.data.groupby(self.data['time'].dt.date)['glucose'].transform('mean')
        
        # Ajustar valores de glucosa
        adjusted_glucose = self.data['glucose'] - daily_means + grand_mean
        
        # Crear DataFrame con valores ajustados
        adjusted_data = self.data.copy()
        adjusted_data['glucose'] = adjusted_glucose
        
        # Agrupar por tiempo específico del día (HH:MM)
        time_groups = adjusted_data.groupby(adjusted_data['time'].dt.strftime("%H:%M"))
        
        # Calcular SD para cada tiempo específico usando valores ajustados
        time_sds = time_groups['glucose'].std()
        
        # Promediar todas las SD
        return {
            'sd': time_sds.mean() if not time_sds.empty else 0.0,
            'mean': grand_mean if not time_sds.empty else 0.0
        }

    def sd_interaction(self) -> dict:
        """
        Calcula la desviación estándar de la interacción entre patrones y días (SDI).

        ESTA NO LA ENTIENDO BIEN LO QUE SE ESTÁ HACIENDO NI SE SI ESTA BIEN IMPLEMENTADA LEYENDO EL ARTÍCULO
        
        :return: SDI calculado mediante ANOVA de dos vías
        """
        # Preparar datos para ANOVA
        data = self.data.copy()
        data['date'] = data['time'].dt.date
        
        # Calcular componentes
        SST = np.sum((data['glucose'] - data['glucose'].mean())**2)
        
        # Variación entre medias diarias
        daily_stats = data.groupby('date').agg({
            'glucose': ['mean', 'count']
        })
        SSdm = sum(row['glucose']['count'] * (row['glucose']['mean'] - data['glucose'].mean())**2 
                   for _, row in daily_stats.iterrows())
        
        # Variación entre puntos en el día (agrupados por "HH:MM")
        timepoint_stats = data.groupby(data['time'].dt.strftime('%H:%M')).agg({
            'glucose': ['mean', 'count']
        })
        SShh = sum(row['glucose']['count'] * (row['glucose']['mean'] - data['glucose'].mean())**2 
                   for _, row in timepoint_stats.iterrows())
        
        # Calcular m como el número medio de observaciones por día
        m = int(round(data.groupby('date').size().mean()))
        d = len(data['date'].unique())
        
        SDI = np.sqrt((SST - SSdm - SShh) / ((m - 1) * (d - 1)))
        return {
            'sd': SDI,
            'mean': data['glucose'].mean() if not data.empty else 0.0
        }

    def sd_segment(self, start_time: str, duration_hours: int) -> dict:
        """
        Calcula la desviación estándar dentro de un segmento específico (SDws).

        Puede estar bien para analizar tramos como noche, día, tarde, etc.
        
        :param start_time: Hora de inicio en formato "HH:MM".
        :param duration_hours: Duración del segmento en horas.
        :return: Desviación estándar de las lecturas dentro del segmento.
        """
        from datetime import datetime, time, timedelta
        
        # Convertir start_time a objeto time
        start_hour, start_minute = map(int, start_time.split(':'))
        start = time(hour=start_hour, minute=start_minute)
        
        # Calcular la hora de fin (considerando cruce de medianoche)
        base = datetime(2000, 1, 1, start.hour, start.minute)
        end_dt = base + timedelta(hours=duration_hours)
        end = end_dt.time()
        
        # Filtrar datos según si el segmento cruza la medianoche o no
        if start <= end:
            segment_data = self.data[self.data['time'].apply(lambda dt: start <= dt.time() < end)]
        else:
            segment_data = self.data[self.data['time'].apply(lambda dt: dt.time() >= start or dt.time() < end)]
        
        return {
            'sd': segment_data['glucose'].std() if not segment_data.empty else 0.0,
            'mean': segment_data['glucose'].mean() if not segment_data.empty else 0.0
        }
    
    def calculate_all_sd_metrics(self) -> dict:
        """
        Calcula todas las métricas de desviación estándar disponibles.
        """
        return {
            'SDT': self.sd_total()['sd'],
            'SDw': self.sd_within_day()['sd'],
            'SDhh:mm': self.sd_timepoint_pattern()['sd'],
            'Noche': self.sd_segment("00:00", 8)['sd'],
            'Día': self.sd_segment("08:00", 8)['sd'],
            'Tarde': self.sd_segment("16:00", 8)['sd'],
            'SDws_1h': self.sd_within_series(hours=1)['sd'],
            'SDws_6h': self.sd_within_series(hours=6)['sd'],
            'SDws_24h': self.sd_within_series(hours=24)['sd'],
            'SDdm': self.sd_daily_mean()['sd'],
            'SDbhh:mm': self.sd_same_timepoint()['sd'],
            'SDbhh:mm_dm': self.sd_same_timepoint_adjusted()['sd'],
            'SDI': self.sd_interaction()['sd']
        }
    
    def calculate_all_cv_metrics(self) -> dict:
        """
        Calcula todas las métricas de coeficiente de variación disponibles.
        """
        return {
            'CVT': self.sd_total()['sd']/self.sd_total()['mean']*100,
            'CVw': self.sd_within_day()['sd']/self.sd_within_day()['mean']*100,
            'CVhh:mm': self.sd_timepoint_pattern()['sd']/self.sd_timepoint_pattern()['mean']*100,
            'CVNoche': self.sd_segment("00:00", 8)['sd']/self.sd_segment("00:00", 8)['mean']*100,
            'CVDía': self.sd_segment("08:00", 8)['sd']/self.sd_segment("08:00", 8)['mean']*100,
            'CVTarde': self.sd_segment("16:00", 8)['sd']/self.sd_segment("16:00", 8)['mean']*100,
            'CVSDws_1h': self.sd_within_series(hours=1)['sd']/self.sd_within_series(hours=1)['mean']*100,
            'CVSDws_6h': self.sd_within_series(hours=6)['sd']/self.sd_within_series(hours=6)['mean']*100,
            'CVSDws_24h': self.sd_within_series(hours=24)['sd']/self.sd_within_series(hours=24)['mean']*100,
            'CVdm': self.sd_daily_mean()['sd']/self.sd_daily_mean()['mean']*100,
            'CVbhh:mm': self.sd_same_timepoint()['sd']/self.sd_same_timepoint()['mean']*100,
            'CVbhh:mm_dm': self.sd_same_timepoint_adjusted()['sd']/self.sd_same_timepoint_adjusted()['mean']*100,
            'CVSDI': self.sd_interaction()['sd']/self.sd_interaction()['mean']*100
        }

    def anova_two_way(self) -> dict:
        """
        Realiza un ANOVA de dos vías para analizar la variabilidad de la glucosa.
        
        Returns:
            dict: Componentes del ANOVA y estadísticas relacionadas
        """
        data = self.data.copy()
        
        # 1. Preparar los datos
        data['date'] = data['time'].dt.date
        data['timepoint'] = data['time'].dt.strftime('%H:%M')
        
        # 2. Calcular medias
        grand_mean = data['glucose'].mean()  # Media general
        
        # Calcular medias y conteos por grupos
        daily_stats = data.groupby('date').agg({
            'glucose': ['mean', 'count']
        })
        time_stats = data.groupby('timepoint').agg({
            'glucose': ['mean', 'count']
        })
        cell_stats = data.groupby(['date', 'timepoint']).agg({
            'glucose': ['mean', 'count']
        })
        
        # 3. Calcular sumas de cuadrados
        # SST - Suma de cuadrados total
        SST = ((data['glucose'] - grand_mean)**2).sum()
        
        # SSdm - Suma de cuadrados de días
        SSdm = sum(row[('glucose', 'count')] * (row[('glucose', 'mean')] - grand_mean)**2 
                   for _, row in daily_stats.iterrows())
        
        # SShh - Suma de cuadrados de momentos
        SShh = sum(row[('glucose', 'count')] * (row[('glucose', 'mean')] - grand_mean)**2 
                   for _, row in time_stats.iterrows())
        
        # SSI - Suma de cuadrados de interacción
        SSI = 0
        for (date, timepoint), row in cell_stats.iterrows():
            daily_mean = daily_stats.loc[date, ('glucose', 'mean')]
            time_mean = time_stats.loc[timepoint, ('glucose', 'mean')]
            SSI += row[('glucose', 'count')] * (row[('glucose', 'mean')] - daily_mean - time_mean + grand_mean)**2
        
        # 4. Calcular grados de libertad
        df_days = len(daily_stats) - 1
        df_time = len(time_stats) - 1
        df_interaction = df_days * df_time
        df_total = len(data) - 1
        
        # 5. Calcular medias cuadráticas
        MS_days = SSdm / df_days if df_days > 0 else 0
        MS_time = SShh / df_time if df_time > 0 else 0
        MS_interaction = SSI / df_interaction if df_interaction > 0 else 0
        
        # 6. Calcular SDI
        SDI = np.sqrt(MS_interaction)
        
        return {
            'SST': SST,
            'SSdm': SSdm,
            'SShh': SShh,
            'SSI': SSI,
            'MS_days': MS_days,
            'MS_time': MS_time,
            'MS_interaction': MS_interaction,
            'SDI': SDI,
            'df_days': df_days,
            'df_time': df_time,
            'df_interaction': df_interaction
        }

    def pattern_stability_metrics(self) -> dict:
        """
        Calcula métricas de estabilidad del patrón glucémico según el artículo.

        Revisar con alguien que domine estadística para comprobar todo!!
        
        Returns:
            dict: Diccionario con las diferentes métricas de estabilidad
        """
        # 1. Ratio SDhh:mm/SDw
        ratio_sd = (self.sd_timepoint_pattern()['sd'] / self.sd_within_day()['sd'])**2
        # 1. Ratio SDhh:mm/SDw por parte del día (corregido)
        ratio_sd_by_part_of_day = {
            "Noche": (self.sd_timepoint_pattern_segment("00:00", 8)['sd'] / self.sd_within_day_segment("00:00", 8)['sd'])**2,
            "Día": (self.sd_timepoint_pattern_segment("08:00", 8)['sd'] / self.sd_within_day_segment("08:00", 8)['sd'])**2,
            "Tarde": (self.sd_timepoint_pattern_segment("16:00", 8)['sd'] / self.sd_within_day_segment("16:00", 8)['sd'])**2
        }
        
        # 2. Correlación entre días
        data_pivot = self.data.pivot_table(
            index=self.data['time'].dt.strftime('%H:%M'),
            columns=self.data['time'].dt.date,
            values='glucose'
        )
        
        # Calcular correlaciones entre días
        correlations = []
        days = data_pivot.columns
        for i in range(len(days)):
            for j in range(i + 1, len(days)):
                day1 = data_pivot[days[i]].dropna()
                day2 = data_pivot[days[j]].dropna()
                common_times = day1.index.intersection(day2.index)
                if len(common_times) > 0:
                    r = np.corrcoef(day1[common_times], day2[common_times])[0,1]
                    correlations.append(np.sign(r) * r**2)  # r|r| como sugiere el artículo
        
        avg_correlation = np.mean(correlations) if correlations else 0
        
        # 3. Correlación con el perfil medio
        mean_profile = data_pivot.mean(axis=1)
        correlations_with_mean = []
        
        for day in days:
            day_data = data_pivot[day].dropna()
            common_times = day_data.index.intersection(mean_profile.index)
            if len(common_times) > 0:
                r = np.corrcoef(day_data[common_times], mean_profile[common_times])[0,1]
                correlations_with_mean.append(np.sign(r) * r**2)
        
        avg_correlation_with_mean = np.mean(correlations_with_mean) if correlations_with_mean else 0
        
        # 4. Error RMS con regresión lineal
        all_observed = []
        all_expected = []
        slopes = []

        for day in days:
            # Obtener datos del día y perfil medio
            day_data = data_pivot[day].dropna()
            mean_data = mean_profile[day_data.index].dropna()
            
            # Asegurarse de que solo usamos tiempos donde tenemos ambos valores
            common_times = day_data.index.intersection(mean_data.index)
            
            if len(common_times) > 1:  # Necesitamos al menos 2 puntos para regresión
                # Filtrar NaN y preparar datos para regresión
                x = mean_data[common_times].values.reshape(-1, 1)
                y = day_data[common_times].values
                
                # Verificar que no hay NaN
                if not np.isnan(x).any() and not np.isnan(y).any():
                    # Regresión lineal forzando intercepto a cero
                    reg = LinearRegression(fit_intercept=False).fit(x, y)
                    slope = reg.coef_[0]
                    slopes.append(slope)
                    
                    # Calcular residuos
                    y_pred = reg.predict(x)
                    all_observed.extend(y)
                    all_expected.extend(y_pred)

        # Calcular métricas
        if all_observed:
            rms_error = np.sqrt(np.mean((np.array(all_observed) - np.array(all_expected))**2))
            avg_slope = np.mean(slopes)
        else:
            rms_error = 0
            avg_slope = 0
        
        # 5. Bootstrap RMS mejorado
        bootstrap_errors = []
        bootstrap_slopes = []

        for i, day in enumerate(days):
            # Calcular perfil medio excluyendo el día actual
            other_days = [d for d in days if d != day]
            mean_profile_bootstrap = data_pivot[other_days].mean(axis=1)
            
            # Obtener datos del día y perfil medio bootstrap
            day_data = data_pivot[day].dropna()
            mean_data = mean_profile_bootstrap[day_data.index].dropna()
            
            # Asegurarse de que solo usamos tiempos donde tenemos ambos valores
            common_times = day_data.index.intersection(mean_data.index)
            
            if len(common_times) > 1:
                # Filtrar NaN y preparar datos para regresión
                x = mean_data[common_times].values.reshape(-1, 1)
                y = day_data[common_times].values
                
                # Verificar que no hay NaN
                if not np.isnan(x).any() and not np.isnan(y).any():
                    # Regresión lineal forzando intercepto a cero
                    reg = LinearRegression(fit_intercept=False).fit(x, y)
                    y_pred = reg.predict(x)
                    
                    bootstrap_errors.extend(y - y_pred)
                    bootstrap_slopes.append(reg.coef_[0])

        bootstrap_rms = np.sqrt(np.mean(np.array(bootstrap_errors)**2)) if bootstrap_errors else 0
        avg_bootstrap_slope = np.mean(bootstrap_slopes) if bootstrap_slopes else 0
        
        return {
            'ratio_SDhh:mm_SDw': ratio_sd,
            'ratio_SDhh:mm_SDw_by_part_of_day': ratio_sd_by_part_of_day,
            'avg_correlation_between_days': avg_correlation,
            'avg_correlation_with_mean': avg_correlation_with_mean,
            'rms_error': rms_error,
            'avg_slope': avg_slope,
            'bootstrap_rms': bootstrap_rms,
            'avg_bootstrap_slope': avg_bootstrap_slope
        }

    
    def MAGE_Baghurst(self, threshold_sd: int = 1, approach: int = 1) -> dict:
        """
        Calcula el MAGE según el algoritmo de Baghurst.
        
        Cambios principales:
        1. Manejo correcto de bordes en el suavizado
        2. Búsqueda de turning points en datos originales entre mínimos/máximos del perfil suavizado
        3. Proceso iterativo de eliminación de puntos no válidos
        4. Manejo de excursiones al inicio/final del dataset
        
        :param threshold_sd: Número de desviaciones estándar para el umbral
        :param approach: 1 para usar suavizado, 2 para eliminación directa
        :return: Diccionario con MAGE+, MAGE- y métricas relacionadas

        Approach 1: Usa suavizado para identificar turning points
        Approach 2: Eliminación directa de puntos intermedios en secuencias monótonas
        """
        glucose = self.data['glucose'].values
        times = self.data['time'].values
        sd = self.sd()
        threshold = threshold_sd * sd
        
        if approach == 1:
            # 1. Aplicar filtro de suavizado con manejo de bordes
            weights = np.array([1,2,4,8,16,8,4,2,1])/46
            smoothed = np.zeros_like(glucose)
            
            # Suavizado central
            for i in range(4, len(glucose)-4):
                smoothed[i] = np.dot(weights, glucose[i-4:i+5])
                
            # Manejo de bordes con media simple
            for i in range(4):
                smoothed[i] = glucose[:i+5].mean()
                smoothed[-(i+1)] = glucose[-(i+5):].mean()
                
            # 2. Identificar mínimos/máximos en el perfil suavizado
            delta = np.diff(smoothed)
            turning_smoothed = np.where(np.diff(np.sign(delta)))[0] + 1
            
            # 3. Buscar turning points reales en datos originales entre los intervalos suavizados
            turning_points = []
            for i in range(len(turning_smoothed)-1):
                start = turning_smoothed[i]
                end = turning_smoothed[i+1]
                
                # Buscar máximo real en intervalo ascendente
                if smoothed[start] < smoothed[end]:
                    true_peak = np.argmax(glucose[start:end]) + start
                    turning_points.append(true_peak)
                # Buscar mínimo real en intervalo descendente
                else:
                    true_valley = np.argmin(glucose[start:end]) + start
                    turning_points.append(true_valley)
                    
            turning_points = np.unique(turning_points)
        else:
            # Approach 2: Eliminación directa
            turning_points = []
            i = 0
            
            # 1. Primera pasada: eliminar puntos intermedios en secuencias monótonas
            while i < len(glucose) - 2:
                if (glucose[i] <= glucose[i+1] <= glucose[i+2]) or \
                   (glucose[i] >= glucose[i+1] >= glucose[i+2]):
                    # El punto intermedio es parte de una secuencia monótona
                    i += 1
                else:
                    # Punto i+1 es un turning point potencial
                    turning_points.append(i+1)
                    i += 2
            
            # Asegurar que incluimos el primer y último punto si son relevantes
            if len(turning_points) == 0 or turning_points[0] > 0:
                turning_points.insert(0, 0)
            if turning_points[-1] < len(glucose) - 1:
                turning_points.append(len(glucose) - 1)
            
            # 2. Segunda pasada: eliminar excursiones que no superan el umbral
            valid_points = []
            for i in range(1, len(turning_points)-1):
                prev_val = glucose[turning_points[i-1]]
                curr_val = glucose[turning_points[i]]
                next_val = glucose[turning_points[i+1]]
                
                # Verificar si es un máximo o mínimo válido
                if ((curr_val > prev_val and curr_val > next_val) or 
                    (curr_val < prev_val and curr_val < next_val)) and (
                    abs(curr_val - prev_val) >= threshold or 
                    abs(curr_val - next_val) >= threshold):
                    valid_points.append(turning_points[i])
            
            # Asegurar que mantenemos puntos inicial y final si son necesarios
            if valid_points and valid_points[0] > 0:
                valid_points.insert(0, 0)
            if valid_points and valid_points[-1] < len(glucose) - 1:
                valid_points.append(len(glucose) - 1)
            
            turning_points = valid_points
        
        # 3. Calcular excursiones válidas
        excursions = []
        last_val = glucose[turning_points[0]]
        
        for point in turning_points[1:]:
            curr_val = glucose[point]
            diff = abs(curr_val - last_val)
            
            if diff >= threshold:
                excursions.append({
                    'start': last_val,
                    'end': curr_val,
                    'type': 'up' if curr_val > last_val else 'down',
                    'magnitude': diff
                })
                last_val = curr_val
        
        # Separar excursiones y calcular métricas
        excursions_up = [e['magnitude'] for e in excursions if e['type'] == 'up']
        excursions_down = [e['magnitude'] for e in excursions if e['type'] == 'down']
        
        mage_plus = np.mean(excursions_up) if excursions_up else 0
        mage_minus = np.mean(excursions_down) if excursions_down else 0
        mage_avg = np.mean(excursions_up + excursions_down) if (excursions_up or excursions_down) else 0
        
        return {
            'MAGE+': round(mage_plus, 2),
            'MAGE-': round(mage_minus, 2),
            'MAGE_avg': round(mage_avg, 2),
            'SD_used': round(sd, 2),
            'threshold': round(threshold, 2),
            'num_excursions': len(excursions)
        }

    def MAGE(self) -> float:
        """
        Calcula el MAGE (Mean Amplitude of Glycemic Excursions).
        :return: Valor de MAGE.
        """
        sd = self.sd()
        peaks_and_nadirs = self.data[
            (self.data['glucose'].shift(1) < self.data['glucose']) & (self.data['glucose'] > self.data['glucose'].shift(-1)) | 
            (self.data['glucose'].shift(1) > self.data['glucose']) & (self.data['glucose'] < self.data['glucose'].shift(-1))
        ].reset_index(drop=True)
        
        excursions = []
        starts_with_peak = peaks_and_nadirs['glucose'][0] > peaks_and_nadirs['glucose'][1]

        for i in range(0, len(peaks_and_nadirs) - 1, 2):
            if starts_with_peak:
                peak, nadir = peaks_and_nadirs['glucose'][i], peaks_and_nadirs['glucose'][i + 1]
            else:
                nadir, peak = peaks_and_nadirs['glucose'][i], peaks_and_nadirs['glucose'][i + 1]
            if abs(peak - nadir) > sd:
                excursions.append(abs(peak - nadir))

        return sum(excursions) / len(excursions) if excursions else 0
    
    def MAGE_cgm(self, std: int = 1) -> float:
        """
            Computes and returns the mean amplitude of glucose excursions
            Args:
                (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
                sd (integer): standard deviation for computing range (default=1)
            Returns:
                MAGE (float): the mean amplitude of glucose excursions 
            Refs:
                Sneh Gajiwala: https://github.com/snehG0205/NCSA_genomics/tree/2bfbb87c9c872b1458ef3597d9fb2e56ac13ad64

            No funciona bien.
                
        """
        # Extraer valores de glucosa e índices
        glucose = self.data['glucose'].tolist()
        ix = list(range(len(glucose)))
        
        # Encontrar mínimos y máximos locales
        a = np.diff(np.sign(np.diff(glucose))).nonzero()[0] + 1
        valleys = (np.diff(np.sign(np.diff(glucose))) > 0).nonzero()[0] + 1  # mínimos locales
        peaks = (np.diff(np.sign(np.diff(glucose))) < 0).nonzero()[0] + 1    # máximos locales

        # Almacenar mínimos y máximos locales
        excursion_points = pd.DataFrame(columns=['Index', 'Time', 'Glucose', 'Type'])
        k = 0
        for i in range(len(peaks)):
            excursion_points.loc[k] = [peaks[i], self.data['time'].iloc[peaks[i]], 
                                     self.data['glucose'].iloc[peaks[i]], "P"]
            k += 1

        for i in range(len(valleys)):
            excursion_points.loc[k] = [valleys[i], self.data['time'].iloc[valleys[i]], 
                                     self.data['glucose'].iloc[valleys[i]], "V"]
            k += 1

        excursion_points = excursion_points.sort_values(by=['Index'])
        excursion_points = excursion_points.reset_index(drop=True)

        # Seleccionar puntos de inflexión
        turning_points = pd.DataFrame(columns=['Index', 'Time', 'Glucose', 'Type'])
        k = 0
        for i in range(std, len(excursion_points.Index)-std):
            positions = [i-std, i, i+std]
            for j in range(0, len(positions)-1):
                if excursion_points.Type[positions[j]] == excursion_points.Type[positions[j+1]]:
                    if excursion_points.Type[positions[j]] == 'P':
                        if excursion_points.Glucose[positions[j]] >= excursion_points.Glucose[positions[j+1]]:
                            turning_points.loc[k] = excursion_points.loc[positions[j+1]]
                        else:
                            turning_points.loc[k] = excursion_points.loc[positions[j+1]]
                        k += 1
                    else:
                        if excursion_points.Glucose[positions[j]] <= excursion_points.Glucose[positions[j+1]]:
                            turning_points.loc[k] = excursion_points.loc[positions[j]]
                        else:
                            turning_points.loc[k] = excursion_points.loc[positions[j+1]]
                        k += 1

        if len(turning_points.index) < 10:
            turning_points = excursion_points.copy()
            excursion_count = len(excursion_points.index)
        else:
            excursion_count = len(excursion_points.index)/2

        turning_points = turning_points.drop_duplicates(subset="Index", keep="first")
        turning_points = turning_points.reset_index(drop=True)
        
        # Calcular MAGE
        mage = turning_points.Glucose.sum()/excursion_count if excursion_count > 0 else 0
        
        return round(mage, 3)
    
    def MODD(self, days: int = 1) -> dict:
        """
        Calcula el MODD (Mean Of Daily Differences) para múltiples días.
        
        :param days: Número de días para calcular diferencias (1-6)
        :return: Diccionario con valores MODD y estadísticas relacionadas
        :reference: DOI: 10.1089/dia.2009.0015 (Rodbard)
        """
        if not 1 <= days <= 6:
            raise ValueError("El número de días debe estar entre 1 y 6")
        
        # Usar el intervalo típico calculado por la clase padre
        intervals_per_day = int(24 * 60 / self.typical_interval)
        
        results = {}
        correlations = []
        
        for d in range(1, days + 1):
            # Calcular diferencias para d días
            shift_intervals = intervals_per_day * d
            
            # Crear copia de datos con valores desplazados
            data_copy = self.data.copy()
            data_copy[f'glucose_{d}d_ago'] = self.data['glucose'].shift(shift_intervals)
            
            # Eliminar filas con valores faltantes
            valid_data = data_copy.dropna(subset=['glucose', f'glucose_{d}d_ago'])
            
            if len(valid_data) > 0:
                # Calcular diferencias absolutas
                abs_diff = (valid_data['glucose'] - valid_data[f'glucose_{d}d_ago']).abs()
                
                # Calcular MODD para d días
                modd_value = abs_diff.mean()
                
                # Calcular correlación entre días
                corr = valid_data['glucose'].corr(valid_data[f'glucose_{d}d_ago'])
                
                results[f'MODD{d}'] = {
                    'value': modd_value,
                    'n_observations': len(valid_data),
                    'std': abs_diff.std(),
                    'correlation': corr
                }
                
                correlations.append(corr)
            else:
                results[f'MODD{d}'] = {
                    'value': None,
                    'n_observations': 0,
                    'std': None,
                    'correlation': None
                }
        
        # Añadir estadísticas generales
        results['summary'] = {
            'mean_correlation': np.mean(correlations) if correlations else None,
            'stability_index': np.std(correlations) if len(correlations) > 1 else None,
            'days_analyzed': days
        }
    
        return results
    
    def CONGA(self, min: int = 5, hours: int = 24) -> float:
        """
        Calcula el CONGA (Continuous Overall Net Glycemic Action).
        :param min: Distancia en minutos entre las mediciones.
        :param hours: Número de horas para calcular la diferencia.
        :return: Valor de CONGA.
        """
        intervals = hours * int(60/min)
        self.data['glucose_n_hours_ago'] = self.data['glucose'].shift(intervals)
        valid_data = self.data.dropna(subset=['glucose_n_hours_ago']).copy()
        valid_data['Dt'] = valid_data['glucose'] - valid_data['glucose_n_hours_ago']
        D_mean = valid_data['Dt'].mean()
        sum_squared_diff = ((valid_data['Dt'] - D_mean) ** 2).sum()
        k_star = len(valid_data)
        return np.sqrt(sum_squared_diff / (k_star - 1))

    

    def j_index(self) -> float:
        """Calcula el J-index."""
        return 0.001 * (self.mean() + self.sd())**2

    def LBGI(self) -> float:
        """
        Calcula el Low Blood Glucose Index (LBGI).
        :return: Valor de LBGI.
        :reference: DOI: 10.2337/db12-1396
        """
        self.data["f_bg"] = 1.509 * ((np.log(self.data["glucose"]))**1.084 - 5.381)
        self.data["r_bg"] = 10 * (self.data["f_bg"])**2
        self.data["rl_bg"] = self.data.apply(lambda row: row["r_bg"] if row["f_bg"] < 0 else 0, axis=1)
        return self.data["rl_bg"].mean()

    def HBGI(self) -> float:
        """
        Calcula el High Blood Glucose Index (HBGI).
        :return: Valor de HBGI.
        :reference: DOI: 10.2337/db12-1396
        """
        self.data["f_bg"] = 1.509 * ((np.log(self.data["glucose"]))**1.084 - 5.381)
        self.data["r_bg"] = 10 * (self.data["f_bg"])**2
        self.data["rh_bg"] = self.data.apply(lambda row: row["r_bg"] if row["f_bg"] > 0 else 0, axis=1)
        return self.data["rh_bg"].mean()

    

    def M_Value(self, target_glucose: float = 80) -> float:
        """
        Calcula el M-Value para evaluar la variabilidad de la glucosa en sangre.
        :param target_glucose: Valor objetivo de glucosa (por defecto 80 mg/dL).
        :return: M-Value.
        :reference: DOI: 10.2337/db12-1396
        """
        def calculate_M(PG):
            return abs(10 * np.log10(PG / target_glucose)) ** 3

        self.data['M'] = self.data['glucose'].apply(calculate_M)
        return self.data['M'].mean()

    def Lability_index(self, interval: int = 1, period: str = 'week') -> float:
        """
        Calcula el índice de labilidad (LI).
        :param interval: El intervalo de tiempo en horas para el cálculo.
        :param period: El periodo para la media del LI ('day', 'week', 'month').
        :return: El índice de labilidad (LI) medio para el periodo especificado.
        """
        data_copy = self.data.copy().set_index('time')
        resampled_data = data_copy.resample(f'{interval}h').asfreq().dropna().reset_index()
        
        if period == 'day':
            resampled_data['period'] = resampled_data['time'].dt.date
        elif period == 'week':
            resampled_data['period'] = (resampled_data['time'] - resampled_data['time'].min()).dt.days // 7
        elif period == 'month':
            resampled_data['period'] = resampled_data['time'].dt.to_period('M').apply(lambda r: r.start_time)
        else:
            raise ValueError("Periodo no válido. Usa 'day', 'week' o 'month'.")
        
        li_values = []
        for _, group in resampled_data.groupby('period'):
            glucose_readings = group['glucose'].values
            times = group['time'].values.astype('datetime64[h]').astype(int)
            
            if len(glucose_readings) < 2:
                continue
            
            li_sum = sum((glucose_readings[i] - glucose_readings[i + 1])**2 / (times[i + 1] - times[i])
                         for i in range(len(glucose_readings) - 1)
                         if 1 <= times[i + 1] - times[i] <= interval)
            
            li_values.append(li_sum / (len(glucose_readings) - 1))
        
        return np.mean(li_values) if li_values else 0

    def Variability(self) -> str:
        """
        Calcula todas las métricas de variabilidad.
        :return: Un string JSON con todas las métricas de variabilidad.
        """
        variability_metrics = {
            "CONGA1": self.CONGA(min=5, hours=1),
            "CONGA2": self.CONGA(min=5, hours=2),
            "CONGA4": self.CONGA(min=5, hours=4),
            "CONGA6": self.CONGA(min=5, hours=6),
            "CONGA24": self.CONGA(min=5, hours=24),
            "MODD": self.MODD(days=1),
            "J_index": self.j_index(),
            "LBGI": self.LBGI(),
            "HBGI": self.HBGI(),
            "MAGE": self.MAGE(),
            "M_value": self.M_Value(target_glucose=80),
            "LI_day": self.Lability_index(interval=1, period='day'),
            "LI_week": self.Lability_index(interval=1, period='week'),
            "LI_month": self.Lability_index(interval=1, period='month')
        }
        return json.dumps(variability_metrics)
    
    

    
    