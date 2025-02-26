import datetime
from .glucose_data import GlucoseData
from typing import Union
import numpy as np
import json
import pandas as pd
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
    
    def sd_within_day(self, min_count_threshold: float = 0.5) -> dict:
        """
        Calcula la desviación estándar dentro del día (SDw) como el promedio de las 
        desviaciones estándar de cada día individual y media de medias diarias.
        
        :param min_count_threshold: Umbral para considerar un día como válido 
                               (proporción de la mediana de conteos). Por defecto 0.5 (50%).
        :return: Diccionario con la SD promedio, media y datos adicionales
        """
        # Calcular estadísticas para cada día
        daily_stats = self.data.groupby(self.data['time'].dt.date)['glucose'].agg(['std', 'mean', 'count'])
        
        # Si no hay datos, devolver valores por defecto
        if daily_stats.empty:
            return {'sd': 0.0, 'mean': 0.0, 'dias_analizados': 0, 'dias_filtrados': 0, 'umbral_conteo': 0}
        
        # Identificar días con pocos datos
        count_threshold = daily_stats['count'].median() * min_count_threshold
        
        # Filtrar días con suficientes datos para el cálculo principal
        filtered_stats = daily_stats[daily_stats['count'] >= count_threshold]
        low_count_days = daily_stats[daily_stats['count'] < count_threshold]
        
        # Si después de filtrar no quedan días, usar todos los datos
        if filtered_stats.empty:
            return {
                'sd': daily_stats['std'].mean(),
                'mean': daily_stats['mean'].mean(),
                'dias_analizados': len(daily_stats),
                'dias_filtrados': 0,
                'umbral_conteo': count_threshold,
                'advertencia': 'No hay días con suficientes datos, se usaron todos los días disponibles'
            }
        
        # Calcular SDw y media con los días filtrados
        return {
            'sd': filtered_stats['std'].mean(),
            'mean': filtered_stats['mean'].mean(),
            'dias_analizados': len(filtered_stats),
            'dias_filtrados': len(low_count_days),
            'umbral_conteo': count_threshold,
            'rango_sd': {
                'min': filtered_stats['std'].min() if not filtered_stats.empty else 0,
                'max': filtered_stats['std'].max() if not filtered_stats.empty else 0
            },
            'estadisticas_diarias': filtered_stats.to_dict(),
            'dias_con_pocos_datos': low_count_days.to_dict() if not low_count_days.empty else {}
        }
    def sdw(self, min_count_threshold: float = 0.5) -> float:
        """
        Calcula la desviación estándar dentro del día (SDw).
        Este es un método simplificado que devuelve solo el valor de SD.
        
        :param min_count_threshold: Umbral para considerar un día como válido 
                                   (proporción de la mediana de conteos). Por defecto 0.5 (50%).
        :return: Valor de SDw (float)
        """
        return self.sd_within_day(min_count_threshold)['sd']
    
    def sd_within_day_segment(self, start_time: str, duration_hours: int) -> dict:
        """
        Calcula la desviación estándar within-day para un segmento específico del día.
        Para cada día, calcula la SD del segmento horario especificado y luego
        promedia estas SD diarias.
        
        :param start_time: Hora de inicio en formato "HH:MM"
        :param duration_hours: Duración del segmento en horas
        :return: Promedio de las SD del segmento de cada día y promedio de las medias diarias
        
        Ejemplo:
            sd_within_day_segment("00:00", 8)  # SD promedio del segmento nocturno (00:00-08:00)
            sd_within_day_segment("08:00", 8)  # SD promedio del segmento diurno (08:00-16:00)
        """
        # Obtener los datos del segmento
        segment_data = self._get_segment_data(start_time, duration_hours)
        
        if segment_data.empty:
            return {'sd': 0.0, 'mean': 0.0}
        
        # Calcular SD y media para el segmento de cada día
        daily_segment_stats = segment_data.groupby(segment_data['time'].dt.date)['glucose'].agg(['std', 'mean'])
        
        return {
            'sd': daily_segment_stats['std'].mean() if not daily_segment_stats.empty else 0.0,
            'mean': daily_segment_stats['mean'].mean() if not daily_segment_stats.empty else 0.0
        }

    def sd_between_timepoints(self, min_count_threshold: float = 0.5, filter_outliers: bool = True, 
                         agrupar_por_intervalos: bool = False, intervalo_minutos: int = 5) -> dict:
        """
        Calcula la desviación estándar entre puntos temporales (SDhh:mm)
        
        Esta métrica mide la variabilidad del patrón de glucosa a lo largo del día.
        
        Versión optimizada para grandes conjuntos de datos.
        
        :param min_count_threshold: Umbral para considerar una marca temporal como válida
                               (proporción de la mediana de conteos). Por defecto 0.5 (50%).
        :param filter_outliers: Si es True, filtra las marcas temporales con pocos datos.
        :param agrupar_por_intervalos: Si es True, agrupa los datos en intervalos regulares de tiempo.
        :param intervalo_minutos: Tamaño del intervalo en minutos para la agrupación (por defecto 5 min).
        :return: Diccionario con el valor de SDhh:mm y estadísticas relacionadas.
        """
        # Crear una copia de los datos con solo las columnas necesarias
        df = self.data[['time', 'glucose']].copy()
        
        # Extraer solo hora y minuto para reducir la carga computacional
        df['hour_min'] = df['time'].apply(lambda x: x.hour * 60 + x.minute)
        
        if agrupar_por_intervalos:
            # Agrupar por intervalos de tiempo para reducir la cantidad de puntos temporales
            df['interval'] = (df['hour_min'] // intervalo_minutos) * intervalo_minutos
            # Usar groupby con transform más eficiente que apply para grandes datasets
            grouped = df.groupby(['day', 'interval'])
            df_agg = grouped.agg({'glucose': 'mean'}).reset_index()
            
            # Calcular la hora y minuto a partir del intervalo para el resultado final
            df_agg['hour'] = df_agg['interval'] // 60
            df_agg['minute'] = df_agg['interval'] % 60
            
            # Usar estadísticas descriptivas vectorizadas
            timepoint_means = df_agg.groupby(['hour', 'minute'])['glucose'].mean()
            df_final = pd.DataFrame({'mean': timepoint_means})
            
            # Contar número de días con datos para cada punto temporal
            timepoint_counts = df_agg.groupby(['hour', 'minute']).size()
            df_final['count'] = timepoint_counts
        else:
            # Extraer características temporales vectorizadamente
            df['hour'] = df['time'].dt.hour
            df['minute'] = df['time'].dt.minute
            df['day'] = df['time'].dt.date
            
            # Calcular estadísticas por hora:minuto de forma vectorizada
            grouped = df.groupby(['hour', 'minute', 'day'])
            daily_means = grouped['glucose'].mean().reset_index()
            
            # Agrupar nuevamente para obtener las medias por punto temporal
            timepoint_stats = daily_means.groupby(['hour', 'minute'])
            
            # Calcular estadísticas de forma vectorizada
            df_final = pd.DataFrame({
                'mean': timepoint_stats['glucose'].mean(),
                'count': timepoint_stats.size()
            })
        
        # Filtrar puntos con pocos datos si se solicita
        if filter_outliers:
            median_count = df_final['count'].median()
            threshold = median_count * min_count_threshold
            valid_timepoints = df_final[df_final['count'] >= threshold]
        else:
            valid_timepoints = df_final
        
        # Calcular SDhh:mm (la desviación estándar del patrón promedio)
        sd_value = valid_timepoints['mean'].std()
        mean_value = valid_timepoints['mean'].mean()
        
        # Crear el resultado como diccionario
        result = {
            'sd': sd_value,
            'mean': mean_value,
            'valid_timepoints': len(valid_timepoints),
            'total_timepoints': len(df_final),
            'median_count': df_final['count'].median(),
            'min_count': df_final['count'].min(),
            'max_count': df_final['count'].max()
        }
        
        return result
    
    def sd_between_timepoints_segment(self, start_time: str, duration_hours: int) -> dict:
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
        Calcula SDws y media de las series temporales. A menor número de horas, más pequeño es el valor de SDws porque da menos tiempo para que la glucosa varíe.
        Devuelve: {'sd': float, 'mean': float}
        """
        # Convertir horas a intervalos según la frecuencia de los datos
        data = self.data.sort_values('time')
        series_stats = []
        
        # Crear ventanas móviles del tamaño especificado
        for start_time in data['time']:
            end_time = start_time + pd.Timedelta(hours=hours)
            series = data[(data['time'] >= start_time) & (data['time'] < end_time)]['glucose']
            
            # Solo calcular estadísticas si hay suficientes datos en la serie
            if len(series) > 1:  # Necesitamos al menos 2 puntos para calcular SD
                series_stats.append({
                    'sd': series.std(),
                    'mean': series.mean()
                })
        
        return {
            'sd': np.mean([s['sd'] for s in series_stats]) if series_stats else 0.0,
            'mean': np.mean([s['mean'] for s in series_stats]) if series_stats else 0.0
        }
    
    def sd_daily_mean(self, min_count_threshold: float = 0.5) -> dict:
        """
        Calcula la desviación estándar de los promedios diarios (SDdm).
        
        :param min_count_threshold: Umbral para considerar un día como válido 
                               (proporción de la mediana de conteos). Por defecto 0.5 (50%).
        :return: Diccionario con la SD, media y datos adicionales
        """
        # Calcular estadísticas para cada día
        daily_stats = self.data.groupby(self.data['time'].dt.date)['glucose'].agg(['mean', 'count'])
        
        # Si no hay datos, devolver valores por defecto
        if daily_stats.empty:
            return {'sd': 0.0, 'mean': 0.0, 'dias_analizados': 0, 'dias_filtrados': 0, 'umbral_conteo': 0}
        
        # Identificar días con pocos datos
        count_threshold = daily_stats['count'].median() * min_count_threshold
        
        # Filtrar días con suficientes datos para el cálculo principal
        filtered_stats = daily_stats[daily_stats['count'] >= count_threshold]
        low_count_days = daily_stats[daily_stats['count'] < count_threshold]
        
        # Si después de filtrar no quedan días, usar todos los datos
        if filtered_stats.empty:
            return {
                'sd': daily_stats['mean'].std(),
                'mean': daily_stats['mean'].mean(),
                'dias_analizados': len(daily_stats),
                'dias_filtrados': 0,
                'umbral_conteo': count_threshold,
                'advertencia': 'No hay días con suficientes datos, se usaron todos los días disponibles'
            }
        
        # Calcular SDdm y media con los días filtrados
        return {
            'sd': filtered_stats['mean'].std(),
            'mean': filtered_stats['mean'].mean(),
            'dias_analizados': len(filtered_stats),
            'dias_filtrados': len(low_count_days),
            'umbral_conteo': count_threshold,
            'rango_medias': {
                'min': filtered_stats['mean'].min(),
                'max': filtered_stats['mean'].max()
            },
            'estadisticas_diarias': filtered_stats.to_dict(),
            'dias_con_pocos_datos': low_count_days.to_dict() if not low_count_days.empty else {}
        }

    def sd_same_timepoint(self, min_count_threshold: float = 0.5, filter_outliers: bool = True, 
                         agrupar_por_intervalos: bool = False, intervalo_minutos: int = 5) -> dict:
        """
        Calcula la SD entre días para cada punto temporal ($SD_{b,hh:mm}$) con opciones avanzadas.
        Para cada tiempo específico del día (HH:MM), calcula la SD entre días, luego promedia todas estas SD.
        
        :param min_count_threshold: Umbral para considerar una marca temporal como válida 
                               (proporción de la mediana de conteos). Por defecto 0.5 (50%).
        :param filter_outliers: Si es True, filtra las marcas temporales con pocos datos 
                           antes de calcular la SD.
        :param agrupar_por_intervalos: Si es True, agrupa los datos en intervalos regulares de tiempo.
        :param intervalo_minutos: Tamaño del intervalo en minutos para la agrupación (por defecto 5 min).
        :return: Diccionario con la SD promedio, media y datos por marca temporal
        """
        if agrupar_por_intervalos:
            # Crear una columna con el tiempo redondeado al intervalo más cercano
            data_copy = self.data.copy()
            
            # Convertir la hora a minutos desde medianoche
            minutos_del_dia = data_copy['time'].dt.hour * 60 + data_copy['time'].dt.minute
            
            # Redondear al intervalo más cercano
            intervalo_redondeado = (minutos_del_dia // intervalo_minutos) * intervalo_minutos
            
            # Convertir de nuevo a formato HH:MM
            horas = intervalo_redondeado // 60
            minutos = intervalo_redondeado % 60
            data_copy['time_interval'] = horas.astype(str).str.zfill(2) + ':' + minutos.astype(str).str.zfill(2)
            
            # Agrupar por el intervalo de tiempo y calcular estadísticas
            time_stats = data_copy.groupby(['time_interval', data_copy['time'].dt.date])['glucose'].agg(['mean', 'count'])
            time_sds = time_stats.groupby(level=0)['mean'].agg(['std', 'mean', 'count'])
        else:
            # Comportamiento original: agrupar por la marca de tiempo exacta "HH:MM"
            time_stats = self.data.groupby([self.data['time'].dt.strftime("%H:%M"), self.data['time'].dt.date])['glucose'].agg(['mean', 'count'])
            time_sds = time_stats.groupby(level=0)['mean'].agg(['std', 'mean', 'count'])
        
        # Identificar marcas temporales con pocos datos (potencialmente problemáticas)
        count_threshold = time_sds['count'].median() * min_count_threshold
        low_count_times = time_sds[time_sds['count'] < count_threshold]
        
        # Convertir el índice a formato de hora decimal para facilitar la visualización
        low_count_dict = {}
        for time_str in low_count_times.index:
            h, m = map(int, time_str.split(':'))
            decimal_time = h + m/60.0
            low_count_dict[time_str] = {
                'hora_decimal': decimal_time,
                'conteo': int(low_count_times.loc[time_str, 'count']),
                'valor': low_count_times.loc[time_str, 'mean']
            }
        
        # Filtrar marcas temporales con pocos datos si se solicita
        if filter_outliers and low_count_dict:
            filtered_stats = time_sds[~time_sds.index.isin(low_count_dict.keys())]
            sd_value = filtered_stats['std'].mean()
            mean_value = filtered_stats['mean'].mean()
        else:
            sd_value = time_sds['std'].mean()
            mean_value = time_sds['mean'].mean()
        
        return {
            'sd': sd_value,
            'mean': mean_value,
            'conteo_por_marca': time_sds['count'].to_dict(),
            'valores_por_marca': time_sds['mean'].to_dict(),
            'sd_por_marca': time_sds['std'].to_dict(),
            'marcas_con_pocos_datos': low_count_dict,
            'umbral_conteo': count_threshold,
            'filtrado_aplicado': filter_outliers,
            'agrupacion_por_intervalos': agrupar_por_intervalos,
            'intervalo_minutos': intervalo_minutos if agrupar_por_intervalos else None
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
            'SDhh:mm': self.sd_between_timepoints()['sd'],
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
            'CVhh:mm': self.sd_between_timepoints()['sd']/self.sd_between_timepoints()['mean']*100,
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


    def pattern_stability_metrics(self) -> dict:
        """
        Calcula métricas de estabilidad del patrón glucémico según el artículo.

        Revisar con alguien que domine estadística para comprobar todo!!
        
        Returns:
            dict: Diccionario con las diferentes métricas de estabilidad
        """
        # 1. Ratio SDhh:mm/SDw
        ratio_sd = (self.sd_between_timepoints()['sd'] / self.sd_within_day()['sd'])**2
        # 1. Ratio SDhh:mm/SDw por parte del día (corregido)
        ratio_sd_by_part_of_day = {
            "Noche": (self.sd_between_timepoints_segment("00:00", 8)['sd'] / self.sd_within_day_segment("00:00", 8)['sd'])**2,
            "Día": (self.sd_between_timepoints_segment("08:00", 8)['sd'] / self.sd_within_day_segment("08:00", 8)['sd'])**2,
            "Tarde": (self.sd_between_timepoints_segment("16:00", 8)['sd'] / self.sd_within_day_segment("16:00", 8)['sd'])**2
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

    ## MEDIDAS AVANZADAS DE VARIABILIDAD

    def MAGE_Baghurst(self, threshold_sd: int = 1, approach: int = 1, plot: bool = False) -> dict:
        """
        Calcula el MAGE según el algoritmo de Baghurst.
        
        Cambios principales:
        1. Manejo correcto de bordes en el suavizado
        2. Búsqueda de turning points en datos originales entre mínimos/máximos del perfil suavizado
        3. Proceso iterativo de eliminación de puntos no válidos
        4. Manejo de excursiones al inicio/final del dataset
        
        :param threshold_sd: Número de desviaciones estándar para el umbral
        :param approach: 1 para usar suavizado según Baghurst original, 2 para eliminación directa, 3 para suavizado mejorado
        :param plot: Si es True, genera una visualización de los picos y valles identificados
        :return: Diccionario con MAGE+, MAGE- y métricas relacionadas
        
        Approach 1: Algoritmo original de Baghurst con suavizado y proceso iterativo de eliminación
        Approach 2: Eliminación directa de puntos intermedios en secuencias monótonas
        Approach 3: Suavizado mejorado con filtrado adicional de turning points
        """
        glucose = self.data['glucose'].values
        times = self.data['time'].values
        sd = self.sd()
        threshold = threshold_sd * sd
        
        # Guardar los turning points para cada enfoque si plot=True
        turning_points_approaches = {}
        
        # Enfoque 1: Suavizado según algoritmo original de Baghurst
        if approach == 1 or plot:
            # PASO 1: Aplicar filtro de suavizado e identificar turning points en datos suavizados
            weights = np.array([1,2,4,8,16,8,4,2,1])/46
            smoothed = np.zeros_like(glucose)
            
            # Suavizado central
            for i in range(4, len(glucose)-4):
                smoothed[i] = np.dot(weights, glucose[i-4:i+5])
                
            # Manejo de bordes con media simple
            for i in range(4):
                smoothed[i] = glucose[:i+5].mean()
                smoothed[-(i+1)] = glucose[-(i+5):].mean()
                
            # Identificar turning points en datos suavizados mediante primeras diferencias
            delta = np.diff(smoothed)
            turning_smoothed = np.where(np.diff(np.sign(delta)))[0] + 1
            
            # PASO 2: Identificar máximos/mínimos locales en datos originales
            turning_points_1 = []
            for i in range(len(turning_smoothed)-1):
                start = turning_smoothed[i]
                end = turning_smoothed[i+1]
                
                # Buscar máximo real en intervalo ascendente
                if smoothed[start] < smoothed[end]:
                    true_peak = np.argmax(glucose[start:end]) + start
                    turning_points_1.append(true_peak)
                # Buscar mínimo real en intervalo descendente
                else:
                    true_valley = np.argmin(glucose[start:end]) + start
                    turning_points_1.append(true_valley)
            
            # Añadir el primer y último punto si son extremos
            if len(turning_points_1) > 0 and turning_points_1[0] > 0:
                if glucose[0] > glucose[turning_points_1[0]] or glucose[0] < glucose[turning_points_1[0]]:
                    turning_points_1.insert(0, 0)
            
            if len(turning_points_1) > 0 and turning_points_1[-1] < len(glucose) - 1:
                if glucose[-1] > glucose[turning_points_1[-1]] or glucose[-1] < glucose[turning_points_1[-1]]:
                    turning_points_1.append(len(glucose) - 1)
            
            # PASO 3: Eliminar turning points asociados con excursiones no contables en ambos lados
            # Mantener aquellos cuyos máximos/mínimos adyacentes son más bajos/altos en ambos lados
            keep_iterating = True
            while keep_iterating:
                to_delete = []
                
                for i in range(1, len(turning_points_1) - 1):
                    current_idx = turning_points_1[i]
                    prev_idx = turning_points_1[i-1]
                    next_idx = turning_points_1[i+1]
                    
                    current_val = glucose[current_idx]
                    prev_val = glucose[prev_idx]
                    next_val = glucose[next_idx]
                    
                    # Verificar si ambas diferencias son menores que el umbral
                    if (abs(current_val - prev_val) < threshold and 
                        abs(current_val - next_val) < threshold):
                        
                        # Retener si es un máximo local (ambos adyacentes más bajos)
                        is_local_max = current_val > prev_val and current_val > next_val
                        # Retener si es un mínimo local (ambos adyacentes más altos)
                        is_local_min = current_val < prev_val and current_val < next_val
                        
                        # Si no es un máximo/mínimo local, marcar para eliminación
                        if not (is_local_max or is_local_min):
                            to_delete.append(i)
                
                # Si no hay más puntos para eliminar, terminar
                if not to_delete:
                    keep_iterating = False
                else:
                    # Eliminar puntos marcados
                    for idx in sorted(to_delete, reverse=True):
                        turning_points_1.pop(idx)
                
                # PASO 4: Eliminar observaciones que ya no son turning points
                delta_turning = np.diff([glucose[tp] for tp in turning_points_1])
                false_turning = []
                
                for i in range(1, len(delta_turning)):
                    # Si las diferencias tienen el mismo signo, no es un turning point
                    if delta_turning[i-1] * delta_turning[i] > 0:
                        false_turning.append(i)
                
                # Eliminar puntos falsos
                for idx in sorted(false_turning, reverse=True):
                    turning_points_1.pop(idx)
            
            # PASO 5: Eliminar turning points con excursión contable en un solo lado
            if len(turning_points_1) >= 3:
                to_delete = []
                
                for i in range(1, len(turning_points_1) - 1):
                    current_idx = turning_points_1[i]
                    prev_idx = turning_points_1[i-1]
                    next_idx = turning_points_1[i+1]
                    
                    current_val = glucose[current_idx]
                    prev_val = glucose[prev_idx]
                    next_val = glucose[next_idx]
                    
                    # Verificar si sólo hay excursión significativa en un lado
                    has_sig_prev = abs(current_val - prev_val) >= threshold
                    has_sig_next = abs(current_val - next_val) >= threshold
                    
                    if has_sig_prev != has_sig_next:  # XOR lógico - solo uno es verdadero
                        to_delete.append(i)
                
                # Eliminar puntos marcados
                for idx in sorted(to_delete, reverse=True):
                    turning_points_1.pop(idx)
                
                # Verificar de nuevo si hay puntos que ya no son turning points
                delta_turning = np.diff([glucose[tp] for tp in turning_points_1])
                false_turning = []
                
                for i in range(1, len(delta_turning)):
                    if delta_turning[i-1] * delta_turning[i] > 0:
                        false_turning.append(i)
                
                for idx in sorted(false_turning, reverse=True):
                    turning_points_1.pop(idx)
            
            # PASO 6: Eliminar excursiones no contables al inicio o final
            if len(turning_points_1) >= 2:
                # Verificar excursión inicial
                if abs(glucose[turning_points_1[0]] - glucose[turning_points_1[1]]) < threshold:
                    turning_points_1.pop(0)
                
                # Verificar excursión final
                if len(turning_points_1) >= 2 and abs(glucose[turning_points_1[-1]] - glucose[turning_points_1[-2]]) < threshold:
                    turning_points_1.pop(-1)
            
            # Asegurar que los puntos están ordenados y son únicos
            turning_points_1 = sorted(list(set(turning_points_1)))
            turning_points_approaches[1] = turning_points_1
            
            if approach == 1:
                turning_points = turning_points_1
        
        # Enfoque 2: Eliminación directa
        if approach == 2 or plot:
            turning_points_2 = []
            i = 0
            
            # 1. Primera pasada: eliminar puntos intermedios en secuencias monótonas
            while i < len(glucose) - 2:
                if (glucose[i] <= glucose[i+1] <= glucose[i+2]) or \
                   (glucose[i] >= glucose[i+1] >= glucose[i+2]):
                    # El punto intermedio es parte de una secuencia monótona
                    i += 1
                else:
                    # Punto i+1 es un turning point potencial
                    turning_points_2.append(i+1)
                    i += 2
            
            # Asegurar que incluimos el primer y último punto si son relevantes
            if len(turning_points_2) == 0 or turning_points_2[0] > 0:
                turning_points_2.insert(0, 0)
            if turning_points_2[-1] < len(glucose) - 1:
                turning_points_2.append(len(glucose) - 1)
            
            # 2. Segunda pasada: eliminar excursiones que no superan el umbral
            valid_points = []
            for i in range(1, len(turning_points_2)-1):
                prev_val = glucose[turning_points_2[i-1]]
                curr_val = glucose[turning_points_2[i]]
                next_val = glucose[turning_points_2[i+1]]
                
                # Verificar si es un máximo o mínimo válido
                if ((curr_val > prev_val and curr_val > next_val) or 
                    (curr_val < prev_val and curr_val < next_val)) and (
                    abs(curr_val - prev_val) >= threshold or 
                    abs(curr_val - next_val) >= threshold):
                    valid_points.append(turning_points_2[i])
            
            # Asegurar que mantenemos puntos inicial y final si son necesarios
            if valid_points and valid_points[0] > 0:
                valid_points.insert(0, 0)
            if valid_points and valid_points[-1] < len(glucose) - 1:
                valid_points.append(len(glucose) - 1)
            
            turning_points_2 = valid_points
            turning_points_approaches[2] = turning_points_2
            
            if approach == 2:
                turning_points = turning_points_2
    
        # Enfoque 3: Suavizado mejorado
        if approach == 3 or plot:
            # 1. Aplicar filtro de suavizado con manejo de bordes (igual que enfoque 1)
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
            # y aplicar filtrado adicional
            
            # Primero identificamos todos los turning points potenciales
            potential_turning_points = []
            for i in range(len(turning_smoothed)-1):
                start = turning_smoothed[i]
                end = turning_smoothed[i+1]
                
                # Buscar máximo real en intervalo ascendente
                if smoothed[start] < smoothed[end]:
                    true_peak = np.argmax(glucose[start:end]) + start
                    potential_turning_points.append((true_peak, 'peak', glucose[true_peak]))
                # Buscar mínimo real en intervalo descendente
                else:
                    true_valley = np.argmin(glucose[start:end]) + start
                    potential_turning_points.append((true_valley, 'valley', glucose[true_valley]))
            
            # Ahora procesamos los turning points para eliminar picos/valles intermedios menores
            turning_points_3 = []
            if potential_turning_points:
                # Añadir el primer punto
                turning_points_3.append(potential_turning_points[0][0])
                
                # Procesar el resto de puntos
                for i in range(1, len(potential_turning_points)-1):
                    prev_point, prev_type, prev_value = potential_turning_points[i-1]
                    curr_point, curr_type, curr_value = potential_turning_points[i]
                    next_point, next_type, next_value = potential_turning_points[i+1]
                    
                    # Si tenemos un patrón valle-pico-valle o pico-valle-pico
                    if curr_type == prev_type:
                        # Saltamos este punto, es redundante
                        continue
                        
                    # Si tenemos un pico entre dos valles, verificar si es significativo
                    if curr_type == 'peak' and prev_type == 'valley' and next_type == 'valley':
                        # Si el pico no es significativamente más alto que ambos valles, lo saltamos
                        if (curr_value - prev_value < threshold/2) or (curr_value - next_value < threshold/2):
                            continue
                            
                    # Si tenemos un valle entre dos picos, verificar si es significativo
                    if curr_type == 'valley' and prev_type == 'peak' and next_type == 'peak':
                        # Si el valle no es significativamente más bajo que ambos picos, lo saltamos
                        if (prev_value - curr_value < threshold/2) or (next_value - curr_value < threshold/2):
                            continue
                    
                    # Si llegamos aquí, el punto es significativo
                    turning_points_3.append(curr_point)
                
                # Añadir el último punto
                turning_points_3.append(potential_turning_points[-1][0])
            
            # Asegurarnos de que tenemos al menos el primer y último punto
            if len(turning_points_3) == 0 and len(glucose) > 0:
                turning_points_3 = [0, len(glucose)-1]
            elif len(turning_points_3) == 1 and len(glucose) > 1:
                if turning_points_3[0] == 0:
                    turning_points_3.append(len(glucose)-1)
                else:
                    turning_points_3.insert(0, 0)
            
            turning_points_3 = np.unique(turning_points_3)
            turning_points_approaches[3] = turning_points_3
            
            if approach == 3:
                turning_points = turning_points_3
        
        # 3. Calcular excursiones válidas
        excursions = []
        last_val = glucose[turning_points[0]]
        last_point = turning_points[0]
        
        for point in turning_points[1:]:
            curr_val = glucose[point]
            diff = abs(curr_val - last_val)
            
            if diff >= threshold:
                excursions.append({
                    'start_point': last_point,
                    'end_point': point,
                    'start': last_val,
                    'end': curr_val,
                    'type': 'up' if curr_val > last_val else 'down',
                    'magnitude': diff
                })
                last_val = curr_val
                last_point = point
            else:
                # Si no supera el umbral, actualizamos el último valor sin crear excursión
                last_val = curr_val
                last_point = point
        
        # Separar excursiones y calcular métricas
        excursions_up = [e['magnitude'] for e in excursions if e['type'] == 'up']
        excursions_down = [e['magnitude'] for e in excursions if e['type'] == 'down']
        
        mage_plus = np.mean(excursions_up) if excursions_up else 0
        mage_minus = np.mean(excursions_down) if excursions_down else 0
        mage_avg = np.mean(excursions_up + excursions_down) if (excursions_up or excursions_down) else 0
        
        # Generar visualización si plot=True
        if plot:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from datetime import timedelta
            
            # Obtener todos los días únicos en los datos
            unique_days = pd.Series(times).dt.normalize().unique()
            
            # Configurar la figura y ejes - ahora con 3 subplots
            fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
            plt.ion()  # Modo interactivo
            
            # Calcular excursiones para cada enfoque
            excursions_by_approach = {}
            
            for approach_num in [1, 2, 3]:
                # Usar los turning points específicos de este enfoque
                if approach_num in turning_points_approaches:
                    tp = turning_points_approaches[approach_num]
                    
                    # Calcular excursiones para este enfoque
                    excursions_approach = []
                    if len(tp) > 1:
                        last_val = glucose[tp[0]]
                        last_point = tp[0]
                        
                        for point in tp[1:]:
                            curr_val = glucose[point]
                            diff = abs(curr_val - last_val)
                            
                            if diff >= threshold:
                                excursions_approach.append({
                                    'start_point': last_point,
                                    'end_point': point,
                                    'start': last_val,
                                    'end': curr_val,
                                    'type': 'up' if curr_val > last_val else 'down',
                                    'magnitude': diff
                                })
                            
                            # Siempre actualizamos el último valor y punto
                            last_val = curr_val
                            last_point = point
                    
                    excursions_by_approach[approach_num] = excursions_approach
            
            # Función para actualizar el gráfico con un día específico
            def update_plot(day_index):
                # Limpiar los ejes
                for ax in axs:
                    ax.clear()
                
                # Obtener el día actual
                current_day = unique_days[day_index]
                next_day = current_day + timedelta(days=1)
                
                # Filtrar datos para mostrar solo el día actual
                day_mask = (times >= current_day) & (times < next_day)
                day_times = times[day_mask]
                day_glucose = glucose[day_mask]
                
                if len(day_times) > 0:
                    # Para cada enfoque
                    for i, approach in enumerate([1, 2, 3]):
                        ax = axs[i]
                        # Dibujar la línea de glucosa
                        ax.plot(day_times, day_glucose, 'b-', label='Glucosa')
                        
                        # Obtener turning points para este enfoque
                        approach_turning_points = turning_points_approaches.get(approach, [])
                        
                        # Obtener excursiones para este enfoque
                        approach_excursions = excursions_by_approach.get(approach, [])
                        
                        # Filtrar turning points para este día
                        day_turning_points = [tp for tp in approach_turning_points if day_mask[tp]]
                        
                        # Identificar puntos involucrados en excursiones
                        excursion_points = set()
                        day_excursions = []
                        
                        for exc in approach_excursions:
                            start_point = exc['start_point']
                            end_point = exc['end_point']
                            
                            # Verificar si la excursión está en el día actual
                            if day_mask[start_point] or day_mask[end_point]:
                                if day_mask[start_point]:
                                    excursion_points.add(start_point)
                                if day_mask[end_point]:
                                    excursion_points.add(end_point)
                                day_excursions.append(exc)
                        
                        # Clasificar turning points
                        significant_points = [tp for tp in day_turning_points if tp in excursion_points]
                        non_significant_points = [tp for tp in day_turning_points if tp not in excursion_points]
                        
                        # Dibujar puntos no significativos en azul
                        for tp in non_significant_points:
                            ax.plot(times[tp], glucose[tp], 'bo', markersize=6)
                        
                        # Dibujar puntos significativos en rojo
                        for tp in significant_points:
                            ax.plot(times[tp], glucose[tp], 'ro', markersize=8)
                        
                        # Dibujar líneas para las excursiones
                        for exc in day_excursions:
                            start_point = exc['start_point']
                            end_point = exc['end_point']
                            
                            # Asegurarse de que ambos puntos están en el día actual
                            if day_mask[start_point] and day_mask[end_point]:
                                # Dibujar línea gruesa de color según tipo de excursión
                                color = 'green' if exc['type'] == 'up' else 'red'
                                ax.plot([times[start_point], times[end_point]], 
                                        [glucose[start_point], glucose[end_point]], 
                                        color=color, linewidth=2.5, alpha=0.7)
                        
                        # Calcular MAGE para este enfoque y día
                        excursions_up = [e['magnitude'] for e in day_excursions if e['type'] == 'up']
                        excursions_down = [e['magnitude'] for e in day_excursions if e['type'] == 'down']
                        
                        mage_plus = np.mean(excursions_up) if excursions_up else 0
                        mage_minus = np.mean(excursions_down) if excursions_down else 0
                        mage_avg = np.mean(excursions_up + excursions_down) if (excursions_up or excursions_down) else 0
                        
                        # Configurar título y etiquetas
                        approach_name = "Suavizado" if approach == 1 else "Eliminación directa" if approach == 2 else "Suavizado mejorado"
                        ax.set_title(f'MAGE Baghurst - Enfoque {approach} ({approach_name}) - {current_day.strftime("%d/%m/%Y")}\n'
                                    f'MAGE+: {mage_plus:.1f}, MAGE-: {mage_minus:.1f}, MAGE: {mage_avg:.1f}, Excursiones: {len(day_excursions)}')
                        ax.set_ylabel('Glucosa (mg/dL)')
                        ax.grid(True)
                        ax.axhline(y=self.mean() + threshold, color='g', linestyle='--', label=f'Umbral (+{threshold_sd} SD)')
                        ax.axhline(y=self.mean() - threshold, color='g', linestyle='--', label=f'Umbral (-{threshold_sd} SD)')
                        
                        # Leyenda personalizada
                        from matplotlib.lines import Line2D
                        custom_lines = [
                            Line2D([0], [0], color='b', marker='o', linestyle='None', markersize=6),
                            Line2D([0], [0], color='r', marker='o', linestyle='None', markersize=8),
                            Line2D([0], [0], color='green', linewidth=2.5),
                            Line2D([0], [0], color='red', linewidth=2.5),
                            Line2D([0], [0], color='g', linestyle='--')
                        ]
                        ax.legend(custom_lines, ['Puntos de inflexión', 'Puntos de excursiones', 
                                                'Excursión positiva', 'Excursión negativa', 'Umbral (±1 SD)'])
                    
                    # Configurar eje x para el último gráfico
                    axs[2].set_xlabel('Tiempo')
                    
                    # Formatear eje x para mostrar horas
                    for ax in axs:
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
                        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
                    
                    # Añadir información de navegación
                    fig.suptitle(f'Día {day_index+1} de {len(unique_days)} - Presiona ← o → para navegar, q para salir', fontsize=12)
                    
                    plt.tight_layout()
                    plt.subplots_adjust(top=0.95)  # Hacer espacio para el título superior
                    fig.canvas.draw_idle()
            
            # Índice del día actual
            current_day_index = 0
            
            # Función para manejar eventos de teclado
            def on_key(event):
                nonlocal current_day_index
                
                if event.key == 'right' and current_day_index < len(unique_days) - 1:
                    current_day_index += 1
                    update_plot(current_day_index)
                elif event.key == 'left' and current_day_index > 0:
                    current_day_index -= 1
                    update_plot(current_day_index)
                elif event.key == 'q':
                    plt.close(fig)
            
            # Conectar el evento de teclado
            fig.canvas.mpl_connect('key_press_event', on_key)
            
            # Mostrar el primer día
            update_plot(current_day_index)
            
            # Mostrar instrucciones
            print("Navegación:")
            print("  ← Flecha izquierda: Día anterior")
            print("  → Flecha derecha: Día siguiente")
            print("  q: Salir")
            
            # Bloquear hasta que se cierre la figura
            plt.show(block=True)
        
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
        :reference: DOI: 10.1089/dia.2009/0015 (Rodbard)
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
    
    ## MEDIDAS DE LA CALIDAD DE GLUCEMIA
    
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

    
    
    