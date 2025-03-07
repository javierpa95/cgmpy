import datetime
from .glucose_data import GlucoseData
from typing import Union
import numpy as np
import pandas as pd
import math

class GlucoseMetrics(GlucoseData):
    def __init__(self, data_source: Union[str, pd.DataFrame], date_col: str="time", glucose_col: str="glucose", delimiter: Union[str, None] = None, header: int = 0, start_date: Union[str, datetime.datetime, None] = None, end_date: Union[str, datetime.datetime, None] = None, log: bool = False):
        """
        Inicializa la clase GlucoseMetrics con optimizaciones para conjuntos de datos grandes.
        
        :param data_source: Archivo CSV o DataFrame con los datos de glucosa
        :param date_col: Nombre de la columna de fecha/hora
        :param glucose_col: Nombre de la columna de valores de glucosa
        :param delimiter: Delimitador para archivos CSV
        :param header: Fila de encabezado para archivos CSV
        :param start_date: Fecha de inicio para filtrar datos
        :param end_date: Fecha de fin para filtrar datos
        :param log: Si True, guarda información detallada de las operaciones realizadas
        """
        # Si ya recibimos un DataFrame optimizado desde GlucosePlot, lo pasamos directamente
        if isinstance(data_source, pd.DataFrame):
            # Verificar si el DataFrame ya tiene fecha y glucosa con los nombres correctos
            if date_col in data_source.columns and glucose_col in data_source.columns:
                # Verificar si la columna de fecha ya es datetime
                if not pd.api.types.is_datetime64_dtype(data_source[date_col]):
                    # Convertir a datetime de manera optimizada
                    data_source[date_col] = pd.to_datetime(data_source[date_col], infer_datetime_format=True)
                
                # Crear copia optimizada solo con las columnas necesarias
                df_optimized = data_source[[date_col, glucose_col]].copy()
                
                # Renombrar columnas si es necesario para mantener consistencia interna
                if date_col != "time":
                    df_optimized.rename(columns={date_col: "time"}, inplace=True)
                if glucose_col != "glucose":
                    df_optimized.rename(columns={glucose_col: "glucose"}, inplace=True)
                
                # Inicializar con el DataFrame optimizado
                super().__init__(df_optimized, "time", "glucose", delimiter, header, start_date, end_date, log)
                return
        
        # Si no es un DataFrame optimizado, usar el constructor normal
        super().__init__(data_source, date_col, glucose_col, delimiter, header, start_date, end_date, log)

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
    
    def _split_data(self, frequency: str = 'W') -> dict:
        """
        Divide los datos de glucosa en arrays separados según la frecuencia especificada.
        
        :param frequency: Frecuencia de división ('D'=diario, 'W'=semanal, 'M'=mensual)
        :return: Diccionario {periodo: {'glucose': array, 'time': array, 'metadata': dict}}
        """
        
        # Validar frecuencia
        freq_map = {
            'D': ('D', 'día'),
            'W': ('W-MON', 'semana'),
            'M': ('M', 'mes')
        }
        
        if frequency not in freq_map:
            raise ValueError(f"Frecuencia '{frequency}' no soportada. Usar 'D', 'W' o 'M'")
            
        pd_freq, label = freq_map[frequency]
        
        # Crear copia eficiente con solo las columnas necesarias
        df = self.data[['time', 'glucose']].copy()
        
        # Generar etiquetas de periodo según la frecuencia
        if frequency == 'D':
            df['periodo'] = df['time'].dt.date
        elif frequency == 'W':
            # Usar lunes como inicio de semana (estándar ISO)
            df['periodo'] = df['time'].dt.to_period('W-MON').dt.start_time.dt.date
        elif frequency == 'M':
            # Usar primer día del mes
            df['periodo'] = df['time'].dt.to_period('M').dt.start_time.dt.date
            
        # Agrupar datos por periodo
        periodos = {}
        for periodo, grupo in df.groupby('periodo'):
            # Calcular metadatos del periodo
            start_time = grupo['time'].min()
            end_time = grupo['time'].max()
            duration = (end_time - start_time).total_seconds() / 3600  # horas
            
            # Calcular completitud de datos
            expected_points = duration * 60 / self.typical_interval
            completeness = (len(grupo) / expected_points) * 100 if expected_points > 0 else 0
            
            # Guardar arrays y metadatos
            periodos[periodo] = {
                'glucose': grupo['glucose'].values,  # NumPy array para cálculos rápidos
                'time': grupo['time'].values,
                'metadata': {
                    'start': start_time,
                    'end': end_time,
                    'duration_hours': duration,
                    'completeness': completeness,
                    'n_points': len(grupo)
                }
            }
            
        return periodos
        
    def calculate_metrics(self, metric_func: callable, frequency: str = 'W', 
                         min_data_completeness: float = 70.0) -> dict:
        """
        Calcula una métrica específica para cada periodo según la frecuencia.
        
        :param metric_func: Función o nombre de método que calcula la métrica
        :param frequency: Frecuencia de división ('D'=diario, 'W'=semanal, 'M'=mensual)
        :param min_data_completeness: Porcentaje mínimo de datos requerido
        :return: Diccionario {periodo: valor_métrica}
        """
        # Dividir datos por periodo
        periodos = self._split_data(frequency)
        
        # Preparar función de métrica
        if isinstance(metric_func, str):
            # Si es nombre de método, obtener el método
            if not hasattr(self, metric_func):
                raise ValueError(f"Método '{metric_func}' no existe en GlucoseMetrics")
            func = getattr(self, metric_func)
        else:
            # Si es función, usarla directamente
            func = metric_func
            
        # Calcular métrica para cada periodo
        resultados = {}
        for periodo, datos in periodos.items():
            # Verificar completitud mínima
            if datos['metadata']['completeness'] < min_data_completeness:
                continue
                
            # Calcular métrica
            try:
                if isinstance(metric_func, str):
                    # Para métodos que requieren self, crear instancia temporal
                    temp_gm = GlucoseMetrics(
                        pd.DataFrame({
                            'time': datos['time'], 
                            'glucose': datos['glucose']
                        })
                    )
                    resultado = getattr(temp_gm, metric_func)()
                else:
                    # Para funciones, pasar directamente el array de glucosa
                    resultado = func(datos['glucose'])
                    
                # Guardar resultado con metadatos
                resultados[periodo] = {
                    'value': resultado,
                    'metadata': datos['metadata']
                }
            except Exception as e:
                # Manejar errores en cálculo de métrica
                print(f"Error calculando métrica para {periodo}: {str(e)}")
                
        return resultados
        
    def calculate_multiple_metrics(self, metric_funcs: list, frequency: str = 'W',
                                 min_data_completeness: float = 70.0) -> dict:
        """
        Calcula múltiples métricas para cada periodo según la frecuencia.
        
        :param metric_funcs: Lista de funciones o nombres de métodos
        :param frequency: Frecuencia de división ('D'=diario, 'W'=semanal, 'M'=mensual)
        :param min_data_completeness: Porcentaje mínimo de datos requerido
        :return: Diccionario {periodo: {métrica1: valor1, métrica2: valor2, ...}}
        """
        # Dividir datos por periodo (una sola vez)
        periodos = self._split_data(frequency)
        
        # Inicializar resultados
        resultados = {periodo: {'metadata': datos['metadata']} 
                     for periodo, datos in periodos.items() 
                     if datos['metadata']['completeness'] >= min_data_completeness}
        
        # Calcular cada métrica para todos los periodos válidos
        for metric_name in metric_funcs:
            for periodo in resultados.keys():
                datos = periodos[periodo]
                
                try:
                    # Calcular métrica
                    if isinstance(metric_name, str):
                        temp_gm = GlucoseMetrics(
                            pd.DataFrame({
                                'time': datos['time'], 
                                'glucose': datos['glucose']
                            })
                        )
                        resultado = getattr(temp_gm, metric_name)()
                    else:
                        resultado = metric_name(datos['glucose'])
                        
                    # Guardar resultado
                    resultados[periodo][metric_name if isinstance(metric_name, str) else metric_name.__name__] = resultado
                except Exception as e:
                    print(f"Error calculando {metric_name} para {periodo}: {str(e)}")
                    
        return resultados

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
        Calcula la desviación estándar dentro del día (SDw).
        
        Esta métrica refleja la variabilidad dentro de cada día, promediada
        entre todos los días disponibles.
        
        Versión optimizada para grandes conjuntos de datos.
        
        :param min_count_threshold: Umbral para considerar un día como válido
                               (proporción de la mediana de conteos).
        :return: Diccionario con el valor de SDw y estadísticas relacionadas.
        """
        # Crear copia eficiente con solo las columnas necesarias
        df = self.data[['time', 'glucose']].copy()
        
        # Extraer fecha de forma vectorizada
        df['date'] = df['time'].dt.date
        
        # Calcular estadísticas por día de forma vectorizada
        daily_stats = df.groupby('date').agg({'glucose': ['std', 'mean', 'count']})
        
        # Calcular umbral para filtrar días con pocos datos
        median_count = daily_stats[('glucose', 'count')].median()
        threshold = median_count * min_count_threshold
        
        # Guardar información sobre todos los días
        all_days_sds = daily_stats[('glucose', 'std')].to_dict()
        all_days_means = daily_stats[('glucose', 'mean')].to_dict()
        all_days_counts = daily_stats[('glucose', 'count')].to_dict()
        
        # Filtrar días con suficientes datos
        valid_days = daily_stats[daily_stats[('glucose', 'count')] >= threshold]
        
        if valid_days.empty:
            return {
                'sd': 0.0, 
                'mean': 0.0, 
                'valid_days': 0,
                'total_days': len(daily_stats),
                'daily_sds': {}, 
                'daily_means': {}, 
                'daily_counts': {},
                'all_days_sds': all_days_sds,
                'all_days_means': all_days_means,
                'all_days_counts': all_days_counts,
                'threshold': threshold
            }
        
        # Calcular SDw (promedio de las desviaciones estándar diarias)
        sd_value = valid_days[('glucose', 'std')].mean()
        mean_value = valid_days[('glucose', 'mean')].mean()
        
        # Preparar resultado con estadísticas adicionales
        result = {
            'sd': sd_value,
            'mean': mean_value,
            'valid_days': len(valid_days),
            'total_days': len(daily_stats),
            'daily_sds': valid_days[('glucose', 'std')].to_dict(),
            'daily_means': valid_days[('glucose', 'mean')].to_dict(),
            'daily_counts': valid_days[('glucose', 'count')].to_dict(),
            'all_days_sds': all_days_sds,
            'all_days_means': all_days_means,
            'all_days_counts': all_days_counts,
            'threshold': threshold
        }
        
        return result
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
        Calcula la desviación estándar entre puntos temporales (SDhh:mm). Calcula la media de una marca temporal y luego la desviación estándar de esas medias.
        
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
        Calcula SDws y media de las series temporales. 
        
        A menor número de horas, más pequeño es el valor de SDws porque 
        da menos tiempo para que la glucosa varíe.
        
        Versión optimizada para grandes conjuntos de datos.
        
        :param hours: Tamaño de la ventana en horas
        :return: Diccionario con SD y media promedio de las series temporales
        """
        # Crear copia eficiente con solo las columnas necesarias
        df = self.data[['time', 'glucose']].copy()
        
        # Asegurar que los datos estén ordenados por tiempo
        df = df.sort_values('time')
        
        # Convertir horas a nanosegundos para la ventana de tiempo
        window_ns = pd.Timedelta(hours=hours).value
        
        # Matriz para almacenar resultados
        sd_values = []
        mean_values = []
        
        # Tomar muestras a intervalos regulares para reducir la carga computacional
        # Ajusta el step según el tamaño de tu dataset (mayor step = más rápido pero menos preciso)
        step = max(1, len(df) // 1000)  # Limitar a ~1000 ventanas como máximo
        
        for i in range(0, len(df), step):
            start_time = df.iloc[i]['time']
            end_time = start_time + pd.Timedelta(hours=hours)
            
            # Filtrar los datos en la ventana de tiempo actual
            window_data = df[(df['time'] >= start_time) & (df['time'] < end_time)]
            
            # Solo calcular estadísticas si hay suficientes puntos en la ventana
            if len(window_data) > 2:  # Necesitamos al menos 3 puntos para una buena estimación
                sd_values.append(window_data['glucose'].std())
                mean_values.append(window_data['glucose'].mean())
        
        # Calcular promedios
        result = {
            'sd': np.mean(sd_values) if sd_values else 0.0,
            'mean': np.mean(mean_values) if mean_values else 0.0,
            'windows_analyzed': len(sd_values)
        }
        
        return result
    
    def sd_daily_mean(self, min_count_threshold: float = 0.5) -> dict:
        """
        Calcula la desviación estándar de las medias diarias (SDdm).
        
        Esta métrica refleja la variabilidad entre diferentes días.
        
        Versión optimizada para grandes conjuntos de datos.
        
        :param min_count_threshold: Umbral para considerar un día como válido
                               (proporción de la mediana de conteos).
        :return: Diccionario con el valor de SDdm y estadísticas relacionadas.
        """
        # Crear copia eficiente con solo las columnas necesarias
        df = self.data[['time', 'glucose']].copy()
        
        # Extraer fecha de forma vectorizada
        df['date'] = df['time'].dt.date
        
        # Calcular estadísticas por día de forma vectorizada
        daily_stats = df.groupby('date').agg({'glucose': ['mean', 'count']})
        
        # Calcular umbral para filtrar días con pocos datos
        median_count = daily_stats[('glucose', 'count')].median()
        threshold = median_count * min_count_threshold
        
        # Filtrar días con suficientes datos
        valid_days = daily_stats[daily_stats[('glucose', 'count')] >= threshold]
        
        if valid_days.empty:
            return {'sd': 0.0, 'mean': 0.0}
        
        # Calcular SDdm (SD de las medias diarias)
        sd_value = valid_days[('glucose', 'mean')].std()
        mean_value = valid_days[('glucose', 'mean')].mean()
        
        # Preparar resultado con estadísticas adicionales
        result = {
            'sd': sd_value,
            'mean': mean_value,
            'valid_days': len(valid_days),
            'total_days': len(daily_stats),
            'daily_means': valid_days[('glucose', 'mean')].to_dict(),
            'daily_counts': valid_days[('glucose', 'count')].to_dict()
        }
        
        return result

    def sd_same_timepoint(self, min_count_threshold: float = 0.5, filter_outliers: bool = True, 
                         agrupar_por_intervalos: bool = False, intervalo_minutos: int = 5) -> dict:
        """
        Calcula la desviación estándar entre días para cada punto temporal (SDbhh:mm).
        
        Esta función mide la variabilidad de la glucosa para cada punto temporal específico
        a lo largo de diferentes días, lo que refleja la consistencia día a día.
        
        Versión optimizada para grandes conjuntos de datos.
        
        :param min_count_threshold: Umbral para considerar una marca temporal como válida
                               (proporción de la mediana de conteos).
        :param filter_outliers: Si es True, filtra las marcas temporales con pocos datos.
        :param agrupar_por_intervalos: Si es True, agrupa los datos en intervalos regulares.
        :param intervalo_minutos: Tamaño del intervalo en minutos (por defecto 5 min).
        :return: Diccionario con el valor de SDbhh:mm y estadísticas relacionadas.
        """
        # Crear copia eficiente con solo las columnas necesarias
        df = self.data[['time', 'glucose']].copy()
        
        # Extraer características temporales vectorizadamente
        df['hour'] = df['time'].dt.hour
        df['minute'] = df['time'].dt.minute
        df['day'] = df['time'].dt.date
        
        if agrupar_por_intervalos:
            # Agrupar por intervalos de tiempo
            minutes_of_day = df['hour'] * 60 + df['minute']
            df['interval'] = (minutes_of_day // intervalo_minutos) * intervalo_minutos
            df['hour'] = df['interval'] // 60
            df['minute'] = df['interval'] % 60
        
        # Crear clave de tiempo para agrupación
        df['time_key'] = df['hour'].astype(str).str.zfill(2) + ':' + df['minute'].astype(str).str.zfill(2)
        
        # Agrupar por día y punto temporal
        grouped = df.groupby(['day', 'time_key'])
        
        # Calcular promedios por día y punto temporal
        daily_means = grouped['glucose'].mean().reset_index()
        
        # Agrupar por punto temporal y calcular estadísticas
        timepoint_stats = daily_means.groupby('time_key')
        
        # Calcular estadísticas por punto temporal de forma vectorizada
        sd_por_marca = timepoint_stats['glucose'].std()
        valores_por_marca = timepoint_stats['glucose'].mean()
        conteo_por_marca = timepoint_stats['glucose'].count()
        
        # Calcular umbral para filtrar puntos con pocos datos
        median_count = conteo_por_marca.median()
        threshold = median_count * min_count_threshold
        
        # Guardar el total de puntos temporales antes del filtrado
        total_timepoints = len(sd_por_marca)
        
        # Filtrar puntos temporales con suficientes datos si se solicita
        if filter_outliers:
            valid_mask = conteo_por_marca >= threshold
            sd_por_marca = sd_por_marca[valid_mask]
            valores_por_marca = valores_por_marca[valid_mask]
            conteo_por_marca = conteo_por_marca[valid_mask]
        
        if len(sd_por_marca) == 0:
            return {'sd': 0.0, 'mean': 0.0, 'threshold': threshold, 'total_timepoints': total_timepoints}
        
        # Calcular SDbhh:mm (promedio ponderado de las desviaciones estándar)
        weights = conteo_por_marca / conteo_por_marca.sum()
        sd_value = (sd_por_marca * weights).sum()
        mean_value = valores_por_marca.mean()
        
        # Preparar resultado con estadísticas adicionales
        result = {
            'sd': sd_value,
            'mean': mean_value,
            'sd_por_marca': sd_por_marca.to_dict(),
            'valores_por_marca': valores_por_marca.to_dict(),
            'conteo_por_marca': conteo_por_marca.to_dict(),
            'threshold': threshold,
            'total_timepoints': total_timepoints
        }
        
        return result

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
        Calcula la desviación estándar de interacción (SDI).
        
        SDI cuantifica la variabilidad diaria en el patrón glucémico,
        considerando las interacciones entre la hora del día y el día específico.
        
        Versión optimizada para grandes conjuntos de datos.
        
        :return: Diccionario con el valor de SDI y estadísticas relacionadas.
        """
        # Crear copia eficiente con solo las columnas necesarias
        df = self.data[['time', 'glucose']].copy()
        
        # Extraer características temporales de forma vectorizada
        df['hour'] = df['time'].dt.hour
        df['minute'] = df['time'].dt.minute
        df['day'] = df['time'].dt.date
        df['time_key'] = df['hour'].astype(str).str.zfill(2) + ':' + df['minute'].astype(str).str.zfill(2)
        
        # Calcular valores necesarios para la fórmula de SDI
        
        # 1. Calcular la media global
        global_mean = df['glucose'].mean()
        
        # 2. Calcular las medias diarias
        daily_means = df.groupby('day')['glucose'].mean()
        
        # 3. Calcular las medias por punto temporal
        timepoint_means = df.groupby('time_key')['glucose'].mean()
        
        # 4. Calcular las desviaciones para cada observación
        # Usamos merge para operaciones vectorizadas eficientes
        df_temp = df.copy()
        
        # Convertir day_mean y timepoint_mean a DataFrames para merge
        day_mean_df = pd.DataFrame(daily_means).reset_index()
        day_mean_df.columns = ['day', 'day_mean']
        
        timepoint_mean_df = pd.DataFrame(timepoint_means).reset_index()
        timepoint_mean_df.columns = ['time_key', 'timepoint_mean']
        
        # Hacer merge de forma eficiente
        df_temp = df_temp.merge(day_mean_df, on='day')
        df_temp = df_temp.merge(timepoint_mean_df, on='time_key')
        
        # Calcular la interacción para cada punto
        df_temp['expected'] = global_mean + (df_temp['day_mean'] - global_mean) + (df_temp['timepoint_mean'] - global_mean)
        df_temp['interaction'] = df_temp['glucose'] - df_temp['expected']
        
        # Calcular SDI (desviación estándar de la interacción)
        sdi = df_temp['interaction'].std()
        
        return {
            'sd': sdi,
            'mean': global_mean
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
        Calcula todas las métricas de coeficiente de variación disponibles. Falta revisar sobre todo las medias si están bien.
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
        from sklearn.linear_model import LinearRegression
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

    def variance_components(self) -> dict:
        """
        Calcula y descompone la varianza total de glucosa en sus componentes principales.
        
        Incluye descomposición en: interdía, intradía (patrón, interacción, residual)
        y análisis por segmentos del día.
        
        :return: Diccionario con los componentes de la varianza y sus porcentajes
        """
        # Crear copia eficiente con solo las columnas necesarias
        df = self.data[['time', 'glucose']].copy()
        
        # Añadir columnas para día y hora del día
        df['day'] = df['time'].dt.date
        df['hour_min'] = df['time'].dt.hour * 60 + df['time'].dt.minute
        
        # Añadir columna para segmento del día (0=noche, 1=día, 2=tarde)
        df['segment'] = pd.cut(
            df['time'].dt.hour, 
            bins=[0, 8, 16, 24], 
            labels=[0, 1, 2], 
            include_lowest=True
        )
        
        # 1. Calcular la varianza total
        total_variance = df['glucose'].var()
        
        # 2. Calcular la varianza interdía (entre días)
        daily_means = df.groupby('day')['glucose'].mean()
        interdía_variance = daily_means.var()
        
        # 3. Calcular la varianza intradía (dentro del día)
        intradía_variance = total_variance - interdía_variance
        
        # 4. Descomposición de la varianza intradía
        
        # 4.1 Varianza del patrón (efecto hora)
        hourly_means = df.groupby('hour_min')['glucose'].mean()
        patrón_variance = hourly_means.var()
        
        # 4.2 Varianza de interacción (día x hora)
        day_hour_means = df.groupby(['day', 'hour_min'])['glucose'].mean()
        interacción_variance = day_hour_means.var() - patrón_variance - interdía_variance
        
        # 4.3 Varianza residual
        residual_variance = total_variance - (interdía_variance + patrón_variance + interacción_variance)
        
        # Corregir el cálculo de varianza residual para evitar valores negativos
        if residual_variance < 0:
            # Si es negativo pero cercano a cero (error numérico), ajustar a cero
            if abs(residual_variance) < 0.01 * total_variance:  # Umbral de tolerancia: 1% de la varianza total
                residual_variance = 0
            else:
                # Si es significativamente negativo, ajustar la varianza de interacción
                # para mantener la consistencia matemática
                adjustment = abs(residual_variance)
                interacción_variance -= adjustment
                residual_variance = 0
        
        # 5. Calcular porcentajes sobre la varianza total
        porcentaje_interdía = (interdía_variance / total_variance) * 100
        porcentaje_intradía = (intradía_variance / total_variance) * 100
        
        # 6. Calcular porcentajes de cada componente como proporción de la varianza intradía
        if intradía_variance > 0:
            # Calcular directamente como porcentaje de la varianza intradía
            porcentaje_patrón = (patrón_variance / intradía_variance) * 100
            porcentaje_interacción = (interacción_variance / intradía_variance) * 100
            porcentaje_residual = (residual_variance / intradía_variance) * 100
            
            # Normalizar para garantizar que suman 100%
            total_intraday_pct = porcentaje_patrón + porcentaje_interacción + porcentaje_residual
            if total_intraday_pct > 0:
                porcentaje_patrón = (porcentaje_patrón / total_intraday_pct) * 100
                porcentaje_interacción = (porcentaje_interacción / total_intraday_pct) * 100
                porcentaje_residual = (porcentaje_residual / total_intraday_pct) * 100
        else:
            porcentaje_patrón = 0
            porcentaje_interacción = 0
            porcentaje_residual = 0
        
        # 7. Varianza por segmentos del día
        segment_names = {0: 'noche', 1: 'día', 2: 'tarde'}
        
        # Calcular varianza, medias y conteos para cada segmento
        segment_variances = {}
        segment_means = {}
        segment_counts = {}
        
        for seg_id, name in segment_names.items():
            segment_data = df[df['segment'] == seg_id]['glucose']
            segment_counts[name] = len(segment_data)
            
            if segment_counts[name] > 0:
                segment_variances[name] = segment_data.var()
                segment_means[name] = segment_data.mean()
            else:
                segment_variances[name] = 0
                segment_means[name] = 0
        
        # Calcular varianza entre las medias de los segmentos
        if sum(segment_counts.values()) > 0:
            segment_means_array = np.array([segment_means[name] for name in segment_names.values() if segment_counts[name] > 0])
            segment_weights = np.array([segment_counts[name] for name in segment_names.values() if segment_counts[name] > 0])
            
            if len(segment_means_array) > 1:
                segment_means_weighted_avg = np.average(segment_means_array, weights=segment_weights)
                between_segments_variance = np.average((segment_means_array - segment_means_weighted_avg)**2, weights=segment_weights)
            else:
                between_segments_variance = 0
        else:
            between_segments_variance = 0
        
        # Calcular porcentajes sobre la varianza intradía
        segmentos_porcentaje = {}
        for name in segment_names.values():
            if segment_counts[name] > 0 and intradía_variance > 0:
                within_segment_contribution = segment_variances[name] * segment_counts[name] / sum(segment_counts.values())
                segmentos_porcentaje[name] = (within_segment_contribution / intradía_variance) * 100
            else:
                segmentos_porcentaje[name] = 0
        
        # Normalizar los porcentajes de segmentos para que sumen 100%
        total_segment_pct = sum(segmentos_porcentaje.values())
        if total_segment_pct > 0:
            for name in segmentos_porcentaje:
                segmentos_porcentaje[name] = (segmentos_porcentaje[name] / total_segment_pct) * 100
        
        # Armar el resultado
        resultado = {
            'varianza_total': total_variance,
            'varianza_interdía': interdía_variance,
            'varianza_intradía': intradía_variance,
            'varianza_patrón': patrón_variance,
            'varianza_interacción': interacción_variance,
            'varianza_residual': residual_variance,
            'porcentaje_interdía': porcentaje_interdía,
            'porcentaje_intradía': porcentaje_intradía,
            'porcentaje_patrón': porcentaje_patrón,
            'porcentaje_interacción': porcentaje_interacción,
            'porcentaje_residual': porcentaje_residual,
            
            # Añadir información por segmentos
            'varianza_segmentos': segment_variances,
            'varianza_entre_segmentos': between_segments_variance,
            'porcentaje_segmentos': segmentos_porcentaje,
            'conteos_segmentos': segment_counts
        }
        
        # También incluir las desviaciones estándar
        for key in ['total', 'interdía', 'intradía', 'patrón', 'interacción', 'residual']:
            variance_key = f'varianza_{key}'
            if variance_key in resultado and resultado[variance_key] > 0:
                resultado[f'sd_{key}'] = np.sqrt(resultado[variance_key])
            else:
                resultado[f'sd_{key}'] = 0
        
        # Añadir SD para segmentos
        resultado['sd_segmentos'] = {
            name: np.sqrt(var) if var > 0 else 0 
            for name, var in segment_variances.items()
        }
        
        return resultado
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
    
    
    def MODD(self, days: int = 1) -> dict:
        """
        Calcula el MODD (Mean Of Daily Differences) para múltiples días.
        
        El MODD es una medida de la variabilidad entre días, calculada como la media 
        de la diferencia absoluta de los valores de glucosa obtenidos exactamente a la 
        misma hora del día, entre días consecutivos o separados por un número específico de días.
        
        :param days: Número de días para calcular diferencias (1-6)
        :return: Diccionario con valores MODD y estadísticas relacionadas
        :reference: DOI: 10.1007/BF01218495
        """
        if not 1 <= days <= 6:
            raise ValueError("El número de días debe estar entre 1 y 6")
        
        # Crear copia de datos con información de tiempo
        data_copy = self.data.copy()
        
        # Extraer componentes de tiempo para comparación exacta por hora del día
        data_copy['date'] = data_copy['time'].dt.date
        data_copy['time_of_day'] = data_copy['time'].dt.strftime('%H:%M:%S')
        
        results = {}
        correlations = []
        
        for d in range(1, days + 1):
            # Agrupar por hora del día para comparar valores separados por d días
            grouped = data_copy.groupby('time_of_day')
            
            abs_diffs = []
            day_pairs = []
            
            for _, group in grouped:
                # Ordenar por fecha para cada hora del día
                sorted_group = group.sort_values('date')
                
                # Crear pares de días separados por d días
                for i in range(len(sorted_group) - d):
                    if (sorted_group.iloc[i+d]['date'] - sorted_group.iloc[i]['date']).days == d:
                        # Calcular diferencia absoluta
                        abs_diff = abs(sorted_group.iloc[i+d]['glucose'] - sorted_group.iloc[i]['glucose'])
                        abs_diffs.append(abs_diff)
                        
                        # Guardar par de valores para calcular correlación
                        day_pairs.append((sorted_group.iloc[i]['glucose'], sorted_group.iloc[i+d]['glucose']))
            
            if abs_diffs:
                # Calcular MODD para d días
                modd_value = np.mean(abs_diffs)
                
                # Calcular correlación entre días
                if len(day_pairs) > 1:
                    day1_values, day2_values = zip(*day_pairs)
                    corr = np.corrcoef(day1_values, day2_values)[0, 1]
                else:
                    corr = None
                
                results[f'MODD{d}'] = {
                    'value': modd_value,
                    'n_observations': len(abs_diffs),
                    'std': np.std(abs_diffs) if len(abs_diffs) > 1 else 0,
                    'correlation': corr
                }
                
                if corr is not None:
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
    
    def CONGA(self, hours: int = 4, max_gap_minutes: float = None) -> dict:
        """
        Calcula CONGA (Continuous Overlapping Net Glycemic Action).
        
        CONGA mide la variabilidad intradiaria de la glucemia calculando la desviación
        estándar de las diferencias entre valores actuales y valores de 'n' horas antes.
        
        :param hours: Número de horas para el intervalo de tiempo (n)
        :param max_gap_minutes: Brecha máxima permitida en minutos entre mediciones para 
                           considerar válida una comparación. Si es None, se usa 2 veces
                           el intervalo típico.
        :return: Diccionario con valor CONGA y estadísticas relacionadas
        :reference: McDonnell CM, et al. Diabetes Technol Ther. 2005;7(2):243-9.
                   DOI: 10.1089/dia.2005.7.243
        """
        # Crear copia de datos ordenados por tiempo
        df = self.data.sort_values('time').copy()
        
        # Calcular el intervalo en minutos
        interval_minutes = self.typical_interval  # Ya está en minutos
        
        # Si no se especifica max_gap_minutes, usar 2 veces el intervalo típico
        if max_gap_minutes is None:
            max_gap_minutes = 2 * interval_minutes
        
        # Calcular cuántos intervalos corresponden a 'hours' horas
        n_intervals = int((hours * 60) / interval_minutes)
        
        if n_intervals <= 0:
            raise ValueError(f"El intervalo de {hours} horas es demasiado pequeño para los datos disponibles")
        
        # Calcular diferencias entre valores actuales y valores de 'n' horas antes
        # pero teniendo en cuenta posibles desconexiones
        
        # Método 1: Usando shift pero verificando la diferencia de tiempo real
        df['time_n_hours_ago'] = df['time'].shift(n_intervals)
        df['glucose_n_hours_ago'] = df['glucose'].shift(n_intervals)
        
        # Calcular la diferencia de tiempo real en minutos
        df['time_diff_minutes'] = (df['time'] - df['time_n_hours_ago']).dt.total_seconds() / 60
        
        # Calcular diferencia de glucosa solo si la diferencia de tiempo está cerca del objetivo
        target_diff_minutes = hours * 60
        df['valid_comparison'] = (
            (df['time_diff_minutes'] >= target_diff_minutes - max_gap_minutes) & 
            (df['time_diff_minutes'] <= target_diff_minutes + max_gap_minutes)
        )
        
        # Calcular diferencia solo para comparaciones válidas
        df['difference'] = np.where(
            df['valid_comparison'],
            df['glucose'] - df['glucose_n_hours_ago'],
            np.nan
        )
        
        # Eliminar filas con valores faltantes o comparaciones inválidas
        valid_data = df.dropna(subset=['difference'])
        
        if len(valid_data) == 0:
            return {
                'value': None,
                'n_observations': 0,
                'mean_difference': None,
                'abs_mean_difference': None,
                'std': None,
                'hours': hours,
                'max_gap_minutes': max_gap_minutes
            }
        
        # Calcular CONGA como la desviación estándar de las diferencias
        conga_value = valid_data['difference'].std()
        
        # Calcular estadísticas adicionales
        mean_diff = valid_data['difference'].mean()
        abs_mean_diff = valid_data['difference'].abs().mean()
        
        # Información sobre desconexiones
        total_comparisons = len(df.dropna(subset=['glucose_n_hours_ago']))
        valid_comparisons = len(valid_data)
        invalid_comparisons = total_comparisons - valid_comparisons
        
        return {
            'value': conga_value,
            'n_observations': len(valid_data),
            'mean_difference': mean_diff,
            'abs_mean_difference': abs_mean_diff,
            'hours': hours,
            'max_gap_minutes': max_gap_minutes,
            'total_comparisons': total_comparisons,
            'valid_comparisons': valid_comparisons,
            'invalid_comparisons': invalid_comparisons,
            'percent_valid': (valid_comparisons / total_comparisons * 100) if total_comparisons > 0 else 0
        }

    def Lability_index(self, interval: int = 1, period: str = 'week') -> dict:
        # Añadimos timing para ver dónde se gasta el tiempo
        
        data_copy = self.data.copy()
        data_copy['time_rounded'] = data_copy['time'].dt.floor('h')
        data_copy['week'] = data_copy['time'].dt.isocalendar().week
 

        weekly_li = []
        
        for week, group in data_copy.groupby('week'):
            
            group = group.sort_values('time_rounded')
            
            # Versión vectorizada dentro del grupo
            glucose_diffs = group['glucose'].shift(-interval) - group['glucose']
            li_values = (glucose_diffs ** 2) / interval
            li_week = li_values.dropna().sum()
            weekly_li.append(li_week)
            
        mean_li = np.mean(weekly_li) if weekly_li else 0
        mean_li_mmol = mean_li / (18.0 ** 2)
        
        # Añadimos la interpretación clínica
        mean_li_por_hora = mean_li / 168
        cambio_tipico_por_hora = math.sqrt(mean_li_por_hora)
        
        return {
            'weekly_values': weekly_li,
            'mean_li': mean_li,
            'mean_li_mmol': mean_li_mmol,
            'std_li': np.std(weekly_li) if len(weekly_li) > 1 else 0,
            'n_weeks': len(weekly_li),
            # Nuevos campos de interpretación clínica
            'cambio_tipico_por_hora': cambio_tipico_por_hora,
        }

    def Variability(self) -> str:
        """
        Calcula todas las métricas de variabilidad.
        :return: Un string JSON con todas las métricas de variabilidad.
        """
        variability_metrics = {
            "CONGA1": self.CONGA(hours=1),
            "CONGA2": self.CONGA(hours=2),
            "CONGA4": self.CONGA(hours=4),
            "CONGA6": self.CONGA(hours=6),
            "CONGA24": self.CONGA(hours=24),
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
        return variability_metrics
    
    ## MEDIDAS DE LA CALIDAD DE GLUCEMIA
    
    def M_Value(self, reference_glucose: int = 90) -> dict:
        """
        Calcula el M-Value según la definición de Schlichtkrull y consideración de Service
        
        M-Value es un híbrido entre:
        1. Desviación de la glucemia media
        2. Variabilidad glucémica
        
        Características especiales:
        - Da mayor peso a la hipoglucemia que a la hiperglucemia
        - Usa 90 mg/dL como valor de referencia histórico. Artículo original usaban 120 mg/dL
        - Combina desviación media y amplitud de fluctuación
        
        Fórmula: M = (1/n)∑|10 * log10(BG/120)|³ + W/20 (El factor de corrección se puede obviar cuando hay mas de 24 datos)
        
        :param reference_glucose: Valor de referencia (default 120 mg/dL)
        :return: Diccionario con M-Value y componentes
        :reference: 10.1111/j.0954-6820.1965.tb01810.x
        :reference: 10.2337/db12-1396
        """
        # Convertir directamente a array de NumPy para operaciones más rápidas
        glucose_values = self.data['glucose'].values
        
        # Calcular M_BS vectorizado
        M_BS_values = np.abs(10 * np.log10(glucose_values/reference_glucose))**3
        M_BS_mean = np.mean(M_BS_values)
        return round(M_BS_mean, 2)
    

    def j_index(self) -> float:
        """Calcula el J-index.
        DOI: 10.1055/s-2007-979906
        """
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

    def GRI(self) -> dict:
        """
        Calcula el Glucose Risk Index (GRI).
        
        GRI combina los tiempos en diferentes rangos de glucosa, dando diferentes pesos
        a la hipoglucemia y la hiperglucemia:
        
        GRI = (3.0 × VLow) + (2.4 × Low) + (1.6 × VHigh) + (0.8 × High)
        
        Donde:
        - VLow: % tiempo en hipoglucemia muy baja (<54 mg/dL)
        - Low: % tiempo en hipoglucemia leve (54-70 mg/dL)
        - VHigh: % tiempo en hiperglucemia muy alta (>250 mg/dL)
        - High: % tiempo en hiperglucemia alta (180-250 mg/dL)
        
        :return: Diccionario con el GRI y sus componentes
        :reference: DOI: 10.1016/j.diabres.2013.03.006
        """
        # Calcular los porcentajes de tiempo en cada rango
        vlow = self.TBR(54)  # <54 mg/dL
        low = self.calculate_time_in_range(54, 70)  # 54-70 mg/dL
        vhigh = self.TAR(250)  # >250 mg/dL
        high = self.calculate_time_in_range(180, 250)  # 180-250 mg/dL
        
        # Calcular los componentes del GRI
        hypo_component = vlow + (0.8 * low)
        hyper_component = vhigh + (0.5 * high)
        
        # Calcular el GRI
        gri = (3.0 * vlow) + (2.4 * low) + (1.6 * vhigh) + (0.8 * high)
        
        # Calcular el TIR (tiempo en rango)
        tir = 100 - (vlow + low + vhigh + high)
        
        return {
            'GRI': round(gri, 2),
            'components': {
                'VLow': round(vlow, 2),
                'Low': round(low, 2),
                'VHigh': round(vhigh, 2),
                'High': round(high, 2)
            },
            'derived_metrics': {
                'hypo_component': round(hypo_component, 2),
                'hyper_component': round(hyper_component, 2),
                'TIR': round(tir, 2)
            }
        }

