import datetime
from .glucose_data import GlucoseData
from typing import Union
import numpy as np
import json
import pandas as pd

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
            diferencias = self.data.sort_values('time')['time'].diff()
            interval_minutes = diferencias.median().total_seconds() / 60

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
            'TBR70': self.TBR70(),
            'TBR55': self.TBR55(),
            'TAR250': self.TAR250(),
            'TAR180': self.TAR180(),
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
            'media': self.data['glucose'].mean(),
            'mediana': self.data['glucose'].median(),
            'desviacion_estandar': self.data['glucose'].std(),
            'coef_variacion': (self.data['glucose'].std() / self.data['glucose'].mean()) * 100,
            'asimetria': self.data['glucose'].skew(),
            'curtosis': self.data['glucose'].kurtosis(),
            'percentiles': {
                'p25': self.data['glucose'].quantile(0.25),
                'p75': self.data['glucose'].quantile(0.75),
                'IQR': self.data['glucose'].quantile(0.75) - self.data['glucose'].quantile(0.25)
            }
        }
        return stats
        
    ## VARIABILIDAD

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

    def MODD(self, min: int = 5) -> float:
        """
        Calcula el MODD (Mean Of Daily Differences).
        :param min: Intervalo en minutos entre mediciones.
        :return: Valor de MODD.
        :reference: DOI: 10.1007/BF01218495
        """
        intervals = 24 * int(60/min)
        self.data['glucose_24h_ago'] = self.data['glucose'].shift(intervals)
        valid_data = self.data.dropna(subset=['glucose_24h_ago']).copy()
        valid_data['abs_diff'] = (valid_data['glucose'] - valid_data['glucose_24h_ago']).abs()
        return valid_data['abs_diff'].mean()

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
            "MODD": self.MODD(min=5),
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
