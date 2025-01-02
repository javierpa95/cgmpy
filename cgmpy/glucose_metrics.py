from .glucose_data import GlucoseData
from typing import Union
import numpy as np
import json

class GlucoseMetrics(GlucoseData):
    def __init__(self, file_path: str, date_col: str, glucose_col: str, delimiter: Union[str, None] = None, header: int = 0):

        super().__init__(file_path, date_col, glucose_col, delimiter, header)

    ## ESTADÍSTICAS BÁSICAS
    def mean(self) -> float:
        """Calcula la glucemia media."""
        return self.data['glucose'].mean()
    
    def median(self) -> float:
        """Calcula la mediana de la glucemia."""
        return self.data['glucose'].median()

    def sd(self) -> float:
        """Calcula la desviación estándar de la glucemia."""
        return self.data['glucose'].std()
    
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

    def TIR(self) -> float:
        """Calcula el tiempo en rango entre 70 y 180 mg/dL."""
        return self.calculate_time_in_range(70, 180)
    
    def TBR70(self) -> float:
        """Calcula el tiempo por debajo de 70 mg/dL."""
        return self.calculate_time_in_range(55, 70)
    
    def TBR55(self) -> float:
        """Calcula el tiempo por debajo de 55 mg/dL."""
        return self.TBR(55)
    
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
