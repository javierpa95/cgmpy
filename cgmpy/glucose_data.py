import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import json
from typing import Union

def parse_date(date_string):
    formats = [
        '%d/%m/%Y %H:%M',  # Formato 1: 07/10/2022 00:00
        '%Y-%m-%dT%H:%M:%S'  # Formato 2: 2023-02-01T01:08:04
    ]
    
    for fmt in formats:
        try:
            return pd.to_datetime(date_string, format=fmt)
        except ValueError:
            pass
    
    raise ValueError(f"No se pudo parsear la fecha: {date_string}")

class GlucoseData:
    def __init__(self, file_path: str, date_col: str, glucose_col: str, delimiter: Union[str, None] = ',', header: int = 0):
        """
        Inicializa la clase con los datos de glucemia a partir de un archivo CSV.
        
        :param file_path: Ruta al archivo CSV.
        :param date_col: Nombre de la columna que contiene las fechas.
        :param glucose_col: Nombre de la columna que contiene los valores de glucosa.
        :param delimiter: Delimitador usado en el archivo CSV.
        :param header: Índice de la fila que contiene los nombres de las columnas.
        :raises ValueError: Si no se puede cargar el archivo o las columnas especificadas no existen.
        """
        self.data = self._load_csv(file_path, delimiter, header)
        self._validate_columns(date_col, glucose_col)
        self._process_data(date_col, glucose_col)

    def _load_csv(self, file_path: str, delimiter: Union[str, None], header: int) -> pd.DataFrame:
        """Carga el archivo CSV."""
        if delimiter is None:
            for delim in [',', ';']:
                try:
                    return pd.read_csv(file_path, delimiter=delim, header=header, quotechar='"')
                except pd.errors.ParserError:
                    continue
            raise ValueError("No se pudo cargar el archivo CSV con los delimitadores ',' o ';'.")
        return pd.read_csv(file_path, delimiter=delimiter, header=header, quotechar='"')

    

    def _validate_columns(self, date_col: str, glucose_col: str):
        """Valida que las columnas especificadas existan en el DataFrame."""
        if date_col not in self.data.columns or glucose_col not in self.data.columns:
            raise ValueError(f"Las columnas '{date_col}' o '{glucose_col}' no se encuentran en el archivo CSV. Columnas disponibles: {self.data.columns.tolist()}.")

    def _process_data(self, date_col: str, glucose_col: str):
        """Procesa los datos, convirtiendo fechas y valores de glucosa."""
        self.data = self.data.dropna(subset=[date_col])
        self.data[date_col] = self.data[date_col].apply(parse_date)
        self.data = self.data.dropna(subset=[date_col, glucose_col])
        self.data[glucose_col] = pd.to_numeric(self.data[glucose_col], errors='coerce')
        self.data = self.data.dropna(subset=[glucose_col])
        self.data.rename(columns={date_col: 'time', glucose_col: 'glucose'}, inplace=True)
        self.data = self.data[['time', 'glucose']]

    ## INFORMACIÓN BÁSICA

    def info(self) -> str:
        """
        Muestra información básica del archivo CSV, incluyendo el número de datos y el rango de fechas.
        :return: Un string con la información básica.
        """
        num_datos = len(self.data)
        fecha_inicio = self.data['time'].min().strftime('%d/%m/%Y %H:%M')
        fecha_fin = self.data['time'].max().strftime('%d/%m/%Y %H:%M')
        return f"El archivo CSV contiene {num_datos} datos entre {fecha_inicio} y {fecha_fin}."

    ## ESTADÍSTICAS BÁSICAS
    def mean(self) -> float:
        """Calcula la glucemia media."""
        return self.data['glucose'].mean()
    
    def median(self) -> float:
        """Calcula la mediana de la glucemia."""
        return self.data['glucose'].median()

    def sd(self) -> float:
        """Calcula la desviación estándar de la glucosa."""
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
        resampled_data = data_copy.resample(f'{interval}H').asfreq().dropna().reset_index()
        
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

    # GRÁFICOS
    def generate_agp(self):
        """Genera y muestra el Perfil de Glucosa Ambulatoria (AGP) mejorado."""
        self.data['time_decimal'] = self.data['time'].dt.hour + self.data['time'].dt.minute / 60.0
        
        percentiles = self.data.groupby('time_decimal')['glucose'].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).unstack()
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Zonas de glucemia
        ax.axhspan(0, 70, facecolor='#ffcccb', alpha=0.3, label='Hipoglucemia')
        ax.axhspan(70, 180, facecolor='#90ee90', alpha=0.3, label='Rango objetivo')
        ax.axhspan(180, 400, facecolor='#ffcccb', alpha=0.3, label='Hiperglucemia')
        
        # Líneas de percentiles
        ax.plot(percentiles.index, percentiles[0.5], label='Mediana', color='blue', linewidth=2)
        ax.fill_between(percentiles.index, percentiles[0.25], percentiles[0.75], color='blue', alpha=0.3, label='Rango Intercuartil')
        ax.fill_between(percentiles.index, percentiles[0.05], percentiles[0.95], color='lightblue', alpha=0.2, label='Percentiles 5-95%')
        
        # Líneas horizontales en 70 y 180 mg/dL
        ax.axhline(y=70, color='red', linestyle='--', linewidth=1)
        ax.axhline(y=180, color='red', linestyle='--', linewidth=1)
        
        # Configuración de ejes y etiquetas
        ax.set_xlabel('Hora del Día', fontsize=12)
        ax.set_ylabel('Nivel de Glucosa (mg/dL)', fontsize=12)
        ax.set_title('Perfil de Glucosa Ambulatoria (AGP)', fontsize=16, fontweight='bold')
        
        # Configuración de la leyenda
        ax.legend(title="Leyenda", loc="upper left", fontsize=10)
        
        # Configuración de la cuadrícula
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Configuración de los ticks del eje x
        ax.set_xticks(range(0, 25, 3))
        ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 25, 3)])
        
        # Ajuste de los límites del eje y
        ax.set_ylim(0, 400)
        
        # Ajustes finales
        plt.tight_layout()
        plt.show()


    def day_graph(self, fecha=None):
        """
        Genera y muestra el gráfico de glucosa para un día específico.
        
        :param fecha: Fecha opcional en formato 'YYYY-MM-DD'. Si no se proporciona, se usa el primer día del DataFrame.
        """
        # Si no se proporciona fecha, usar el primer día del DataFrame
        if fecha is None:
            fecha = self.data['time'].dt.date.min()
        else:
            fecha = pd.to_datetime(fecha).date()
        
        # Filtrar datos para el día específico
        day_data = self.data[self.data['time'].dt.date == fecha].copy()
        
        if day_data.empty:
            print(f"No hay datos para la fecha {fecha}")
            return
        
        # Convertir la hora a un formato numérico para el gráfico
        day_data['hours'] = day_data['time'].dt.hour + day_data['time'].dt.minute / 60.0
        
        # Configurar el estilo de seaborn para un aspecto más profesional
        sns.set_style("whitegrid")
        sns.set_context("notebook", font_scale=1.1)

        fig, ax = plt.subplots(figsize=(16, 9))

        # Zonas de glucemia con colores más suaves y transparencia
        ax.axhspan(0, 70, facecolor='#FF9999', alpha=0.2, label='Hipoglucemia')
        ax.axhspan(70, 180, facecolor='#90EE90', alpha=0.2, label='Rango objetivo')
        ax.axhspan(180, 400, facecolor='#FFB266', alpha=0.2, label='Hiperglucemia')

        # Gráfico de línea con marcadores para los puntos de datos
        ax.plot(day_data['hours'], day_data['glucose'], label='Glucosa', 
                color='#3366CC', linewidth=2, marker='o', markersize=4)

        # Líneas horizontales en 70 y 180 mg/dL con etiquetas
        ax.axhline(y=70, color='#FF6666', linestyle='--', linewidth=1)
        ax.axhline(y=180, color='#FF6666', linestyle='--', linewidth=1)
        ax.text(24, 72, '70 mg/dL', va='bottom', ha='right', color='#FF6666')
        ax.text(24, 182, '180 mg/dL', va='bottom', ha='right', color='#FF6666')

        # Configuración de ejes y etiquetas
        ax.set_xlabel('Hora del Día', fontsize=12, fontweight='bold')
        ax.set_ylabel('Nivel de Glucosa (mg/dL)', fontsize=12, fontweight='bold')
        ax.set_title(f'Niveles de Glucosa - {fecha}', fontsize=16, fontweight='bold')

        # Configuración de la leyenda
        ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)

        # Ajuste de los límites del eje y
        ax.set_ylim(0, 400)

        # Configuración del eje x
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 3))
        ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 25, 3)])

        # Rotar las etiquetas del eje x para mejor legibilidad
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # Añadir una cuadrícula suave
        ax.grid(True, linestyle=':', alpha=0.6)

        # Ajustar los márgenes
        plt.tight_layout()

        # Mostrar el gráfico
        plt.show()