import datetime as datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class GlucoseData:
    def __init__(self, file_path, date_col, glucose_col, date_format='%d/%m/%Y %H:%M', delimiter=',', header=0):
        """
        Inicializa la clase con los datos de glucemia a partir de un archivo CSV.
        
        :param file_path: Ruta al archivo CSV.
        :param date_col: Nombre de la columna que contiene las fechas.
        :param glucose_col: Nombre de la columna que contiene los valores de glucosa.
        :param date_format: Formato de fecha esperado en la columna de fechas (por defecto '%d/%m/%Y %H:%M').
        :param delimiter: Delimitador usado en el archivo CSV (por defecto ',').
        :param header: Índice de la fila que contiene los nombres de las columnas (por defecto 0).
        """

        if delimiter is None:
            try:
                # Intentar cargar el archivo CSV con ',' como delimitador
                self.data = pd.read_csv(file_path, delimiter=',', header=header)
            except pd.errors.ParserError:
                try:
                    # Si falla, intentar con ';' como delimitador
                    self.data = pd.read_csv(file_path, delimiter=';', header=header)
                except pd.errors.ParserError as e:
                    raise ValueError("No se pudo cargar el archivo CSV con los delimitadores ',' o ';'.") from e
        else:
            # Cargar el archivo CSV con el delimitador especificado
            self.data = pd.read_csv(file_path, delimiter=delimiter, header=header)

        # Verificar que las columnas especificadas existen
        if date_col not in self.data.columns or glucose_col not in self.data.columns:
            # Cargar el archivo para obtener las columnas
            temp_data = pd.read_csv(file_path, header=header)
            raise ValueError(f"Las columnas especificadas '{date_col}' o '{glucose_col}' no se encuentran en el archivo CSV. \n Columnas disponibles en el archivo: {temp_data.columns.tolist()}.")

        # Eliminar filas donde date_col sea nulo o vacío
        self.data = self.data.dropna(subset=[date_col])
        # Convertir la columna de fechas a datetime con formato especificado
        self.data[date_col] = pd.to_datetime(self.data[date_col], format=date_format, errors='coerce')
        # Filtrar las filas que contienen datos de glucosa válidos
        self.data = self.data.dropna(subset=[date_col, glucose_col])
        # Asegurarnos de que la columna de glucosa es numérica
        self.data[glucose_col] = pd.to_numeric(self.data[glucose_col], errors='coerce')
        self.data = self.data.dropna(subset=[glucose_col])
        # Renombrar las columnas de fecha y glucosa en el DataFrame
        self.data.rename(columns={date_col: 'time', glucose_col: 'glucose'}, inplace=True)
        # Actualizar los atributos de la clase para reflejar los nuevos nombres de las columnas
        self.date_col = 'time'
        self.glucose_col = 'glucose'
        # Seleccionar solo las columnas 'time' y 'glucose'
        self.data = self.data[['time', 'glucose']]

        print("Hay en total", len(self.data), "datos.")
        print("las fechas van entre",self.data['time'].min(),"y ", self.data['time'].max())


    def mean(self):
        """
        Calcula la glucemia media.
        
        :return: La glucemia media.

        DOI: 
        """
        
        return self.data['glucose'].mean()
    
    def median(self):
        """
        Calcula la mediana de la glucemia.
        
        :return: La mediana de la glucemia.
        """
        return self.data['glucose'].median()

    def sd(self):
        """
        Calcula la desviación estándar de la glucemia.
        
        :return: La desviación estándar de la glucemia.
        """
        return self.data['glucose'].std()
    
    def gmi(self):
        """
        Calcula el Glucose management index 
        Returns: GMI (float): glucose management index (an estimate of HbA1c)
        DOI: 10.2337/dc18-1581
        """
        GMI = 3.31 + (0.02392*np.mean(self.data['glucose']))
        return GMI.round(2)
    
    
    def calculate_time_in_range(self, low_threshold, high_threshold):
        """
        Calcula el tiempo en rango (TIR) de glucemia.

        :param data: DataFrame con lecturas de glucemia.
        :param glucose_col: Nombre de la columna que contiene los valores de glucosa.
        :param low_threshold: Umbral inferior del rango.
        :param high_threshold: Umbral superior del rango.
        :return: Porcentaje de tiempo en rango.
        """
        in_range = self.data[(self.data['glucose'] >= low_threshold) & (self.data['glucose'] <= high_threshold)]

        return (len(in_range) / len(self.data['glucose'])) * 100
    
    def TAR(self, threshold):
        """
        Calcula el riesgo de hiperglucemia.
        :param threshold: Umbral de hiperglucemia.
        :return: Porcentaje de lecturas por encima del umbral.
        """
        hyperglycemia = self.data[self.data['glucose'] > threshold]
        return (len(hyperglycemia) / len(self.data)) * 100
    
    def TBR(self, threshold):
        """
        Calcula el % de hipoglucemia.
        :param threshold: Umbral de hiperglucemia.
        :return: Porcentaje de lecturas por encima del umbral.
        """
        hypoglycemia = self.data[self.data['glucose'] < threshold]
        return (len(hypoglycemia) / len(self.data)) * 100

    def TAR250(self):
        # Tiempo en rango por encima de 250 mg/dL
        return self.TAR(250)
    
    def TAR180(self):
        # Tiempo en rango por encima de 250 mg/dL
        return self.calculate_time_in_range(180,250)

    def TIR(self):
        return self.calculate_time_in_range(70,180)
    
    def TBR70(self):
        return self.calculate_time_in_range(55,70)
    
    def TBR55(self):
        return self.TBR(55)
    
    # MEDIDAS DE VARIABILIDAD 
    
    def CONGA(self, min=5, hours=24):
        """
        Calcula y devuelve el valor de CONGA para datos medidos cada 5 minutos.
        :param min: Distancia en minutos entre las mediciones. Por defecto 5 minutos
        :param hours: Número de horas para calcular la diferencia. Por defecto es 24.
        """
        # Calcular el número de intervalos de 5 minutos en 'hours' horas
        segmentos = int(60/min)
        intervals = hours * segmentos
        
        # Crear una columna de glucosa desplazada 'intervals' intervalos hacia atrás
        self.data['glucose_n_hours_ago'] = self.data['glucose'].shift(intervals)

        # Eliminar filas donde la columna desplazada tenga NaN
        valid_data = self.data.dropna(subset=['glucose_n_hours_ago']).copy()
        
        # Calcular Dt como la diferencia entre GRt y GRt-m
        valid_data['Dt'] = valid_data['glucose'] - valid_data['glucose_n_hours_ago']
        
        # Calcular el promedio de Dt
        D_mean = valid_data['Dt'].mean()
        
        # Calcular la suma de las diferencias cuadradas
        sum_squared_diff = ((valid_data['Dt'] - D_mean) ** 2).sum()
        
        # Calcular k*, el número de observaciones válidas
        k_star = len(valid_data)

        # Calcular CONGAn usando la fórmula
        conga_n = np.sqrt(sum_squared_diff / (k_star - 1))
        
        print(f"Para el calculo de CONGA{hours}, hay {intervals} valores que no tienen previo y se han eliminado {len(self.data)-k_star}")
        return conga_n

    def MODD(self, min = 5):
        """
        Calcula y devuelve el valor de MODD para datos medidos cada 5 minutos (por defecto).
        DOI: 10.1007/BF01218495
        """
        # Calcular el número de intervalos de 5 minutos en 24 horas (288 intervalos)

        intervals = 24 * int(60/min) # Estos son los intervalos que hay entre un valor y el siguiente 24 horas después
        
        # Crear una columna de glucosa desplazada 288 intervalos hacia atrás
        self.data['glucose_24h_ago'] = self.data['glucose'].shift(intervals)
        
        # Eliminar filas donde la columna desplazada tenga NaN
        valid_data = self.data.dropna(subset=['glucose_24h_ago']).copy()
        
        # Calcular la diferencia absoluta entre GRt y GRt-1440 (24 horas antes)
        valid_data['abs_diff'] = (valid_data['glucose'] - valid_data['glucose_24h_ago']).abs()
        
        # Calcular k*, el número de observaciones válidas
        k_star = len(valid_data)
       
        # Calcular MODD como el promedio de las diferencias absolutas
        modd = valid_data['abs_diff'].sum() / k_star

        print(f"Para el calculo de modd, hay {intervals} valores que no tienen previo y se han eliminado {len(self.data)-k_star}")
        
        return modd

    def j_index(self):

        j_index= 0.001*(self.mean()+self.sd())**2

        return j_index

    def LBGI(self):
        """
        Calcula la función f(BG) para BG en mg/dL y r(BG)
        Calcula y devuelve el valor de LBGI.
        DOI: 10.2337/db12-1396
        """
        # Calcular f(BG)
        self.data["f_bg"] = 1.509 * ((np.log(self.data["glucose"]))**1.084 - 5.381)
        
        # Calcular r(BG)
        self.data["r_bg"] = 10 * (self.data["f_bg"])**2
        
        # Calcular rl(BG)
        self.data["rl_bg"] = self.data.apply(lambda row: row["r_bg"] if row["f_bg"] < 0 else 0, axis=1)
        
        # Calcular y devolver LBGI
        return self.data["rl_bg"].mean()

    def HBGI(self):
        """
        Calcula la función f(BG) para BG en mg/dL y r(BG)
        Calcula y devuelve el valor de HBGI.
        DOI: 10.2337/db12-1396
        """
        # Calcular f(BG)
        self.data["f_bg"] = 1.509 * ((np.log(self.data["glucose"]))**1.084 - 5.381)
        
        # Calcular r(BG)
        self.data["r_bg"] = 10 * (self.data["f_bg"])**2
        
        # Calcular rh(BG)
        self.data["rh_bg"] = self.data.apply(lambda row: row["r_bg"] if row["f_bg"] > 0 else 0, axis=1)
        
        # Calcular y devolver HBGI
        return self.data["rh_bg"].mean()

    def MAGE(self):
        """
        Calcula el MAGE (Mean Amplitude of Glycemic Excursions).
        
        :return: El valor de MAGE.
        """
        # Paso 1: Calcular la desviación estándar (SD) de los valores de glucosa
        sd = self.sd()
        
        # Paso 2: Determinar todos los valores máximos (picos) y mínimos (valles) locales
        # Utilizando el método shift para comparar cada valor con el anterior y el siguiente
        peaks_and_nadirs = self.data[
            (self.data['glucose'].shift(1) < self.data['glucose']) & (self.data['glucose'] > self.data['glucose'].shift(-1)) | 
            (self.data['glucose'].shift(1) > self.data['glucose']) & (self.data['glucose'] < self.data['glucose'].shift(-1))
        ]
        
        # Restablecer el índice para trabajar con índices secuenciales
        peaks_and_nadirs.reset_index(drop=True, inplace=True)
        
        # Lista para almacenar las excursiones glucémicas válidas
        excursions = []
        
       # Paso 3: Verificar si el primer valor es un pico o un valle
        starts_with_peak = peaks_and_nadirs['glucose'][0] > peaks_and_nadirs['glucose'][1]

        # Evaluar los pares de pico y valle para determinar si cumplen con el criterio de 1 SD
        if starts_with_peak:
            for i in range(0, len(peaks_and_nadirs) - 1, 2):
                peak = peaks_and_nadirs['glucose'][i]
                nadir = peaks_and_nadirs['glucose'][i + 1]
                # Verificar si la diferencia entre pico y valle excede 1 SD
                if abs(peak - nadir) > sd:
                    excursions.append(abs(peak - nadir))
        else:
            for i in range(0, len(peaks_and_nadirs) - 1, 2):
                nadir = peaks_and_nadirs['glucose'][i]
                peak = peaks_and_nadirs['glucose'][i + 1]
                # Verificar si la diferencia entre valle y pico excede 1 SD
                if abs(peak - nadir) > sd:
                    excursions.append(abs(peak - nadir))

        
        # Si no se encontraron excursiones válidas, retornar 0
        if len(excursions) == 0:
            return 0
        print(excursions)
        # Paso 4: Calcular el MAGE promediando las excursiones glucémicas válidas
        print("En mage 2 hay:", len(excursions),"y en total",len(peaks_and_nadirs))
        return sum(excursions) / len(excursions)
    
    def M_Value(self, target_glucose=80):
        """
        Calcula el M-Value para evaluar la variabilidad de la glucosa en sangre.

        :param target_glucose: Valor objetivo de glucosa (por defecto 80 mg/dL).
        :return: El M-Value.

        DOI: 10.2337/db12-1396 - Explica que si tienes mas de 25 datos puedes obviar la corrección. Además el target_glucose en el artículo original lo ponínan en 120, ellos ponen 90!!!
        """
        def calculate_M(PG):
            return abs(10 * np.log10(PG / target_glucose)) ** 3

        self.data['M'] = self.data['glucose'].apply(calculate_M)
        M_sum = self.data['M'].sum()
        N = len(self.data)

        
        M_value = (M_sum / N) 
        
        return M_value

    def Lability_index(self, interval=1, period='week'):
        """
        Calcula el índice de labilidad (LI) basado en las lecturas de glucosa y sus tiempos correspondientes.

        :param interval: El intervalo de tiempo en horas para el cálculo.
        :param period: El periodo para la media del LI ('day', 'week', 'month').
        :return: El índice de labilidad (LI) medio para el periodo especificado.
        """
        # Trabajar con una copia del DataFrame original
        data_copy = self.data.copy()
        data_copy.set_index('time', inplace=True)
        
        # Imprimir el rango de fechas antes del resampleo
        print("Rango de fechas antes del resampleo:", data_copy.index.min(), "a", data_copy.index.max())
        
        # Resamplear los datos a intervalos especificados
        resampled_data = data_copy.resample(f'{interval}H').asfreq().dropna().reset_index()
        
        # Imprimir el rango de fechas después del resampleo
        print("Rango de fechas después del resampleo:", resampled_data['time'].min(), "a", resampled_data['time'].max())
        
        print("Número de filas en los datos resampleados:", len(resampled_data))
        print(resampled_data.head())
        print(resampled_data.tail())
        
        # Asignar periodos según el tipo especificado (día, semana, mes)
        if period == 'day':
            resampled_data['period'] = resampled_data['time'].dt.date
        elif period == 'week':
            # Calcular manualmente la semana basada en el primer dato
            resampled_data['period'] = (resampled_data['time'] - resampled_data['time'].min()).dt.days // 7
        elif period == 'month':
            resampled_data['period'] = resampled_data['time'].dt.to_period('M').apply(lambda r: r.start_time)
        else:
            raise ValueError("Periodo no válido. Usa 'day', 'week' o 'month'.")
        
        print(resampled_data.head())
        
        li_values = []
        
        # Calcular el índice de labilidad para cada grupo de periodos
        for period, group in resampled_data.groupby('period'):
            print(f"Período: {period}, Número de datos: {len(group)}")
            glucose_readings = group['glucose'].values
            times = group['time'].values.astype('datetime64[h]').astype(int)  # Convertir tiempos a horas y luego a enteros
            
            if len(glucose_readings) < 2:
                continue
            
            n = len(glucose_readings)
            li_sum = 0
            
            # Calcular el LI para el grupo actual
            for i in range(n - 1):
                gluc_diff = glucose_readings[i] - glucose_readings[i + 1]
                time_diff = times[i + 1] - times[i]
                
                if 1 <= time_diff <= interval:
                    li_sum += (gluc_diff ** 2) / time_diff
            
            li_value = li_sum / (n - 1)
            print(f"LI para el grupo {period}: {li_value}")
            li_values.append(li_value)
        
        if not li_values:
            return 0
        
        return np.mean(li_values)



    # GRAFICOS
    def generate_agp(self):
       # Asegurarse de que la columna de fechas es de tipo datetime
        self.data["time"] = pd.to_datetime(self.data["time"])
        
        # Extraer la hora del día de la columna de fechas
        self.data['time_of_day'] = self.data["time"].dt.time
        
        # Convertir la hora del día a una representación decimal
        self.data['time_decimal'] = self.data['time_of_day'].apply(lambda x: x.hour + x.minute / 60.0)
        
        # Agrupar los datos por la hora decimal y calcular los percentiles
        percentiles = self.data.groupby('time_decimal')["glucose"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).unstack()
        
       # Graficar los percentiles
        plt.figure(figsize=(12, 6))

        plt.plot(percentiles.index, percentiles[0.5], label='Mediana (50%)', color='blue')
        plt.fill_between(percentiles.index, percentiles[0.25], percentiles[0.75], color='blue', alpha=0.3, label='Rango Intercuartil (25%-75%)')
        plt.fill_between(percentiles.index, percentiles[0.05], percentiles[0.95], color='lightblue', alpha=0.2, label='Percentiles Extremos (5%-95%)')

        plt.xlabel('Hora del Día')
        plt.ylabel('Nivel de Glucosa (mg/dL)')
        plt.title('Perfil de Glucosa Ambulatoria (AGP)')
        plt.legend(title="Leyenda", loc="upper left")
        plt.grid(True)
        plt.xticks(ticks=range(0, 25), labels=[f'{h:02d}:00' for h in range(25)])
        plt.show()
