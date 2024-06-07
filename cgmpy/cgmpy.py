import datetime as datetime
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

class GlucoseData:
    def __init__(self, file_path, date_col, glucose_col, delimiter=',', header=0):
        """
        Inicializa la clase con los datos de glucemia a partir de un archivo CSV.
        
        :param file_path: Ruta al archivo CSV.
        :param date_col: Nombre de la columna que contiene las fechas.
        :param glucose_col: Nombre de la columna que contiene los valores de glucosa.
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
        # Convertir la columna de fechas a datetime
        self.data[date_col] = pd.to_datetime(self.data[date_col], errors='coerce')
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
    
    def conga(self, min=5, hours=24):
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

    def modd(self, min = 5):
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


class Dexcom(GlucoseData):
    def __init__(self, file_path):
        """
        Inicializa la clase Dexcom con los datos de glucemia a partir de un archivo CSV.
        
        :param file_path: Ruta al archivo CSV.
        :param delimiter: Delimitador usado en el archivo CSV (por defecto ',').
        """
        # Llama al constructor de la clase base con las columnas específicas de Dexcom
        super().__init__(file_path, date_col="Marca temporal (AAAA-MM-DDThh:mm:ss)", glucose_col="Nivel de glucosa (mg/dl)")

