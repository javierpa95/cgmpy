import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
from typing import Union
from .utils import parse_date


class GlucoseData:
    def __init__(self, data_source: Union[str, pd.DataFrame], date_col: str="time", glucose_col: str="glucose", delimiter: Union[str, None] = None, header: int = 0):
        """
        Inicializa la clase con los datos de glucemia a partir de un archivo CSV o DataFrame.
        
        :param data_source: Ruta al archivo CSV o DataFrame de pandas.
        :param date_col: Nombre de la columna que contiene las fechas.
        :param glucose_col: Nombre de la columna que contiene los valores de glucosa.
        :param delimiter: Delimitador usado en el archivo CSV (solo si data_source es str).
        :param header: Índice de la fila que contiene los nombres de las columnas (solo si data_source es str).
        :raises ValueError: Si no se puede cargar el archivo o las columnas especificadas no existen.
        """
        self.date_col = date_col
        self.glucose_col = glucose_col
        
        if isinstance(data_source, pd.DataFrame):
            self.data = data_source[[date_col, glucose_col]].copy()
        else:
            self.data = self._load_csv(data_source, delimiter, header)
        
        self._validate_columns(date_col, glucose_col)
        self._process_data(date_col, glucose_col)

    def _load_csv(self, file_path: str, delimiter: Union[str, None], header: int) -> pd.DataFrame:
        """Carga el archivo CSV solo con las columnas necesarias."""
        if delimiter is None:
            for delim in [',', ';']:
                try:
                    # Primero leemos solo unas pocas filas para detectar el delimitador
                    df = pd.read_csv(file_path, delimiter=delim, header=header, quotechar='"', nrows=5)
                    if len(df.columns) > 1:
                        # Si encontramos el delimitador correcto, leemos solo las columnas necesarias
                        return pd.read_csv(
                            file_path, 
                            delimiter=delim, 
                            header=header, 
                            quotechar='"',
                            usecols=[self.date_col, self.glucose_col]
                        )
                except pd.errors.ParserError:
                    continue
            raise ValueError("No se pudo detectar automáticamente el delimitador del archivo CSV.")
        else:
            return pd.read_csv(
                file_path, 
                delimiter=delimiter, 
                header=header, 
                quotechar='"',
                usecols=[self.date_col, self.glucose_col]
            )

    

    def _validate_columns(self, date_col: str, glucose_col: str):
        """Valida que las columnas especificadas existan en el DataFrame."""
        if date_col not in self.data.columns or glucose_col not in self.data.columns:
            raise ValueError(f"Las columnas '{date_col}' o '{glucose_col}' no se encuentran en el archivo CSV. Columnas disponibles: {self.data.columns.tolist()}.")

    def _process_data(self, date_col: str, glucose_col: str):
        """Procesa los datos, convirtiendo fechas y valores de glucosa."""
        self.data = self.data.dropna(subset=[date_col])
        
        # Convertir timestamp a datetime si es necesario
        if pd.api.types.is_numeric_dtype(self.data[date_col]):
            self.data[date_col] = pd.to_datetime(self.data[date_col], unit='s')
        else:
            self.data[date_col] = self.data[date_col].apply(parse_date)
        
        self.data = self.data.dropna(subset=[date_col, glucose_col])
        # Convertir glucosa a int16 ya que los valores no superan 500
        self.data[glucose_col] = pd.to_numeric(self.data[glucose_col], errors='coerce').astype('int16')
        self.data = self.data.dropna(subset=[glucose_col])
        self.data.rename(columns={date_col: 'time', glucose_col: 'glucose'}, inplace=True)
        self.data = self.data[['time', 'glucose']]
        
        # Después de todas las transformaciones, ordenar por tiempo
        self.data = self.data.sort_values('time', ascending=True).reset_index(drop=True)
        
        # Optimizar el DataFrame completo
        self.data = self.data.copy()
        self.data['time'] = pd.to_datetime(self.data['time'])

    ## INFORMACIÓN BÁSICA

    def __str__(self) -> str:
        """Devuelve una representación en string del objeto con la información básica."""
        info = self.info()
        resumen = (f"El archivo CSV contiene {info['num_datos']} datos entre {info['fecha_inicio']} y {info['fecha_fin']}.\n"
                   f"Intervalo típico entre mediciones: {info['intervalo_tipico']:.1f} minutos.\n"
                   f"Datos teóricos esperados: {info['datos_teoricos']}\n"
                   f"Porcentaje de datos disponibles: {info['porcentaje_disponibilidad']:.1f}%\n"
                   f"Se detectaron {info['num_desconexiones']} desconexiones.\n"
                   f"Tiempo total de desconexión: {info['tiempo_total_desconexion']:.1f} horas.\n"
                   f"Uso de memoria del DataFrame: {info['uso_memoria_mb']:.2f} MB")
        
        return resumen

    def info(self, include_disconnections: bool = False) -> dict:
        """
        Muestra información básica del archivo CSV en formato JSON.
        
        :param include_disconnections: Si es True, incluye el detalle de todas las desconexiones.
                                       Si es False, solo muestra el resumen básico.
        :return: Diccionario con la información solicitada.
        """
        # Información básica
        num_datos = len(self.data)
        
        # Primero obtener las fechas
        fecha_min = self.data['time'].min()
        fecha_max = self.data['time'].max()
        
        # Luego convertir a string
        fecha_inicio = fecha_min.strftime('%Y-%m-%d %H:%M:%S')
        fecha_fin = fecha_max.strftime('%Y-%m-%d %H:%M:%S')
        
        # Ordenar los datos por fecha antes de calcular las diferencias
        datos_ordenados = self.data.sort_values('time')
        diferencias = datos_ordenados['time'].diff()
        intervalo_tipico = diferencias.median().total_seconds() / 60
        
        # Asegurarse de que el intervalo sea positivo
        intervalo_tipico = abs(intervalo_tipico)
        
        # Umbral de desconexión: intervalo típico + 10 minutos
        umbral_desconexion = pd.Timedelta(minutes=intervalo_tipico + 10)
        desconexiones = diferencias[diferencias > umbral_desconexion]
        num_desconexiones = len(desconexiones)
        
        # Calcular tiempo total de desconexión
        tiempo_total_desconexion = desconexiones.sum()
        horas_desconexion = tiempo_total_desconexion.total_seconds() / 3600
        
        # Crear lista detallada de desconexiones
        lista_desconexiones = []
        if num_desconexiones > 0:
            indices_desconexiones = diferencias[diferencias > pd.Timedelta(minutes=15)].index
            for idx, indice in enumerate(indices_desconexiones, 1):
                posicion_actual = self.data.index.get_loc(indice)
                if posicion_actual > 0:
                    fecha_fin = self.data.iloc[posicion_actual]['time']
                    fecha_inicio = self.data.iloc[posicion_actual - 1]['time']
                    duracion_minutos = (fecha_fin - fecha_inicio).total_seconds() / 60
                    horas = int(duracion_minutos // 60)
                    minutos = int(duracion_minutos % 60)
                    lista_desconexiones.append({
                        "inicio": fecha_inicio.strftime('%d/%m/%Y %H:%M'),
                        "fin": fecha_fin.strftime('%d/%m/%Y %H:%M'),
                        "duracion": f"{horas:02d} horas y {minutos:02d} minutos"
                    })
        
        # Calcular el uso de memoria
        memoria_bytes = self.data.memory_usage(deep=True).sum()
        memoria_mb = memoria_bytes / (1024 * 1024)
        
        # Calcular el número teórico de datos
        tiempo_total = (self.data['time'].max() - self.data['time'].min()).total_seconds() / 60
        datos_teoricos = int(tiempo_total / intervalo_tipico)
        porcentaje_disponibilidad = (num_datos / datos_teoricos) * 100
        
        # Crear el diccionario de resumen
        resumen = {
            "num_datos": num_datos,
            "fecha_inicio": fecha_inicio,
            "fecha_fin": fecha_fin,
            "intervalo_tipico": intervalo_tipico,
            "datos_teoricos": datos_teoricos,
            "porcentaje_disponibilidad": porcentaje_disponibilidad,
            "num_desconexiones": num_desconexiones,
            "tiempo_total_desconexion": horas_desconexion,
            "uso_memoria_mb": memoria_mb,
        }

        if include_disconnections:
            resumen["lista_desconexiones"] = lista_desconexiones

        return resumen

    def debug_fechas(self):
        """
        Función temporal para debuggear el comportamiento de las fechas
        """
        print("\n=== Debug de Fechas ===")
        print(f"Total de registros: {len(self.data)}")
        print(f"\nTipo de dato de 'time': {self.data['time'].dtype}")
        
        print("\nComparación de métodos:")
        print(f"Usando min(): {self.data['time'].min()}")
        print(f"Usando iloc[0]: {self.data.iloc[0]['time']}")
        print(f"Usando max(): {self.data['time'].max()}")
        print(f"Usando iloc[-1]: {self.data.iloc[-1]['time']}")
        
        print("\nPrimeras 3 fechas:")
        print(self.data['time'].head(3))
        print("\nÚltimas 3 fechas:")
        print(self.data['time'].tail(3))
        
        # Verificar valores nulos o NaT
        nulos = self.data['time'].isna().sum()
        print(f"\nValores nulos: {nulos}")


class Dexcom(GlucoseData):
    def __init__(self, file_path):
        """
        Inicializa la clase Dexcom con los datos de glucemia a partir de un archivo CSV.
        
        :param file_path: Ruta al archivo CSV.
        :param delimiter: Delimitador usado en el archivo CSV (por defecto ',').
        """
        # Llama al constructor de la clase base con las columnas específicas de Dexcom
        super().__init__(file_path, date_col="Marca temporal (AAAA-MM-DDThh:mm:ss)", glucose_col="Nivel de glucosa (mg/dl)") 