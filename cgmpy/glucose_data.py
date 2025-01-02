import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
from typing import Union
from .utils import parse_date


class GlucoseData:
    def __init__(self, file_path: str, date_col: str, glucose_col: str, delimiter: Union[str, None] = None, header: int = 0):
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
                    df = pd.read_csv(file_path, delimiter=delim, header=header, quotechar='"', nrows=5)
                    if len(df.columns) > 1:
                        return pd.read_csv(file_path, delimiter=delim, header=header, quotechar='"')
                except pd.errors.ParserError:
                    continue
            raise ValueError("No se pudo detectar automáticamente el delimitador del archivo CSV.")
        else:
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



class Dexcom(GlucoseData):
    def __init__(self, file_path):
        """
        Inicializa la clase Dexcom con los datos de glucemia a partir de un archivo CSV.
        
        :param file_path: Ruta al archivo CSV.
        :param delimiter: Delimitador usado en el archivo CSV (por defecto ',').
        """
        # Llama al constructor de la clase base con las columnas específicas de Dexcom
        super().__init__(file_path, date_col="Marca temporal (AAAA-MM-DDThh:mm:ss)", glucose_col="Nivel de glucosa (mg/dl)") 