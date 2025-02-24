import datetime
import pandas as pd
import csv  # Importamos csv para detectar el delimitador automáticamente
from typing import Union
from .utils import parse_date


class GlucoseData:
    def __init__(self, data_source: Union[str, pd.DataFrame], date_col: str="time", glucose_col: str="glucose", delimiter: Union[str, None] = None, header: int = 0, start_date: Union[str, datetime.datetime, None] = None, end_date: Union[str, datetime.datetime, None] = None):
        """
        Inicializa la clase con los datos de glucemia a partir de un archivo CSV o DataFrame.
        
        :param data_source: Ruta al archivo CSV o DataFrame de pandas.
        :param date_col: Nombre de la columna que contiene las fechas.
        :param glucose_col: Nombre de la columna que contiene los valores de glucosa.
        :param delimiter: Delimitador usado en el archivo CSV (solo si data_source es str).
        :param header: Índice de la fila que contiene los nombres de las columnas (solo si data_source es str).
        :param start_date: Fecha inicial para filtrar los datos (opcional) YYYY_MM_DD.
        :param end_date: Fecha final para filtrar los datos (opcional) YYYY_MM_DD.
        :raises ValueError: Si no se puede cargar el archivo o las columnas especificadas no existen.
        """
        self.date_col = date_col
        self.glucose_col = glucose_col
        self.typical_interval = None  # Inicializamos el atributo
        
        if isinstance(data_source, pd.DataFrame):
            self.data = data_source[[date_col, glucose_col]].copy()
        else:
            self.data = self._load_csv(data_source, delimiter, header)
        
        self._validate_columns(date_col, glucose_col)
        self._process_data(date_col, glucose_col)
        
        # Filtrar por fechas si se especifican
        if start_date is not None or end_date is not None:
            self._filter_by_dates(start_date, end_date)
        
        # Calculamos el intervalo típico después de todo el procesamiento
        self.typical_interval = self._calculate_typical_interval()

    def _load_csv(self, file_path: str, delimiter: Union[str, None], header: int) -> pd.DataFrame:
        """Carga el archivo CSV solo con las columnas necesarias."""
        try:
            if delimiter is None:
                # Usar csv.Sniffer para detectar el delimitador automáticamente
                with open(file_path, 'r', encoding='utf-8') as f:
                    sample = f.read(1024)
                    sniffer = csv.Sniffer()
                    dialect = sniffer.sniff(sample)
                    delimiter = dialect.delimiter
            return pd.read_csv(
                file_path, 
                delimiter=delimiter, 
                header=header, 
                quotechar='"',
                usecols=[self.date_col, self.glucose_col]
            )
        except Exception as e:
            # Fallback a detección manual si csv.Sniffer falla
            if delimiter is None:
                for delim in [',', ';']:
                    try:
                        df = pd.read_csv(file_path, delimiter=delim, header=header, quotechar='"', nrows=5)
                        if len(df.columns) > 1:
                            return pd.read_csv(
                                file_path, 
                                delimiter=delim, 
                                header=header, 
                                quotechar='"',
                                usecols=[self.date_col, self.glucose_col]
                            )
                    except pd.errors.ParserError:
                        continue
            raise ValueError(f"Error al parsear CSV: {str(e)}") from e

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
        self.data[glucose_col] = pd.to_numeric(self.data[glucose_col], errors='coerce', downcast='integer')
        self.data = self.data.dropna(subset=[glucose_col])
        self.data.rename(columns={date_col: 'time', glucose_col: 'glucose'}, inplace=True)
        self.data = self.data[['time', 'glucose']]
        
        # Después de todas las transformaciones, ordenar por tiempo
        self.data = self.data.sort_values('time', ascending=True).reset_index(drop=True)
        
        
        # Optimizar el DataFrame completo
        self.data = self.data.copy()
        # Eliminar redundancia en conversión de tiempo
        if not pd.api.types.is_datetime64_any_dtype(self.data['time']):
            self.data['time'] = pd.to_datetime(self.data['time'])

    def _calculate_typical_interval(self) -> float:
        """
        Calcula el intervalo típico entre mediciones en minutos.
        
        :return: Intervalo típico en minutos
        """
        diferencias = self.data['time'].diff()
        intervalo = diferencias.median().total_seconds() / 60
        return abs(intervalo)

    def get_typical_interval(self) -> float:
        """
        Devuelve el intervalo típico entre mediciones en minutos.
        
        :return: Intervalo típico en minutos
        """
        return self.typical_interval

    def _filter_by_dates(self, start_date: Union[str, datetime.datetime, None], 
                        end_date: Union[str, datetime.datetime, None]):
        """Filtra los datos por rango de fechas."""
        if start_date is not None:
            if isinstance(start_date, str):
                start_date = parse_date(start_date)
            self.data = self.data[self.data['time'] >= start_date]
        
        if end_date is not None:
            if isinstance(end_date, str):
                end_date = parse_date(end_date)
            self.data = self.data[self.data['time'] <= end_date]
        
        if len(self.data) == 0:
            raise ValueError("No hay datos disponibles en el rango de fechas especificado.")

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
        
        # Obtener las fechas directamente
        fecha_inicio = self.data['time'].min()
        fecha_fin = self.data['time'].max()
        
        # Reutilizamos el intervalo típico ya calculado
        intervalo_tipico = self.typical_interval

        # Umbral de desconexión: intervalo típico + 10 minutos
        umbral_desconexion = pd.Timedelta(minutes=intervalo_tipico + 10)
        diferencias = self.data['time'].diff()
        desconexiones = diferencias[diferencias > umbral_desconexion]
        num_desconexiones = len(desconexiones)
        
        # Calcular tiempo total de desconexión
        tiempo_total_desconexion = desconexiones.sum()
        horas_desconexion = tiempo_total_desconexion.total_seconds() / 3600
        
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
            "num_desconexiones": f"{num_desconexiones} desconexiones (Para más información, use el método info(include_disconnections=True))",
            "tiempo_total_desconexion": horas_desconexion,
            "uso_memoria_mb": memoria_mb,
        }

        if include_disconnections:
            # Crear lista detallada de desconexiones
            lista_desconexiones = []
            if num_desconexiones > 0:
                for idx, indice in enumerate(desconexiones.index, 1):
                    posicion_actual = self.data.index.get_loc(indice)
                    if posicion_actual > 0:
                        fecha_fin_desconexion = self.data.iloc[posicion_actual]['time']
                        fecha_inicio_desconexion = self.data.iloc[posicion_actual - 1]['time']
                        duracion_minutos = (fecha_fin_desconexion - fecha_inicio_desconexion).total_seconds() / 60
                        horas = int(duracion_minutos // 60)
                        minutos = int(duracion_minutos % 60)
                        lista_desconexiones.append({
                            "inicio": fecha_inicio_desconexion.strftime('%d/%m/%Y %H:%M'),
                            "fin": fecha_fin_desconexion.strftime('%d/%m/%Y %H:%M'),
                            "duracion": f"{horas:02d} horas y {minutos:02d} minutos"
                        })

            resumen["lista_desconexiones"] = lista_desconexiones

        return resumen

class Dexcom(GlucoseData):
    def __init__(self, file_path: str, 
                 start_date: Union[str, datetime.datetime, None] = None,
                 end_date: Union[str, datetime.datetime, None] = None):
        """
        Clase especializada para datos de dispositivos Dexcom.
        
        Ejemplo de uso:
        >>> dexcom = Dexcom("datos.csv")
        >>> print(dexcom.info())
        
        :param file_path: Ruta al archivo CSV exportado de Clarity
        :param start_date: Filtro opcional de fecha inicial (YYYY-MM-DD)
        :param end_date: Filtro opcional de fecha final (YYYY-MM-DD)
        """
        super().__init__(
            file_path, 
            date_col="Marca temporal (AAAA-MM-DDThh:mm:ss)", 
            glucose_col="Nivel de glucosa (mg/dl)",
            start_date=start_date,
            end_date=end_date
        ) 