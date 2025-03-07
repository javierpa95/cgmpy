import datetime
import pandas as pd
import csv  # Importamos csv para detectar el delimitador automáticamente
from typing import Union
from .utils import parse_date
import os
import numpy as np
import time


class GlucoseData:
    def __init__(self, data_source: Union[str, pd.DataFrame], date_col: str="time", glucose_col: str="glucose", delimiter: Union[str, None] = None, header: int = 0, start_date: Union[str, datetime.datetime, None] = None, end_date: Union[str, datetime.datetime, None] = None, log: bool = False):
        """
        Inicializa los datos de glucosa con optimizaciones para archivos grandes.
        
        :param data_source: Archivo CSV o DataFrame con los datos de glucosa
        :param date_col: Nombre de la columna de fecha/hora
        :param glucose_col: Nombre de la columna de valores de glucosa
        :param delimiter: Delimitador para archivos CSV
        :param header: Fila de encabezado para archivos CSV
        :param start_date: Fecha de inicio para filtrar datos (opcional)
        :param end_date: Fecha de fin para filtrar datos (opcional)
        :param log: Si True, guarda información detallada de las operaciones realizadas
        """
        self.log = log  # Atributo para controlar logging en toda la clase
        self.logs = {}  # Diccionario para almacenar los logs de diferentes operaciones
        
        self.date_col = date_col
        self.glucose_col = glucose_col
        self.typical_interval = None  # Inicializamos el atributo
        
        t_start = time.time()

        t0 = time.time()
        # Carga del archivo
        if isinstance(data_source, str):
            if os.path.isfile(data_source):
                # Verificar si el archivo es grande
                file_size = os.path.getsize(data_source)
                is_large_file = file_size > 10*1024*1024  # > 10MB
                
                if is_large_file:
                    print(f"Cargando archivo grande en GlucoseData ({file_size/1024/1024:.1f} MB)...")
                
                # Detectar automáticamente el delimitador si no se especifica
                if delimiter is None:
                    with open(data_source, 'r') as f:
                        first_line = f.readline().strip()
                        for delim in [',', ';', '\t']:
                            if delim in first_line:
                                delimiter = delim
                                break
                        else:
                            delimiter = ','  # Default a coma
                
                # Leer una muestra para detectar tipo de fecha
                sample = pd.read_csv(data_source, delimiter=delimiter, header=header, nrows=5)
                date_is_numeric = pd.api.types.is_numeric_dtype(sample[date_col])
                
                # Crear un diccionario de tipos optimizado
                dtypes = {}
                for col in sample.columns:
                    if col == date_col and date_is_numeric:
                        continue  # Lo manejaremos después
                    elif col != date_col:
                        if pd.api.types.is_integer_dtype(sample[col]):
                            # Cambiamos de int32 a float32 para manejar NAs
                            dtypes[col] = 'float32'
                        elif pd.api.types.is_float_dtype(sample[col]):
                            dtypes[col] = 'float32'
                
                # Leer solo las columnas necesarias
                usecols = [date_col, glucose_col]
                
                # Leer el archivo (sin convertir fechas aún)
                self.data = pd.read_csv(
                    data_source, 
                    delimiter=delimiter, 
                    header=header, 
                    usecols=usecols,
                    dtype=dtypes if is_large_file else None,
                    engine='c' if is_large_file else 'python',
                    na_values=['', 'NA', 'NULL', 'null', 'NaN']  # Especificamos valores nulos explícitamente
                )
                
                # Convertir la columna de fecha después de leer
                if date_is_numeric:
                    # Detectar si son milisegundos o segundos
                    sample_value = self.data[date_col].iloc[0] if len(self.data) > 0 else 0
                    date_unit = 'ms' if sample_value > 10000000000 else 's'
                    
                    # Convertir usando to_datetime con el parámetro unit
                    self.data[date_col] = pd.to_datetime(self.data[date_col], unit=date_unit)
                else:
                    # Para fechas en formato texto
                    try:
                        self.data[date_col] = pd.to_datetime(self.data[date_col], errors='coerce')
                    except:
                        # En caso de error, intentar con el convertidor personalizado
                        self.data[date_col] = self.data[date_col].apply(parse_date)
            else:
                raise FileNotFoundError(f"Archivo no encontrado: {data_source}")
        elif isinstance(data_source, pd.DataFrame):
            # Es un DataFrame, usarlo directamente
            self.data = data_source.copy()
        else:
            raise ValueError("data_source debe ser un archivo CSV o un DataFrame")
        t1 = time.time()

        t2 = time.time()
        # Conversión de fechas
        self.data[date_col] = pd.to_datetime(self.data[date_col], errors='coerce')
        t3 = time.time()

        t4 = time.time()
        # Renombrar columnas
        if date_col != "time":
            self.data = self.data.rename(columns={date_col: "time"})
        if glucose_col != "glucose":
            self.data = self.data.rename(columns={glucose_col: "glucose"})
        t5 = time.time()

        t6 = time.time()
        # Filtrado por fechas
        if start_date is not None:
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            self.data = self.data[self.data["time"] >= start_date]
        
        if end_date is not None:
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            self.data = self.data[self.data["time"] <= end_date]
        t7 = time.time()

        


        
        # Completa el procesamiento de datos
        t8 = time.time()
        self._validate_columns("time", "glucose")
        t9 = time.time()
        self._process_data("time", "glucose")
        t10 = time.time()

        t_end = time.time()

        print(f"""
        Tiempos de carga de datos:
        Lectura del archivo CSV: {t1-t0:.3f}s
        Conversión de fechas: {t3-t2:.3f}s
        Renombrado de columnas: {t5-t4:.3f}s
        Filtrado por fechas: {t7-t6:.3f}s
        Validación de columnas: {t9-t8:.3f}s
        Procesamiento de datos: {t10-t9:.3f}s
        Tiempo total: {t_end-t_start:.3f}s
        """)
        
        # Calculamos el intervalo típico después de todo el procesamiento
        self.typical_interval = self._calculate_typical_interval()

    def _validate_columns(self, date_col: str, glucose_col: str):
        """Valida que las columnas especificadas existan en el DataFrame."""
        if date_col not in self.data.columns or glucose_col not in self.data.columns:
            raise ValueError(f"Las columnas '{date_col}' o '{glucose_col}' no se encuentran en el archivo CSV. Columnas disponibles: {self.data.columns.tolist()}.")

    def _process_data(self, date_col: str, glucose_col: str):
        """Procesa los datos optimizando operaciones costosas."""
        # Operaciones iniciales rápidas
        self.data = self.data.dropna(subset=[date_col, glucose_col]).copy()
        
        # Conversión vectorizada de fechas
        if not pd.api.types.is_datetime64_any_dtype(self.data[date_col]):
            if pd.api.types.is_numeric_dtype(self.data[date_col]):
                unit = 'ms' if self.data[date_col].iloc[0] > 1e10 else 's'
                self.data[date_col] = pd.to_datetime(self.data[date_col], unit=unit)
            else:
                self.data[date_col] = pd.to_datetime(
                    self.data[date_col],
                    errors='coerce',
                    format='mixed'  # Nuevo en pandas 2.0, más eficiente
                )
        
        # Filtrar nulos después de conversión
        self.data = self.data.dropna(subset=[date_col, glucose_col])
        
        # Optimización de tipos
        self.data[glucose_col] = pd.to_numeric(self.data[glucose_col], errors='coerce', downcast='float')
        
        # Manejo de duplicados con operaciones vectorizadas
        duplicados = self.data.duplicated(subset=[date_col], keep=False)
        if duplicados.any():
            print(f"\nAdvertencia: {duplicados.sum()//2} timestamps duplicados.")
            
            # Estrategia optimizada para selección de valores
            mask = self.data[date_col].duplicated(keep=False)
            df_dups = self.data[mask]
            
            # Calcular diferencias absolutas respecto a la media del grupo
            df_dups['diff'] = df_dups.groupby(date_col)[glucose_col].transform(
                lambda x: (x - x.mean()).abs()
            )
            
            # Filtrar el valor más cercano a la media por grupo
            idx = df_dups.groupby(date_col)['diff'].idxmin()
            self.data = pd.concat([
                self.data[~mask],
                df_dups.loc[idx].drop(columns='diff')
            ]).sort_values(date_col)

        # Ordenación final optimizada
        self.data = self.data.sort_values(date_col, ignore_index=True)
        
        # Calculamos las diferencias de tiempo una sola vez y las almacenamos
        self.time_diffs = self.data['time'].diff()
        
        # Validación final
        if not pd.api.types.is_datetime64_any_dtype(self.data[date_col]):
            raise ValueError("Error en conversión de fechas")

    def _calculate_typical_interval(self) -> float:
        """
        Calcula el intervalo típico entre mediciones en minutos.
        
        :return: Intervalo típico en minutos
        """
        # Utilizamos las diferencias ya calculadas
        intervalo = self.time_diffs.median().total_seconds() / 60
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
        
        # Usamos las diferencias ya calculadas
        desconexiones = self.time_diffs[self.time_diffs > umbral_desconexion]
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

    def get_logs(self) -> dict:
        """
        Devuelve todos los logs almacenados.
        
        :return: Diccionario con todos los logs generados
        """
        if not self.log:
            print("Logging no está activado. Inicialice la clase con log=True para generar logs.")
            return {}
        return self.logs

    

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

class Libreview(GlucoseData):
    def __init__(self, file_path: str, header: int = 2,
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
            date_col="Sello de tiempo del dispositivo", 
            glucose_col="Historial de glucosa mg/dL",
            header=header,
            start_date=start_date,
            end_date=end_date
        ) 







       