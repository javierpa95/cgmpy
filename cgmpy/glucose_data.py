import datetime
import pandas as pd
from typing import Union
from .utils import parse_date
import os
import numpy as np
import time


class GlucoseData:
    def __init__(self, data_source: Union[str, pd.DataFrame], date_col: str="time", glucose_col: str="glucose", delimiter: Union[str, None] = None, header: int = 0, start_date: Union[str, datetime.datetime, None] = None, end_date: Union[str, datetime.datetime, None] = None, log: bool = False):
        """
        Inicializa los datos de glucosa con soporte para CSV y Parquet.
        
        :param data_source: Archivo CSV/Parquet o DataFrame con los datos de glucosa
        :param date_col: Nombre de la columna de fecha/hora
        :param glucose_col: Nombre de la columna de valores de glucosa
        :param delimiter: Delimitador para archivos CSV
        :param header: Fila de encabezado para archivos CSV
        :param start_date: Fecha de inicio para filtrar datos (opcional)
        :param end_date: Fecha de fin para filtrar datos (opcional)
        :param log: Si True, guarda información detallada de las operaciones realizadas
        
        Nota: Para archivos Parquet, se espera que la columna 'time' esté en formato datetime
        y la columna 'glucose' en formato int16 para un rendimiento óptimo.
        """
        self.log = log  # Atributo para controlar logging en toda la clase
        self.logs = {}  # Diccionario para almacenar los logs de diferentes operaciones
        
        self.date_col = date_col
        self.glucose_col = glucose_col
        self.typical_interval = None  # Inicializamos el atributo
        
        t_start = time.time()

        # Carga del archivo
        t0 = time.time()
        if isinstance(data_source, str):
            if os.path.isfile(data_source):
                # Detectar si es un archivo Parquet por la extensión
                is_parquet = data_source.lower().endswith('.parquet')
                
                if is_parquet:
                    # Procesamiento para archivos Parquet
                    try:
                        self.data = pd.read_parquet(data_source, columns=[date_col, glucose_col])
                    except Exception as e:
                        raise ValueError(f"Error al leer el archivo Parquet: {str(e)}")
                else:
                    # Procesamiento para archivos CSV
                    try:
                        # Si no se especifica delimitador, intentar con coma
                        if delimiter is None:
                            delimiter = ','
                        
                        # Intentar leer el archivo con el delimitador especificado
                        self.data = pd.read_csv(
                            data_source, 
                            delimiter=delimiter, 
                            header=header, 
                            usecols=[date_col, glucose_col]
                        )
                    except Exception as e:
                        # Si falla con coma, intentar con punto y coma
                        if delimiter == ',':
                            try:
                                self.data = pd.read_csv(
                                    data_source, 
                                    delimiter=';', 
                                    header=header, 
                                    usecols=[date_col, glucose_col]
                                )
                            except Exception:
                                # Si ambos fallan, lanzar el error original
                                raise ValueError(
                                    f"Error al leer el archivo CSV: {str(e)}. "
                                    "Intente especificar manualmente el delimitador con el parámetro 'delimiter'."
                                ) from e
                        else:
                            # Si se especificó un delimitador y falló, lanzar el error
                            raise ValueError(f"Error al leer el archivo CSV con delimitador '{delimiter}': {str(e)}")
            else:
                raise FileNotFoundError(f"Archivo no encontrado: {data_source}")
        elif isinstance(data_source, pd.DataFrame):
            # Es un DataFrame, usarlo directamente
            self.data = data_source.copy()
        else:
            raise ValueError("data_source debe ser un archivo CSV o un DataFrame")
        t1 = time.time()

        # Renombrar columnas
        t2 = time.time()
        if date_col != "time":
            self.data = self.data.rename(columns={date_col: "time"})
        if glucose_col != "glucose":
            self.data = self.data.rename(columns={glucose_col: "glucose"})
        t3 = time.time()

        # Filtrado por fechas
        t4 = time.time()
        if start_date is not None:
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            self.data = self.data[self.data["time"] >= start_date]
        
        if end_date is not None:
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            self.data = self.data[self.data["time"] <= end_date]
        t5 = time.time()
        
        # Completa el procesamiento de datos
        t6 = time.time()
        self._validate_columns("time", "glucose")
        t7 = time.time()
        self._process_data("time", "glucose")
        t8 = time.time()

        # Cálculo del intervalo típico
        t9 = time.time()
        self.typical_interval = self._calculate_typical_interval()
        t10 = time.time()

        t_end = time.time()

        if self.log:
            print(f"""
            Tiempos de carga de datos:
            Lectura del archivo: {t1-t0:.3f}s
            Renombrado de columnas: {t3-t2:.3f}s
            Filtrado por fechas: {t5-t4:.3f}s
            Validación de columnas: {t7-t6:.3f}s
            Procesamiento de datos: {t8-t7:.3f}s
            Cálculo de intervalo típico: {t10-t9:.3f}s
            Tiempo total: {t_end-t_start:.3f}s
            """)
        else:
            # Mensaje simple si log=False
            print(f"Archivo cargado en {t_end-t_start:.3f} segundos")

    def _validate_columns(self, date_col: str, glucose_col: str):
        """Valida que las columnas especificadas existan en el DataFrame."""
        if date_col not in self.data.columns or glucose_col not in self.data.columns:
            raise ValueError(f"Las columnas '{date_col}' o '{glucose_col}' no se encuentran en el archivo CSV. Columnas disponibles: {self.data.columns.tolist()}.")

    def _process_data(self, date_col: str, glucose_col: str):
        """Procesa los datos optimizando operaciones costosas según el origen."""
        t_start = time.time()
        if self.log:
            print("\n--- ANÁLISIS DETALLADO DE RENDIMIENTO ---")
        
        # Paso 1: Determinar origen de datos
        from_parquet = (pd.api.types.is_datetime64_any_dtype(self.data[date_col]) and 
                       (self.data[glucose_col].dtype == 'int16'))
        
        if from_parquet:
            if self.log:
                print("Detectados datos de origen Parquet con tipos optimizados.")
                print("Aplicando ruta rápida para datos Parquet...")
            
            # Paso 1: Verificar si hay nulos (muy rápido)
            t_nulos = time.time()
            if self.data.isna().any().any():
                self.data = self.data.dropna(subset=[date_col, glucose_col])
                if self.log:
                    print(f"  - Eliminados valores nulos: {time.time() - t_nulos:.3f}s")
            elif self.log:
                print(f"  - No hay valores nulos: {time.time() - t_nulos:.3f}s")
            
            # Paso 2: Verificar ordenación (muy rápido)
            t_orden = time.time()
            if not self.data[date_col].is_monotonic_increasing:
                if self.log:
                    print("  - Ordenando datos...")
                self.data = self.data.sort_values(date_col, ignore_index=True)
            elif self.log:
                print("  - Datos ya ordenados")
            
            if self.log:
                print(f"  - Verificación de orden: {time.time() - t_orden:.3f}s")
            
            # Paso 3: Cálculo eficiente de diferencias de tiempo
            t_diff = time.time()
            # Usar NumPy para cálculo más rápido
            time_values = self.data[date_col].values
            # Crear array de diferencias con NumPy (mucho más rápido)
            time_diffs_ns = np.diff(time_values.astype('datetime64[ns]'))
            # Añadir un 0 al inicio para mantener la misma longitud
            time_diffs_ns = np.insert(time_diffs_ns, 0, np.timedelta64(0, 'ns'))
            # Convertir a Series de pandas
            self.time_diffs = pd.Series(
                pd.TimedeltaIndex(time_diffs_ns),
                index=self.data.index
            )
            if self.log:
                print(f"  - Cálculo optimizado de diferencias: {time.time() - t_diff:.3f}s")
            
        else:
            if self.log:
                print("Procesando datos con conversión de tipos y validaciones completas.")
            
            # RUTA NORMAL: Mantener todas las comprobaciones para CSV y otros formatos
            # Paso 2: Manejo de valores nulos
            t_nulos = time.time()
            filas_antes = len(self.data)
            self.data = self.data.dropna(subset=[date_col, glucose_col])
            filas_despues = len(self.data)
            if self.log:
                print(f"2. Eliminación de nulos: {time.time() - t_nulos:.3f}s (Eliminadas {filas_antes - filas_despues} filas)")
            
            # Paso 3: Conversión de tipos
            t_tipos = time.time()
            if not pd.api.types.is_datetime64_any_dtype(self.data[date_col]):
                if self.log:
                    print(f"  - Convirtiendo columna '{date_col}' a datetime...")
                if pd.api.types.is_numeric_dtype(self.data[date_col]):
                    unit = 'ms' if self.data[date_col].iloc[0] > 1e10 else 's'
                    self.data[date_col] = pd.to_datetime(self.data[date_col], unit=unit)
                else:
                    self.data[date_col] = pd.to_datetime(
                        self.data[date_col],
                        errors='coerce',
                        format='mixed'
                    )
            
            # Modificación aquí: Asegurarse de que la columna de glucosa sea numérica
            if not pd.api.types.is_numeric_dtype(self.data[glucose_col]):
                if self.log:
                    print(f"  - Convirtiendo columna '{glucose_col}' a numérica...")
                # Convertir a numérico, forzando errores a NaN
                self.data[glucose_col] = pd.to_numeric(self.data[glucose_col], errors='coerce')
                # Eliminar filas con valores NaN después de la conversión
                self.data = self.data.dropna(subset=[glucose_col])
            
            # Ahora que sabemos que es numérica, podemos intentar optimizar
            if self.data[glucose_col].dtype != 'int16':
                if self.log:
                    print(f"  - Optimizando columna '{glucose_col}'...")
                min_val = self.data[glucose_col].min()
                max_val = self.data[glucose_col].max()
                
                # Ahora podemos hacer la comparación con seguridad
                if pd.notna(min_val) and pd.notna(max_val) and min_val >= -32768 and max_val <= 32767:
                    self.data[glucose_col] = self.data[glucose_col].astype('int16')
                else:
                    self.data[glucose_col] = pd.to_numeric(self.data[glucose_col], errors='coerce', downcast='float')
            
            if self.log:
                print(f"3. Conversión de tipos: {time.time() - t_tipos:.3f}s")
            
            # Paso 4: Manejo de duplicados
            t_dups = time.time()
            duplicados = self.data.duplicated(subset=[date_col], keep=False)
            num_duplicados = duplicados.sum()
            if self.log:
                print(f"4a. Detección de duplicados: {time.time() - t_dups:.3f}s (Encontrados {num_duplicados//2 if num_duplicados > 0 else 0} duplicados)")
            
            if duplicados.any():
                t_proc_dups = time.time()
                # Estrategia para manejar duplicados
                mask = self.data[date_col].duplicated(keep=False)
                df_dups = self.data[mask].copy()
                df_dups['diff'] = df_dups.groupby(date_col)[glucose_col].transform(
                    lambda x: (x - x.mean()).abs()
                )
                
                # Filtrar el valor más cercano a la media por grupo
                idx = df_dups.groupby(date_col)['diff'].idxmin()
                self.data = pd.concat([
                    self.data[~mask],
                    df_dups.loc[idx].drop(columns='diff')
                ])
                if self.log:
                    print(f"4b. Procesamiento de duplicados: {time.time() - t_proc_dups:.3f}s")
            
            # Paso 5: Ordenación
            t_orden = time.time()
            if not self.data[date_col].is_monotonic_increasing:
                if self.log:
                    print("  - Ordenando datos por timestamp...")
                self.data = self.data.sort_values(date_col, ignore_index=True)
            elif self.log:
                print("  - Datos ya ordenados, omitiendo ordenación.")
            if self.log:
                print(f"5. Ordenación: {time.time() - t_orden:.3f}s")
            
            # Paso 6: Cálculo de diferencias de tiempo
            t_diff = time.time()
            self.time_diffs = self.data[date_col].diff()
            if self.log:
                print(f"6. Cálculo de diferencias: {time.time() - t_diff:.3f}s")
        
        # Validación final (común para ambas rutas)
        t_valid = time.time()
        if not pd.api.types.is_datetime64_any_dtype(self.data[date_col]):
            raise ValueError("Error en conversión de fechas")
        if self.log:
            print(f"7. Validación final: {time.time() - t_valid:.3f}s")
            
            # Información de memoria
            memoria_bytes = self.data.memory_usage(deep=True).sum()
            memoria_mb = memoria_bytes / (1024 * 1024)
            print(f"Uso de memoria del DataFrame: {memoria_mb:.2f} MB")
            
            t_end = time.time()
            print(f"Tiempo total de procesamiento: {t_end - t_start:.3f}s")
            print("--- FIN DEL ANÁLISIS ---\n")

    def _calculate_typical_interval(self) -> float:
        """Calcula el intervalo típico entre mediciones en minutos de forma optimizada."""
        t_start = time.time()
        if self.log:
            print("\n--- ANÁLISIS DE CÁLCULO DE INTERVALO TÍPICO ---")
        
        # Optimización: usar NumPy para cálculo más rápido de la mediana
        t_calc = time.time()
        # Convertir a array de NumPy para operaciones más rápidas
        time_diffs_seconds = self.time_diffs.dt.total_seconds().values
        # Filtrar valores válidos (mayores que 0)
        valid_diffs = time_diffs_seconds[time_diffs_seconds > 0]
        
        if len(valid_diffs) > 0:
            # Usar NumPy para calcular la mediana (mucho más rápido)
            intervalo = np.median(valid_diffs) / 60
        else:
            # Valor predeterminado si no hay diferencias válidas
            intervalo = 5.0
        
        if self.log:
            print(f"Cálculo optimizado de mediana: {time.time() - t_calc:.3f}s")
            
            t_end = time.time()
            print(f"Tiempo total de cálculo de intervalo: {t_end - t_start:.3f}s")
            print("--- FIN DEL ANÁLISIS ---\n")
        
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

    # Método para guardar datos en formato Parquet
    def to_parquet(self, file_path: str, compression: str = 'snappy', sort: bool = True):
        """
        Guarda los datos en formato Parquet optimizado para carga rápida.
        
        :param file_path: Ruta donde guardar el archivo Parquet
        :param compression: Algoritmo de compresión ('snappy', 'gzip', 'brotli', etc.)
        :param sort: Si es True, ordena los datos por tiempo antes de guardar
        
        Nota: Los datos se guardan con 'time' como datetime y 'glucose' como int16
        para un rendimiento y almacenamiento óptimos.
        """
        print(f"Preparando datos para guardar en formato Parquet optimizado...")
        
        # Asegurar que los tipos sean óptimos antes de guardar
        df_to_save = self.data.copy()
        
        # Paso 1: Ordenar por tiempo (si se solicita)
        if sort and not df_to_save['time'].is_monotonic_increasing:
            print("  - Ordenando datos por timestamp...")
            df_to_save = df_to_save.sort_values('time', ignore_index=True)
        
        # Paso 2: Convertir glucose a int16 si es posible
        if not pd.api.types.is_integer_dtype(df_to_save['glucose']):
            print("  - Convirtiendo 'glucose' a formato numérico...")
            df_to_save['glucose'] = pd.to_numeric(df_to_save['glucose'], errors='coerce')
        
        # Intentar convertir a int16 si los valores lo permiten
        min_val = df_to_save['glucose'].min()
        max_val = df_to_save['glucose'].max()
        
        if pd.notna(min_val) and pd.notna(max_val) and min_val >= -32768 and max_val <= 32767:
            print(f"  - Optimizando 'glucose' a int16 (rango: {min_val} a {max_val})...")
            df_to_save['glucose'] = df_to_save['glucose'].astype('int16')
        else:
            print(f"  - Valores de glucosa fuera del rango de int16 ({min_val} a {max_val}). Usando int32.")
            df_to_save['glucose'] = df_to_save['glucose'].astype('int32')
        
        # Paso 3: Verificar que time sea datetime
        if not pd.api.types.is_datetime64_any_dtype(df_to_save['time']):
            print("  - Convirtiendo 'time' a datetime...")
            df_to_save['time'] = pd.to_datetime(df_to_save['time'], errors='coerce')
        
        # Paso 4: Eliminar duplicados si existen
        duplicados = df_to_save.duplicated(subset=['time'], keep='first')
        if duplicados.any():
            num_duplicados = duplicados.sum()
            print(f"  - Eliminando {num_duplicados} timestamps duplicados...")
            df_to_save = df_to_save.drop_duplicates(subset=['time'], keep='first')
        
        # Paso 5: Guardar en formato Parquet
        t_start = time.time()
        df_to_save.to_parquet(
            file_path, 
            compression=compression,
            index=False,
            engine='pyarrow'
        )
        t_end = time.time()
        
        # Información final
        file_size = os.path.getsize(file_path)/1024/1024
        print(f"Datos guardados en formato Parquet en: {file_path}")
        print(f"  - Tamaño del archivo: {file_size:.2f} MB")
        print(f"  - Tiempo de guardado: {t_end - t_start:.3f}s")
        print(f"  - Registros guardados: {len(df_to_save):,}")
        print(f"  - Rango de fechas: {df_to_save['time'].min()} a {df_to_save['time'].max()}")
        print(f"  - Formato listo para carga rápida")

    def append_to_parquet(self, file_path: str, compression: str = 'snappy', 
                         handle_duplicates: str = 'keep_new'):
        """
        Añade los datos actuales a un archivo Parquet existente, manejando duplicados.
        
        :param file_path: Ruta al archivo Parquet existente
        :param compression: Algoritmo de compresión ('snappy', 'gzip', 'brotli', etc.)
        :param handle_duplicates: Estrategia para manejar duplicados:
                                 'keep_new': Mantiene los nuevos datos en caso de duplicados
                                 'keep_old': Mantiene los datos existentes en caso de duplicados
                                 'keep_both': Mantiene ambos (no recomendado)
        
        :return: Número de registros añadidos
        """
        if not os.path.exists(file_path):
            print(f"El archivo {file_path} no existe. Creando nuevo archivo...")
            self.to_parquet(file_path, compression=compression)
            return len(self.data)
        
        print(f"Añadiendo datos a archivo Parquet existente: {file_path}")
        
        # Paso 1: Cargar datos existentes
        t_start = time.time()
        existing_data = pd.read_parquet(file_path)
        t_load = time.time()
        print(f"  - Archivo existente cargado en {t_load - t_start:.3f}s")
        print(f"  - Registros existentes: {len(existing_data):,}")
        
        # Paso 2: Preparar nuevos datos
        new_data = self.data.copy()
        
        # Asegurar tipos correctos
        if not pd.api.types.is_datetime64_any_dtype(new_data['time']):
            new_data['time'] = pd.to_datetime(new_data['time'], errors='coerce')
        
        if not pd.api.types.is_integer_dtype(new_data['glucose']):
            new_data['glucose'] = pd.to_numeric(new_data['glucose'], errors='coerce')
            
            # Convertir a int16 si es posible
            min_val = new_data['glucose'].min()
            max_val = new_data['glucose'].max()
            if pd.notna(min_val) and pd.notna(max_val) and min_val >= -32768 and max_val <= 32767:
                new_data['glucose'] = new_data['glucose'].astype('int16')
        
        # Paso 3: Identificar duplicados
        # Crear un DataFrame combinado para identificar duplicados
        combined = pd.concat([existing_data, new_data])
        duplicated_times = combined['time'].duplicated(keep=False)
        num_duplicates = duplicated_times.sum() // 2  # Dividir por 2 porque cada duplicado aparece dos veces
        
        if num_duplicates > 0:
            print(f"  - Encontrados {num_duplicates} timestamps duplicados")
            
            # Manejar duplicados según la estrategia elegida
            if handle_duplicates == 'keep_new':
                print("  - Estrategia: Mantener nuevos datos en caso de duplicados")
                # Marcar duplicados en datos existentes
                existing_times = set(existing_data['time'])
                new_times = set(new_data['time'])
                common_times = existing_times.intersection(new_times)
                
                if common_times:
                    # Filtrar datos existentes para eliminar duplicados
                    existing_data = existing_data[~existing_data['time'].isin(common_times)]
                    
            elif handle_duplicates == 'keep_old':
                print("  - Estrategia: Mantener datos existentes en caso de duplicados")
                # Filtrar nuevos datos para eliminar duplicados
                existing_times = set(existing_data['time'])
                new_data = new_data[~new_data['time'].isin(existing_times)]
                
            # 'keep_both' simplemente concatena todo
        
        # Paso 4: Combinar datos
        t_combine = time.time()
        final_data = pd.concat([existing_data, new_data])
        
        # Paso 5: Ordenar por tiempo
        final_data = final_data.sort_values('time', ignore_index=True)
        t_sort = time.time()
        print(f"  - Datos combinados y ordenados en {t_sort - t_combine:.3f}s")
        
        # Paso 6: Guardar resultado
        final_data.to_parquet(
            file_path, 
            compression=compression,
            index=False,
            engine='pyarrow'
        )
        t_save = time.time()
        
        # Información final
        records_added = len(final_data) - len(existing_data)
        file_size = os.path.getsize(file_path)/1024/1024
        print(f"Datos añadidos correctamente:")
        print(f"  - Registros añadidos: {records_added:,}")
        print(f"  - Total registros: {len(final_data):,}")
        print(f"  - Tamaño del archivo: {file_size:.2f} MB")
        print(f"  - Tiempo total: {t_save - t_start:.3f}s")
        
        return records_added

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
            glucose_col="Nivel de glucosa (mg/dL)",
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







       