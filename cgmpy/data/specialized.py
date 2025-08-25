"""
Módulo con clases especializadas para dispositivos específicos de glucosa.
"""
import datetime
from typing import Union
from .core import ModularGlucoseData


class Dexcom(ModularGlucoseData):
    """
    Clase especializada para datos de dispositivos Dexcom.
    
    Esta clase hereda de ModularGlucoseData y configura automáticamente
    los nombres de columnas específicos para archivos exportados de Dexcom Clarity.
    """
    
    def __init__(
        self,
        file_path: str,
        start_date: Union[str, datetime.datetime, None] = None,
        end_date: Union[str, datetime.datetime, None] = None,
        log: bool = False,
    ):
        """
        Inicializa los datos de Dexcom.
        
        :param file_path: Ruta al archivo CSV exportado de Clarity
        :param start_date: Filtro opcional de fecha inicial (YYYY-MM-DD)
        :param end_date: Filtro opcional de fecha final (YYYY-MM-DD)
        :param log: Si True, activa logs detallados de rendimiento
        
        Ejemplo de uso:
        >>> dexcom = Dexcom("datos_dexcom.csv")
        >>> print(dexcom.info())
        """
        super().__init__(
            data_source=file_path,
            date_col="Marca temporal (AAAA-MM-DDThh:mm:ss)",
            glucose_col="Nivel de glucosa (mg/dL)",
            start_date=start_date,
            end_date=end_date,
            log=log,
        )
    
    def __str__(self) -> str:
        """Representación personalizada para Dexcom."""
        info = self.info()
        return (
            f"Datos de Dexcom: {info['num_datos']} lecturas entre {info['fecha_inicio']} y {info['fecha_fin']}.\n"
            f"Intervalo típico: {info['intervalo_tipico']:.1f} minutos.\n"
            f"Disponibilidad: {info['porcentaje_disponibilidad']:.1f}%\n"
            f"Desconexiones detectadas: {info['num_desconexiones']}\n"
            f"Uso de memoria: {info['uso_memoria_mb']:.2f} MB"
        )


class Libreview(ModularGlucoseData):
    """
    Clase especializada para datos de dispositivos Libreview.
    
    Esta clase hereda de ModularGlucoseData y configura automáticamente
    los nombres de columnas específicos para archivos exportados de Libreview.
    """
    
    def __init__(
        self,
        file_path: str,
        header: int = 2,
        start_date: Union[str, datetime.datetime, None] = None,
        end_date: Union[str, datetime.datetime, None] = None,
        log: bool = False,
    ):
        """
        Inicializa los datos de Libreview.
        
        :param file_path: Ruta al archivo CSV exportado de Libreview
        :param header: Fila del encabezado (normalmente 2 para Libreview)
        :param start_date: Filtro opcional de fecha inicial (YYYY-MM-DD)
        :param end_date: Filtro opcional de fecha final (YYYY-MM-DD)
        :param log: Si True, activa logs detallados de rendimiento
        
        Ejemplo de uso:
        >>> libreview = Libreview("datos_libreview.csv")
        >>> print(libreview.info())
        """
        super().__init__(
            data_source=file_path,
            date_col="Sello de tiempo del dispositivo",
            glucose_col="Historial de glucosa mg/dL",
            header=header,
            start_date=start_date,
            end_date=end_date,
            log=log,
        )
    
    def __str__(self) -> str:
        """Representación personalizada para Libreview."""
        info = self.info()
        return (
            f"Datos de Libreview: {info['num_datos']} lecturas entre {info['fecha_inicio']} y {info['fecha_fin']}.\n"
            f"Intervalo típico: {info['intervalo_tipico']:.1f} minutos.\n"
            f"Disponibilidad: {info['porcentaje_disponibilidad']:.1f}%\n"
            f"Desconexiones detectadas: {info['num_desconexiones']}\n"
            f"Uso de memoria: {info['uso_memoria_mb']:.2f} MB"
        )


class MedtronicCarelink(ModularGlucoseData):
    """
    Clase especializada para datos de dispositivos Medtronic CareLink.
    
    Esta clase hereda de ModularGlucoseData y configura automáticamente
    los nombres de columnas específicos para archivos exportados de CareLink.
    """
    
    def __init__(
        self,
        file_path: str,
        start_date: Union[str, datetime.datetime, None] = None,
        end_date: Union[str, datetime.datetime, None] = None,
        log: bool = False,
    ):
        """
        Inicializa los datos de Medtronic CareLink.
        
        :param file_path: Ruta al archivo CSV exportado de CareLink
        :param start_date: Filtro opcional de fecha inicial (YYYY-MM-DD)
        :param end_date: Filtro opcional de fecha final (YYYY-MM-DD)
        :param log: Si True, activa logs detallados de rendimiento
        
        Ejemplo de uso:
        >>> carelink = MedtronicCarelink("datos_carelink.csv")
        >>> print(carelink.info())
        """
        super().__init__(
            data_source=file_path,
            date_col="Fecha y hora",
            glucose_col="Valor del sensor (mg/dL)",
            start_date=start_date,
            end_date=end_date,
            log=log,
        )
    
    def __str__(self) -> str:
        """Representación personalizada para Medtronic CareLink."""
        info = self.info()
        return (
            f"Datos de Medtronic CareLink: {info['num_datos']} lecturas entre {info['fecha_inicio']} y {info['fecha_fin']}.\n"
            f"Intervalo típico: {info['intervalo_tipico']:.1f} minutos.\n"
            f"Disponibilidad: {info['porcentaje_disponibilidad']:.1f}%\n"
            f"Desconexiones detectadas: {info['num_desconexiones']}\n"
            f"Uso de memoria: {info['uso_memoria_mb']:.2f} MB"
        )


class TandemDiabetes(ModularGlucoseData):
    """
    Clase especializada para datos de dispositivos Tandem Diabetes.
    
    Esta clase hereda de ModularGlucoseData y configura automáticamente
    los nombres de columnas específicos para archivos exportados de Tandem.
    """
    
    def __init__(
        self,
        file_path: str,
        start_date: Union[str, datetime.datetime, None] = None,
        end_date: Union[str, datetime.datetime, None] = None,
        log: bool = False,
    ):
        """
        Inicializa los datos de Tandem Diabetes.
        
        :param file_path: Ruta al archivo CSV exportado de Tandem
        :param start_date: Filtro opcional de fecha inicial (YYYY-MM-DD)
        :param end_date: Filtro opcional de fecha final (YYYY-MM-DD)
        :param log: Si True, activa logs detallados de rendimiento
        
        Ejemplo de uso:
        >>> tandem = TandemDiabetes("datos_tandem.csv")
        >>> print(tandem.info())
        """
        super().__init__(
            data_source=file_path,
            date_col="Timestamp",
            glucose_col="CGM Glucose Value (mg/dL)",
            start_date=start_date,
            end_date=end_date,
            log=log,
        )
    
    def __str__(self) -> str:
        """Representación personalizada para Tandem Diabetes."""
        info = self.info()
        return (
            f"Datos de Tandem Diabetes: {info['num_datos']} lecturas entre {info['fecha_inicio']} y {info['fecha_fin']}.\n"
            f"Intervalo típico: {info['intervalo_tipico']:.1f} minutos.\n"
            f"Disponibilidad: {info['porcentaje_disponibilidad']:.1f}%\n"
            f"Desconexiones detectadas: {info['num_desconexiones']}\n"
            f"Uso de memoria: {info['uso_memoria_mb']:.2f} MB"
        )


def detect_device_type(file_path: str) -> str:
    """
    Detecta automáticamente el tipo de dispositivo basado en el archivo.
    
    :param file_path: Ruta al archivo CSV
    :return: Tipo de dispositivo detectado
    """
    import pandas as pd
    
    try:
        # Leer las primeras filas para detectar el formato
        sample = pd.read_csv(file_path, nrows=5)
        columns = sample.columns.tolist()
        
        # Detectar por nombres de columnas característicos
        if "Marca temporal (AAAA-MM-DDThh:mm:ss)" in columns:
            return "dexcom"
        elif "Sello de tiempo del dispositivo" in columns:
            return "libreview"
        elif "Fecha y hora" in columns and "Valor del sensor (mg/dL)" in columns:
            return "medtronic"
        elif "Timestamp" in columns and "CGM Glucose Value (mg/dL)" in columns:
            return "tandem"
        else:
            return "unknown"
            
    except Exception:
        return "unknown"


def create_specialized_loader(file_path: str, device_type: str = None, **kwargs):
    """
    Crea automáticamente el cargador especializado apropiado.
    
    :param file_path: Ruta al archivo
    :param device_type: Tipo de dispositivo (si None, se detecta automáticamente)
    :param kwargs: Argumentos adicionales para el constructor
    :return: Instancia del cargador especializado apropiado
    """
    if device_type is None:
        device_type = detect_device_type(file_path)
    
    device_type = device_type.lower()
    
    if device_type == "dexcom":
        return Dexcom(file_path, **kwargs)
    elif device_type == "libreview":
        return Libreview(file_path, **kwargs)
    elif device_type == "medtronic":
        return MedtronicCarelink(file_path, **kwargs)
    elif device_type == "tandem":
        return TandemDiabetes(file_path, **kwargs)
    else:
        # Usar el cargador genérico si no se reconoce el tipo
        from .core import ModularGlucoseData
        return ModularGlucoseData(file_path, **kwargs) 