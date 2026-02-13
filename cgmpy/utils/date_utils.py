"""
Módulo de utilidades para manejo de fechas.

Este módulo contiene funciones auxiliares para el procesamiento
y validación de fechas en diferentes formatos.
"""

import datetime
from typing import Union

import pandas as pd


def parse_date(
    date_string: Union[str, pd.Timestamp, datetime.datetime],
) -> pd.Timestamp:
    """
    Parsea una fecha desde diferentes formatos comunes.

    Args:
        date_string: Fecha en formato string, Timestamp o datetime

    Returns:
        pd.Timestamp: Fecha parseada

    Raises:
        ValueError: Si no se puede parsear la fecha
    """
    # Si ya es un Timestamp o datetime, devolverlo directamente
    if isinstance(date_string, (pd.Timestamp, datetime.datetime)):
        return pd.Timestamp(date_string)

    # Si es string, intentar parsear con diferentes formatos
    if isinstance(date_string, str):
        formats = [
            "%Y-%m-%d",  # Formato ISO básico: 2024-03-27
            "%d/%m/%Y %H:%M",  # Formato 1: 07/10/2022 00:00
            "%Y-%m-%dT%H:%M:%S",  # Formato 2: 2023-02-01T01:08:04
            "%d-%m-%Y %H:%M",  # Formato 3: 21-03-2023 16:01
            "%Y-%m-%d %H:%M:%S",  # Formato 4: 2022-07-24 00:12:00
        ]

        for fmt in formats:
            try:
                return pd.to_datetime(date_string, format=fmt)
            except ValueError:
                continue

        raise ValueError(f"No se pudo parsear la fecha: {date_string}")

    raise ValueError(f"Tipo de dato no soportado: {type(date_string)}")


def validate_date_range(
    start_date: Union[str, pd.Timestamp, datetime.datetime, None],
    end_date: Union[str, pd.Timestamp, datetime.datetime, None],
) -> tuple:
    """
    Valida y parsea un rango de fechas.

    Args:
        start_date: Fecha de inicio (opcional)
        end_date: Fecha de fin (opcional)

    Returns:
        tuple: (start_date_parsed, end_date_parsed)

    Raises:
        ValueError: Si las fechas son inválidas o el rango es incorrecto
    """
    start_parsed = None
    end_parsed = None

    if start_date is not None:
        start_parsed = parse_date(start_date)

    if end_date is not None:
        end_parsed = parse_date(end_date)

    # Validar que start_date <= end_date si ambas están presentes
    if start_parsed is not None and end_parsed is not None:
        if start_parsed > end_parsed:
            raise ValueError("La fecha de inicio debe ser anterior a la fecha de fin")

    return start_parsed, end_parsed


def format_date_for_display(date: Union[pd.Timestamp, datetime.datetime]) -> str:
    """
    Formatea una fecha para mostrar en pantalla.

    Args:
        date: Fecha a formatear

    Returns:
        str: Fecha formateada como string
    """
    if isinstance(date, (pd.Timestamp, datetime.datetime)):
        return date.strftime("%d/%m/%Y %H:%M")
    return str(date)


def get_date_components(date: Union[pd.Timestamp, datetime.datetime]) -> dict:
    """
    Extrae los componentes de una fecha.

    Args:
        date: Fecha a analizar

    Returns:
        dict: Diccionario con año, mes, día, hora, minuto, segundo
    """
    if isinstance(date, (pd.Timestamp, datetime.datetime)):
        return {
            "year": date.year,
            "month": date.month,
            "day": date.day,
            "hour": date.hour,
            "minute": date.minute,
            "second": date.second,
            "weekday": date.weekday(),
            "weekday_name": date.strftime("%A"),
        }
    return {}


def calculate_date_difference(
    date1: Union[pd.Timestamp, datetime.datetime],
    date2: Union[pd.Timestamp, datetime.datetime],
    unit: str = "days",
) -> float:
    """
    Calcula la diferencia entre dos fechas.

    Args:
        date1: Primera fecha
        date2: Segunda fecha
        unit: Unidad de tiempo ('days', 'hours', 'minutes', 'seconds')

    Returns:
        float: Diferencia en la unidad especificada
    """
    if not isinstance(date1, (pd.Timestamp, datetime.datetime)) or not isinstance(
        date2, (pd.Timestamp, datetime.datetime)
    ):
        raise ValueError("Ambas fechas deben ser Timestamp o datetime")

    diff = abs(date2 - date1)

    if unit == "days":
        return diff.total_seconds() / (24 * 3600)
    elif unit == "hours":
        return diff.total_seconds() / 3600
    elif unit == "minutes":
        return diff.total_seconds() / 60
    elif unit == "seconds":
        return diff.total_seconds()
    else:
        raise ValueError(f"Unidad no soportada: {unit}")


def is_business_day(date: Union[pd.Timestamp, datetime.datetime]) -> bool:
    """
    Verifica si una fecha es un día laboral (lunes a viernes).

    Args:
        date: Fecha a verificar

    Returns:
        bool: True si es día laboral, False en caso contrario
    """
    if isinstance(date, (pd.Timestamp, datetime.datetime)):
        return date.weekday() < 5  # 0-4 son lunes a viernes
    return False


def get_quarter_dates(date: Union[pd.Timestamp, datetime.datetime]) -> tuple:
    """
    Obtiene las fechas de inicio y fin del trimestre para una fecha dada.

    Args:
        date: Fecha de referencia

    Returns:
        tuple: (inicio_trimestre, fin_trimestre)
    """
    if isinstance(date, (pd.Timestamp, datetime.datetime)):
        year = date.year
        quarter = (date.month - 1) // 3 + 1

        start_month = (quarter - 1) * 3 + 1
        end_month = quarter * 3

        start_date = pd.Timestamp(year, start_month, 1)
        if end_month == 12:
            end_date = pd.Timestamp(year + 1, 1, 1) - pd.Timedelta(days=1)
        else:
            end_date = pd.Timestamp(year, end_month + 1, 1) - pd.Timedelta(days=1)

        return start_date, end_date

    raise ValueError("Fecha debe ser Timestamp o datetime")
