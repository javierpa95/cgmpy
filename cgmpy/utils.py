import pandas as pd

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