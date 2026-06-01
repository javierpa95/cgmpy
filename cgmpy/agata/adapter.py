# cgmpy/agata/adapter.py

import pandas as pd

from ..data.core import ModularGlucoseData


def prepare_data_for_agata(
    glucose_data: ModularGlucoseData, resample_freq: str = "5min"
) -> pd.DataFrame:
    """
    Prepara los datos de un objeto cgmpy para ser analizados por py_agata,
    manejando tiempos de inicio no alineados.
    """
    df = glucose_data.data.copy()
    date_col = glucose_data.date_col
    glucose_col = glucose_data.glucose_col

    # Asegurarse de que la columna de fecha es de tipo datetime
    df[date_col] = pd.to_datetime(df[date_col])

    # Ordenar los datos cronológicamente primero
    df_limpio = df.sort_values(date_col)

    # -- PASO CLAVE: ESTANDARIZAR LA COLUMNA DE TIEMPO --
    # Redondea cada tiempo hacia abajo al intervalo de 5 minutos más cercano.
    df_limpio[date_col] = df_limpio[date_col].dt.floor(resample_freq)

    # Ahora, eliminamos duplicados. Si 00:01 y 00:03 se redondearon a 00:00,
    # nos quedamos con el primero que apareció.
    df_limpio = df_limpio.drop_duplicates(subset=[date_col], keep="first")

    # El resto del proceso ya funciona perfectamente
    tiempo_inicio = df_limpio[date_col].min()
    tiempo_fin = df_limpio[date_col].max()
    rango_tiempo = pd.date_range(start=tiempo_inicio, end=tiempo_fin, freq=resample_freq)
    df_homogeneo = pd.DataFrame({date_col: rango_tiempo})

    # Fusionar con los datos ya estandarizados
    df_final = df_homogeneo.merge(df_limpio, on=date_col, how="left")

    # Renombrar columnas
    df_final = df_final.rename(columns={date_col: "t", glucose_col: "glucose"})

    return df_final[["t", "glucose"]]
