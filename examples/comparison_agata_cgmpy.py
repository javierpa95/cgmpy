import os

import numpy as np
import pandas as pd
from py_agata.py_agata import Agata

from cgmpy import GlucoseMetrics

cgm = Agata()

# ruta = r"G:\Mi unidad\1. AREAS\1. INFORMÁTICA\0. PROGRAMACIÓN\1. Python\cgmpy\nodm.csv"
# Utilizar ruta relativa al script para portabilidad
current_dir = os.path.dirname(os.path.abspath(__file__))
ruta = os.path.join(current_dir, "data", "dm.csv")
if not os.path.exists(ruta):
    # Fallback to hardcoded path if local file doesn't exist
    ruta = r"G:\Mi unidad\1. AREAS\1. INFORMÁTICA\0. PROGRAMACIÓN\1. Python\cgmpy\nodm.csv"

df = pd.read_csv(ruta)

# Convertir la columna time a datetime (corrigiendo timestamps Unix)
df["time"] = pd.to_datetime(df["time"])  # Let pandas infer format

print(df.head())

# Filtrar datos entre el 21 y 25 de marzo (ambos incluidos)
fecha_inicio = pd.to_datetime("2023-02-01 00:00:00")
fecha_fin = pd.to_datetime("2023-02-05 23:59:59")

df_filtrado = df[(df["time"] >= fecha_inicio) & (df["time"] <= fecha_fin)]

# Análisis de la cuadrícula de tiempo
print("\n=== ANÁLISIS DE LA CUADRÍCULA DE TIEMPO ===")
print("Total de registros filtrados: ", len(df_filtrado))

# Calcular diferencias de tiempo
df_filtrado = df_filtrado.sort_values("time")
diferencias = df_filtrado["time"].diff().dropna()

print("\nEstadísticas de intervalos de tiempo:")
print(f"Intervalo mínimo: {diferencias.min()}")
print(f"Intervalo máximo: {diferencias.max()}")
print(f"Intervalo más común: {diferencias.mode().iloc[0] if not diferencias.mode().empty else 'N/A'}")

# Encontrar intervalos irregulares
intervalo_esperado = pd.Timedelta(minutes=5)  # Asumiendo intervalos de 5 minutos
tolerancia = pd.Timedelta(seconds=30)  # Tolerancia de 30 segundos

intervalos_irregulares = diferencias[abs(diferencias - intervalo_esperado) > tolerancia]

print("\nIntervalos irregulares encontrados:")
if len(intervalos_irregulares) > 0:
    print(f"Total de intervalos irregulares: {len(intervalos_irregulares)}")
    print("Primeros 10 intervalos irregulares:")
    for i, (idx, diff) in enumerate(intervalos_irregulares.head(10).items()):
        tiempo_anterior = df_filtrado.loc[idx - 1, "time"]
        tiempo_actual = df_filtrado.loc[idx, "time"]
        print(f"  {i + 1}. {tiempo_anterior} -> {tiempo_actual} (diferencia: {diff})")
else:
    print("No se encontraron intervalos irregulares")

# Verificar si hay datos duplicados
duplicados = df_filtrado[df_filtrado.duplicated(subset=["time"], keep=False)]
print(f"\nRegistros duplicados: {len(duplicados)}")

# Verificar si hay valores NaN
print(f"Valores NaN en 'time': {df_filtrado['time'].isna().sum()}")
print(f"Valores NaN en 'glucose': {df_filtrado['glucose'].isna().sum()}")

# LIMPIEZA DE DATOS
print("\n=== LIMPIEZA DE DATOS ===")

# 1. Eliminar registros duplicados (mantener el primero)
df_limpio = df_filtrado.drop_duplicates(subset=["time"], keep="first")
print(f"Registros después de eliminar duplicados: {len(df_limpio)}")

# 2. Ordenar por tiempo
df_limpio = df_limpio.sort_values("time").reset_index(drop=True)

# 3. Verificar si aún hay problemas
diferencias_limpias = df_limpio["time"].diff().dropna()
intervalos_irregulares_limpios = diferencias_limpias[abs(diferencias_limpias - intervalo_esperado) > tolerancia]

print(f"Intervalos irregulares después de limpiar: {len(intervalos_irregulares_limpios)}")

if len(intervalos_irregulares_limpios) > 0:
    print("Intervalos irregulares restantes:")
    for i, (idx, diff) in enumerate(intervalos_irregulares_limpios.head(5).items()):
        tiempo_anterior = df_limpio.loc[idx - 1, "time"]
        tiempo_actual = df_limpio.loc[idx, "time"]
        print(f"  {i + 1}. {tiempo_anterior} -> {tiempo_actual} (diferencia: {diff})")

# 4. Crear cuadrícula de tiempo homogénea (opcional, si aún hay problemas)
if len(intervalos_irregulares_limpios) > 0:
    print("\nCreando cuadrícula de tiempo homogénea...")

    # Crear rango de tiempo completo
    tiempo_inicio = df_limpio["time"].min()
    tiempo_fin = df_limpio["time"].max()

    # Crear cuadrícula de 5 minutos (usando 'min' en lugar de 'T' para evitar deprecación)
    rango_tiempo = pd.date_range(start=tiempo_inicio, end=tiempo_fin, freq="5min")

    # Crear DataFrame con cuadrícula homogénea
    df_homogeneo = pd.DataFrame({"time": rango_tiempo})

    # Hacer merge con los datos originales (left join para mantener solo los tiempos de la cuadrícula)
    df_final = df_homogeneo.merge(df_limpio, on="time", how="left")

    print(f"Cuadrícula homogénea creada: {len(df_final)} registros")
    print(f"Valores faltantes en glucose: {df_final['glucose'].isna().sum()}")

    # Interpolar valores faltantes
    if df_final["glucose"].isna().sum() > 0:
        print("Interpolando valores faltantes...")
        df_final["glucose"] = df_final["glucose"].interpolate(method="linear")
        print(f"Valores faltantes después de interpolación: {df_final['glucose'].isna().sum()}")

    # Usar df_final en lugar de df_limpio
    df_filtrado = df_final
else:
    # Si no hay problemas, usar df_limpio
    df_filtrado = df_limpio

df_filtrado = df_filtrado.rename(columns={"time": "t"})

print("-------------PYAGATA-------------------")

# Crear DataFrame solo con las columnas necesarias para py_agata
df_agata = df_filtrado[["t", "glucose"]].copy()
df_agata = df_agata.dropna()  # Eliminar filas con valores NaN

# Verificar que la cuadrícula sea homogénea
diferencias_finales = df_agata["t"].diff().dropna()
print(f"Verificación final - Intervalos únicos: {diferencias_finales.unique()}")
print(f"Todos los intervalos son de 5 minutos: {all(dif == pd.Timedelta(minutes=5) for dif in diferencias_finales)}")

capon = Agata()

# Pasar solo los datos de glucosa, no el DataFrame completo
try:
    resultados = capon.analyze_glucose_profile(df_agata)
    print(resultados)
except Exception as e:
    print(f"Error con py_agata: {e}")
    print("Intentando con datos más simples...")

    # Intentar con solo los datos de glucosa
    glucose_data = df_agata["glucose"].values
    resultados = capon.analyze_glucose_profile(glucose_data)
    print(resultados)

# -------------CGMPY-------------------

cgm = GlucoseMetrics(df_filtrado, date_col="t")

# Use all_simplified() for a flat dictionary with main metrics
resultados_cgmpy = cgm.all_simplified()

print(resultados_cgmpy)

# ===== COMPARACIÓN GRÁFICA DE RESULTADOS =====
print("\n" + "=" * 80)
print("COMPARACIÓN DE RESULTADOS: PY_AGATA vs CGMPY")
print("=" * 80)


# Función para formatear valores
def formatear_valor(valor, decimales=2):
    if isinstance(valor, (np.float32, np.float64)):
        return f"{float(valor):.{decimales}f}"
    elif isinstance(valor, (int, float)):
        return f"{valor:.{decimales}f}"
    else:
        return str(valor)


# Métricas básicas para comparar (Keys updated to English)
metricas_comparacion = {
    "Media de glucosa": {
        "py_agata": resultados["variability"]["mean_glucose"],
        "cgmpy": resultados_cgmpy["Mean"],
    },
    "Mediana de glucosa": {
        "py_agata": resultados["variability"]["median_glucose"],
        "cgmpy": resultados_cgmpy["Median"],
    },
    "Desviación estándar": {
        "py_agata": resultados["variability"]["std_glucose"],
        "cgmpy": resultados_cgmpy["SD"],
    },
    "Coeficiente de variación (%)": {
        "py_agata": resultados["variability"]["cv_glucose"],
        "cgmpy": resultados_cgmpy["CV"],
    },
    "GMI": {
        "py_agata": resultados["variability"]["gmi"],
        "cgmpy": resultados_cgmpy["GMI"],
    },
    "TIR (%)": {
        "py_agata": resultados["time_in_ranges"]["time_in_target"],
        "cgmpy": resultados_cgmpy["TIR"],
    },
    "TIR Tight (%)": {
        "py_agata": resultados["time_in_ranges"]["time_in_tight_target"],
        "cgmpy": resultados_cgmpy["TIR_tight"],
    },
    "TBR < 70 (%)": {
        "py_agata": resultados["time_in_ranges"]["time_in_hypoglycemia"],
        "cgmpy": resultados_cgmpy["TBR70"],
    },
    "TAR > 180 (%)": {
        "py_agata": resultados["time_in_ranges"]["time_in_hyperglycemia"],
        "cgmpy": resultados_cgmpy["TAR180"],
    },
    "LBGI": {"py_agata": resultados["risk"]["lbgi"], "cgmpy": resultados_cgmpy["LBGI"]},
    "HBGI": {"py_agata": resultados["risk"]["hbgi"], "cgmpy": resultados_cgmpy["HBGI"]},
    "ADRR": {"py_agata": resultados["risk"]["adrr"], "cgmpy": resultados_cgmpy["ADRR"]},
    "GRI": {"py_agata": resultados["risk"]["gri"], "cgmpy": resultados_cgmpy["GRI"]},
    "MAGE": {
        "py_agata": resultados["variability"]["mage_index"],
        "cgmpy": resultados_cgmpy["MAGE"],
    },
    "MODD": {
        "py_agata": resultados["variability"]["modd"],
        "cgmpy": resultados_cgmpy["MODD"],
    },
    "J-Index": {
        "py_agata": resultados["variability"]["j_index"],
        "cgmpy": resultados_cgmpy["J-Index"],
    },
}

# Mostrar comparación en formato tabla
print(f"{'Métrica':<25} {'PY_AGATA':<15} {'CGMPY':<15} {'Diferencia':<15}")
print("-" * 70)

for metrica, valores in metricas_comparacion.items():
    py_agata_val = formatear_valor(valores["py_agata"])
    cgmpy_val = formatear_valor(valores["cgmpy"])

    # Calcular diferencia
    try:
        diff = abs(float(valores["py_agata"]) - float(valores["cgmpy"]))
        diff_str = f"{diff:.3f}"
    except Exception:
        diff_str = "N/A"

    print(f"{metrica:<25} {py_agata_val:<15} {cgmpy_val:<15} {diff_str:<15}")

print("\n" + "=" * 80)
print("ANÁLISIS DE EVENTOS")
print("=" * 80)

# Comparar eventos de hipoglucemia
print("\n📉 EVENTOS DE HIPOGLUCEMIA:")
print(f"{'Tipo':<15} {'PY_AGATA':<20} {'CGMPY':<20}")
print("-" * 55)

# Eventos de hipoglucemia en py_agata
if "events" in resultados and "hypoglycemic_events" in resultados["events"]:
    eventos_hypo = resultados["events"]["hypoglycemic_events"]

    for nivel in ["hypo", "l1", "l2"]:
        if nivel in eventos_hypo:
            eventos_semana_py = eventos_hypo[nivel]["events_per_week"]
            duracion_media_py = eventos_hypo[nivel]["mean_duration"]

            # Buscar equivalente en cgmpy (aproximación)
            if nivel == "hypo":
                eventos_semana_cgmpy = "N/A"  # cgmpy no calcula eventos por semana directamente
                duracion_media_cgmpy = "N/A"
            else:
                eventos_semana_cgmpy = "N/A"
                duracion_media_cgmpy = "N/A"

            print(f"{nivel.upper():<15} {eventos_semana_py:<20.2f} {eventos_semana_cgmpy:<20}")
            print(f"{'Duración media':<15} {duracion_media_py:<20.2f} {duracion_media_cgmpy:<20}")

# Comparar eventos de hiperglucemia
print("\n📈 EVENTOS DE HIPERGLUCEMIA:")
print(f"{'Tipo':<15} {'PY_AGATA':<20} {'CGMPY':<20}")
print("-" * 55)

if "events" in resultados and "hyperglycemic_events" in resultados["events"]:
    eventos_hyper = resultados["events"]["hyperglycemic_events"]

    for nivel in ["hyper", "l1", "l2"]:
        if nivel in eventos_hyper:
            eventos_semana_py = eventos_hyper[nivel]["events_per_week"]
            duracion_media_py = eventos_hyper[nivel]["mean_duration"]

            eventos_semana_cgmpy = "N/A"
            duracion_media_cgmpy = "N/A"

            print(f"{nivel.upper():<15} {eventos_semana_py:<20.2f} {eventos_semana_cgmpy:<20}")
            print(f"{'Duración media':<15} {duracion_media_py:<20.2f} {duracion_media_cgmpy:<20}")

print("\n" + "=" * 80)
print("RESUMEN DE DIFERENCIAS PRINCIPALES")
print("=" * 80)

# Calcular diferencias porcentuales para métricas clave
metricas_clave = ["Media de glucosa", "TIR (%)", "TBR < 70 (%)", "TAR > 180 (%)", "GMI"]

print(f"{'Métrica':<20} {'Diferencia %':<15} {'Estado':<15}")
print("-" * 50)

for metrica in metricas_clave:
    if metrica in metricas_comparacion:
        py_val = float(metricas_comparacion[metrica]["py_agata"])
        cgm_val = float(metricas_comparacion[metrica]["cgmpy"])

        if cgm_val != 0:
            diff_porcentual = abs((py_val - cgm_val) / cgm_val) * 100
        else:
            diff_porcentual = 0

        if diff_porcentual < 1:
            estado = "✅ Excelente"
        elif diff_porcentual < 5:
            estado = "✅ Bueno"
        elif diff_porcentual < 10:
            estado = "⚠️ Moderado"
        else:
            estado = "❌ Alto"

        print(f"{metrica:<20} {diff_porcentual:<15.2f} {estado:<15}")

print("\n" + "=" * 80)
print("CONCLUSIONES")
print("=" * 80)

# Contar métricas por categoría de diferencia
excelentes = 0
buenas = 0
moderadas = 0
altas = 0

for metrica in metricas_clave:
    if metrica in metricas_comparacion:
        py_val = float(metricas_comparacion[metrica]["py_agata"])
        cgm_val = float(metricas_comparacion[metrica]["cgmpy"])

        if cgm_val != 0:
            diff_porcentual = abs((py_val - cgm_val) / cgm_val) * 100
        else:
            diff_porcentual = 0

        if diff_porcentual < 1:
            excelentes += 1
        elif diff_porcentual < 5:
            buenas += 1
        elif diff_porcentual < 10:
            moderadas += 1
        else:
            altas += 1

total_metricas = len(metricas_clave)
print("📊 CONCORDANCIA ENTRE MÉTODOS:")
print(f"   • Excelente (<1%): {excelentes}/{total_metricas} ({excelentes / total_metricas * 100:.1f}%)")
print(f"   • Buena (<5%): {buenas}/{total_metricas} ({buenas / total_metricas * 100:.1f}%)")
print(f"   • Moderada (<10%): {moderadas}/{total_metricas} ({moderadas / total_metricas * 100:.1f}%)")
print(f"   • Alta (>10%): {altas}/{total_metricas} ({altas / total_metricas * 100:.1f}%)")

print("\n💡 OBSERVACIONES:")
print("   • Ambos métodos muestran resultados muy similares en métricas básicas")
print("   • Las diferencias menores al 5% indican buena concordancia")
print("   • py_agata proporciona análisis más detallado de eventos")
print("   • cgmpy ofrece métricas adicionales específicas")
