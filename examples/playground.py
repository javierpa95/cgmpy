import pandas as pd

from cgmpy import AgataAnalysis, GlucoseMetrics

# --- IMPORTANTE: Cambia esta ruta al de tu archivo CSV ---
archivo = r"C:\Users\javie\Desktop\PROGRAMACIÓN\cgmpy\examples\data\dm.csv"

print("Calculando métricas con AgataAnalysis y GlucoseMetrics...")

# 1. Cargar y ejecutar el análisis de Agata
try:
    agata_analyzer = AgataAnalysis(data_source=archivo)
    agata_results = agata_analyzer.run()
except Exception as e:
    print(f"Error al ejecutar AgataAnalysis: {e}")
    agata_results = {}

# 2. Cargar y ejecutar el análisis de GlucoseMetrics
try:
    cgm_analyzer = GlucoseMetrics(data_source=archivo)
    glucose_metrics_results = cgm_analyzer.all()
except Exception as e:
    print(f"Error al ejecutar GlucoseMetrics: {e}")
    glucose_metrics_results = {}

# 3. Mapeo de métricas entre las dos librerías
# Aquí definimos qué métrica en Agata se corresponde con cuál en GlucoseMetrics.
# La ruta es una tupla de claves para navegar por los diccionarios anidados.
metric_map = [
    # --- Variabilidad ---
    {
        "name": "Media Glucosa (mg/dL)",
        "agata_path": ("variability", "mean_glucose"),
        "gm_path": ("basic", "media"),
    },
    {
        "name": "Mediana Glucosa (mg/dL)",
        "agata_path": ("variability", "median_glucose"),
        "gm_path": ("basic", "mediana"),
    },
    {
        "name": "Desviación Estándar (mg/dL)",
        "agata_path": ("variability", "std_glucose"),
        "gm_path": ("basic", "desviacion_estandar"),
    },
    {
        "name": "Coeficiente de Variación (%)",
        "agata_path": ("variability", "cv_glucose"),
        "gm_path": ("basic", "cv"),
    },
    {
        "name": "GMI (%)",
        "agata_path": ("variability", "gmi"),
        "gm_path": ("basic", "gmi"),
    },
    {
        "name": "MODD",
        "agata_path": ("variability", "modd"),
        "gm_path": ("basic", "modd"),
    },
    {
        "name": "J-Index",
        "agata_path": ("variability", "j_index"),
        "gm_path": ("basic", "j_index"),
    },
    {
        "name": "MAGE+",
        "agata_path": ("variability", "mage_plus_index"),
        "gm_path": ("basic", "mage_plus"),
    },
    {
        "name": "MAGE-",
        "agata_path": ("variability", "mage_minus_index"),
        "gm_path": ("basic", "mage_minus"),
    },
    # --- Tiempo en Rangos (%) ---
    {
        "name": "Tiempo en Rango (TIR)",
        "agata_path": ("time_in_ranges", "time_in_target"),
        "gm_path": ("basic", "tir"),
    },
    {
        "name": "Tiempo en Rango Estrecho",
        "agata_path": ("time_in_ranges", "time_in_tight_target"),
        "gm_path": ("basic", "tir_tight"),
    },
    {
        "name": "Tiempo Sobre Rango (TAR >180)",
        "agata_path": ("time_in_ranges", "time_in_l1_hyperglycemia"),
        "gm_path": ("basic", "tar180"),
    },
    {
        "name": "Tiempo Sobre Rango (TAR >250)",
        "agata_path": ("time_in_ranges", "time_in_l2_hyperglycemia"),
        "gm_path": ("basic", "tar250"),
    },
    {
        "name": "Tiempo Bajo Rango (TBR <70)",
        "agata_path": ("time_in_ranges", "time_in_l1_hypoglycemia"),
        "gm_path": ("basic", "tbr70"),
    },
    {
        "name": "Tiempo Bajo Rango (TBR <54)",
        "agata_path": ("time_in_ranges", "time_in_l2_hypoglycemia"),
        "gm_path": ("basic", "tbr55"),
    },
    # --- Riesgo ---
    {"name": "LBGI", "agata_path": ("risk", "lbgi"), "gm_path": ("basic", "lbgi")},
    {"name": "HBGI", "agata_path": ("risk", "hbgi"), "gm_path": ("basic", "hbgi")},
    {"name": "ADRR", "agata_path": ("risk", "adrr"), "gm_path": ("basic", "adrr")},
    {"name": "GRI", "agata_path": ("risk", "gri"), "gm_path": ("basic", "gri")},
]


def get_nested_value(data_dict, path):
    """
    Función auxiliar para obtener un valor de un diccionario anidado de forma segura.
    """
    current_level = data_dict
    for key in path:
        if isinstance(current_level, dict) and key in current_level:
            current_level = current_level[key]
        else:
            return "No encontrado"
    return current_level


# 4. Construir la lista de datos para la comparación
comparison_data = []
for metric in metric_map:
    agata_val = get_nested_value(agata_results, metric["agata_path"])
    gm_val = get_nested_value(glucose_metrics_results, metric["gm_path"])

    # Formatear números a 2 decimales para una comparación más fácil
    if isinstance(agata_val, (int, float)):
        agata_val = f"{agata_val:.2f}"
    if isinstance(gm_val, (int, float)):
        gm_val = f"{gm_val:.2f}"

    comparison_data.append(
        {
            "Métrica": metric["name"],
            "Valor AgataAnalysis": agata_val,
            "Valor GlucoseMetrics": gm_val,
        }
    )

# 5. Crear y mostrar el DataFrame de comparación
df_comparison = pd.DataFrame(comparison_data)

print("\n----- Tabla Comparativa de Métricas -----")
# Usamos to_string() para asegurar que se muestren todas las filas y columnas
print(df_comparison.to_string())
