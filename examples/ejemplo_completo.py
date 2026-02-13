from cgmpy import GlucoseAnalysis

# --- IMPORTANTE: Cambia esta ruta al de tu archivo CSV ---
archivo = r"C:\Users\javie\Desktop\PROGRAMACIÓN\cgmpy\examples\data\dm.csv"

# 1. Creamos el objeto de análisis completo
analysis = GlucoseAnalysis(archivo)

# 2. Imprimimos el resumen de texto (que es muy completo)
print(analysis.get_summary_string())

# 3. Obtenemos el reporte en formato diccionario para acceder a datos específicos
reporte = analysis.get_comprehensive_report()
print(f"\nVariabilidad (MAGE): {reporte['metricas_variabilidad']['MAGE']:.2f}")
