import pandas as pd

from cgmpy import GlucoseData

# --- IMPORTANTE: Cambia esta ruta al de tu archivo CSV ---
archivo = r"C:\Users\javie\Desktop\PROGRAMACIÓN\cgmpy\examples\data\dm.csv"


datos =GlucoseData(archivo)

print(datos)