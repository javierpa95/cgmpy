import os

import pandas as pd

from cgmpy import AgataAnalysis

file = r"C:\Users\javie\Desktop\PROGRAMACIÓN\cgmpy\examples\data\nodm.csv"

Agata = AgataAnalysis(file, glycemic_target="diabetes")

results = Agata.run(summary=True)

# 1. Mostrar por consola (opcional)
for var, val in results.items():
    print(f"{var}: {round(val, 2)}")

# 2. Guardar/Añadir a CSV
df = pd.DataFrame([results])
# Insertamos una columna al inicio para identificar al paciente (usamos el nombre del archivo)
df.insert(0, "patient_id", os.path.basename(file))

csv_path = "agata_results.csv"
# Si el archivo no existe, escribimos cabecera; si existe, solo añadimos la fila
df.to_csv(csv_path, mode="a", index=False, header=not os.path.exists(csv_path))

print(f"\n[OK] Resultados añadidos a: {csv_path}")
