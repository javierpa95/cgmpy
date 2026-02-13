import datetime

import numpy as np
import pandas as pd

from cgmpy.metrics.pregnancy import GestationalDiabetes


def generate_mock_data():
    # Generar 9 meses de datos (5 min cada uno)
    start_date = datetime.datetime(2023, 1, 1)
    end_date = start_date + datetime.timedelta(days=280)
    dates = pd.date_range(start=start_date, end=end_date, freq="5min")
    glucose = np.random.normal(100, 20, len(dates))
    return pd.DataFrame({"time": dates, "glucose": glucose})


def test_pregnancy():
    print("Iniciando prueba de GestationalDiabetes...")
    df = generate_mock_data()

    # Fecha de parto esperada (aprox 9 meses después de empezar los datos)
    fecha_parto = "2023-10-10"

    try:
        # 38 semanas + 0 días
        preg = GestationalDiabetes(data_source=df, fecha_parto=fecha_parto, week=38, day=0)
        preg.info()
        print("\n--- Resumen del objeto ---")
        print(preg)

        print("\n--- Calculando todas las métricas ---")
        metrics = preg.calculate_all_metrics()

        # Verificar presencia de métricas clave
        for t in ["primer_trimestre", "segundo_trimestre", "tercer_trimestre"]:
            key = f"metricas_basicas_{t}"
            if key in metrics:
                print(f"Métricas OK para {t}: GMI={metrics[key]['GMI']}")
            else:
                print(f"ERROR: Falta {key}")

        print("\nPrueba finalizada con éxito.")

    except Exception as e:
        print(f"\nERROR durante la prueba: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_pregnancy()
