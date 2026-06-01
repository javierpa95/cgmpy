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

    # Expected delivery date (approx 9 months after data start)
    delivery_date = "2023-10-10"

    try:
        # 38 weeks + 0 days
        preg = GestationalDiabetes(data_source=df, delivery_date=delivery_date, week=38, day=0)
        preg.info()
        print("\n--- Object Summary ---")
        print(preg)

        print("\n--- Calculating all metrics ---")
        metrics = preg.calculate_all_metrics()

        # Check for key metrics presence
        for t in ["first_trimester", "second_trimester", "third_trimester"]:
            key = f"basic_metrics_{t}"
            if key in metrics:
                print(f"Metrics OK for {t}: GMI={metrics[key]['GMI']}")
            else:
                print(f"ERROR: Missing {key}")

        print("\nPrueba finalizada con éxito.")

    except Exception as e:
        print(f"\nERROR durante la prueba: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_pregnancy()
