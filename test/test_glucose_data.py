import unittest
import pandas as pd
import numpy as np
from cgmpy.glucose_data import GlucoseData

class TestGlucoseData(unittest.TestCase):
    def setUp(self):
        # Crear un DataFrame de prueba
        dates = pd.date_range(start='2024-01-01', periods=48, freq='30min')
        glucose_values = np.random.normal(140, 30, 48)  # Media 140, SD 30
        self.test_df = pd.DataFrame({
            'DateTime': dates,
            'Glucose': glucose_values
        })
        
        # Guardar el DataFrame como CSV temporal
        self.test_file = 'test_glucose.csv'
        self.test_df.to_csv(self.test_file, index=False)
        
        # Inicializar GlucoseData
        self.glucose_data = GlucoseData(
            file_path=self.test_file,
            date_col='DateTime',
            glucose_col='Glucose'
        )

    def test_basic_stats(self):
        """Prueba las estadísticas básicas"""
        # Verificar que los métodos estadísticos básicos funcionan
        self.assertIsInstance(self.glucose_data.mean(), float)
        self.assertIsInstance(self.glucose_data.median(), float)
        self.assertIsInstance(self.glucose_data.sd(), float)
        self.assertIsInstance(self.glucose_data.gmi(), float)

    def test_time_in_ranges(self):
        """Prueba los cálculos de tiempo en rango"""
        # Verificar que los porcentajes están entre 0 y 100
        tir = self.glucose_data.TIR()
        self.assertGreaterEqual(tir, 0)
        self.assertLessEqual(tir, 100)

        tar = self.glucose_data.TAR180()
        self.assertGreaterEqual(tar, 0)
        self.assertLessEqual(tar, 100)

        tbr = self.glucose_data.TBR70()
        self.assertGreaterEqual(tbr, 0)
        self.assertLessEqual(tbr, 100)

    def test_variability_metrics(self):
        """Prueba las métricas de variabilidad"""
        # Verificar que las métricas de variabilidad devuelven valores numéricos
        self.assertIsInstance(self.glucose_data.CONGA(), float)
        self.assertIsInstance(self.glucose_data.MODD(), float)
        self.assertIsInstance(self.glucose_data.j_index(), float)
        self.assertIsInstance(self.glucose_data.LBGI(), float)
        self.assertIsInstance(self.glucose_data.HBGI(), float)

    def test_info(self):
        """Prueba el método info"""
        info_str = self.glucose_data.info()
        self.assertIsInstance(info_str, str)
        self.assertIn("datos entre", info_str)

    def tearDown(self):
        """Limpieza después de las pruebas"""
        import os
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

if __name__ == '__main__':
    unittest.main() 