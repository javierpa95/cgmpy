import unittest
import pandas as pd

from ..cgmpy import GlucoseData



class TestGlucoseData(unittest.TestCase):
    def setUp(self):
        # Crear un DataFrame de prueba
        data = {
            'time': ['2023-01-01 00:00', '2023-01-01 00:05', '2023-01-01 00:10'],
            'glucose': [100, 110, 105]
        }
        self.df = pd.DataFrame(data)
        self.df['time'] = pd.to_datetime(self.df['time'])
        
        # Guardar el DataFrame como CSV
        self.df.to_csv('test_data.csv', index=False)
        
        # Crear una instancia de GlucoseData
        self.gd = GlucoseData('test_data.csv', 'time', 'glucose')

    def test_mean(self):
        self.assertAlmostEqual(self.gd.mean(), 105, places=2)

    def test_median(self):
        self.assertEqual(self.gd.median(), 105)

    def test_sd(self):
        self.assertAlmostEqual(self.gd.sd(), 5, places=2)

    def test_gmi(self):
        expected_gmi = 3.31 + (0.02392 * 105)
        self.assertAlmostEqual(self.gd.gmi(), expected_gmi, places=2)

    def tearDown(self):
        # Eliminar el archivo CSV de prueba
        import os
        os.remove('test_data.csv')

if __name__ == '__main__':
    unittest.main()
