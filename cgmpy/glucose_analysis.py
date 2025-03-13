import datetime
from .glucose_metrics import GlucoseMetrics
from .glucose_plot import GlucosePlot
from typing import Union
import pandas as pd

class GlucoseAnalysis(GlucosePlot):
    """
    Clase que combina todas las funcionalidades de análisis de glucosa:
    - Manejo de datos (GlucoseData)
    - Métricas y estadísticas (GlucoseMetrics)
    - Visualización (GlucosePlot)
    """
    def __init__(self, 
                 data_source: Union[str, pd.DataFrame], 
                 date_col: str="time", 
                 glucose_col: str="glucose", 
                 delimiter: Union[str, None] = None, 
                 header: int = 0, 
                 start_date: Union[str, datetime.datetime, None] = None,
                 end_date: Union[str, datetime.datetime, None] = None,
                 log: bool = False):
        
        # Verificar si GlucoseData ya ha sido inicializado
        if not hasattr(self, 'data'):
            super().__init__(data_source, date_col, glucose_col, delimiter, header, 
                           start_date, end_date, log)




