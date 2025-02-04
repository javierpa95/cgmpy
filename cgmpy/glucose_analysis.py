import datetime
from .glucose_metrics import GlucoseMetrics
from .glucose_plot import GlucosePlot
from typing import Union
import pandas as pd

class GlucoseAnalysis(GlucoseMetrics, GlucosePlot):
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
                 end_date: Union[str, datetime.datetime, None] = None):
        
        GlucoseMetrics.__init__(self, data_source, date_col, glucose_col, delimiter, header, start_date, end_date) 




