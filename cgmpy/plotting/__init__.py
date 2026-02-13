"""
Módulo de visualización de datos de glucosa.

Este módulo contiene las clases y funciones para:
- Perfiles ambulatorios de glucosa (AGP)
- Gráficos diarios y de superposición
- Gráficos estadísticos: histogramas, boxplots
- Dashboards y visualizaciones combinadas
"""

# Importaciones disponibles
from .agp import AGPPlotter
from .daily_plots import DailyPlotter
from .statistical_plots import StatisticalPlotter

# Importaciones que estarán disponibles cuando se implemente
# from .dashboard import DashboardPlotter


# Clase combinada que integra todos los plotters modulares
class ModularGlucosePlot(AGPPlotter, DailyPlotter, StatisticalPlotter):
    """
    Clase que combina todos los plotters modulares.

    Esta clase permite usar los gráficos de forma modular mientras
    mantiene compatibilidad con la interfaz existente.
    """

    pass


__all__ = ["AGPPlotter", "DailyPlotter", "StatisticalPlotter", "ModularGlucosePlot"]
