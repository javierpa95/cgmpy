# cgmpy/__init__.py

# Importaciones modulares (internas)
from . import agata, analysis, data, metrics, plotting, utils

# Wrapper para Agata
from .agata.metrics import AgataAnalysis
from .analysis.core import GlucoseAnalysis
from .data.analyzer import DataAnalyzer

# Importaciones específicas de módulos refactorizados
from .data.core import ModularGlucoseData
from .data.exporter import DataExporter
from .data.loader import DataLoader
from .data.pregnancy_data import PregnancyData, PregnancyDataHandler
from .data.processor import DataProcessor
from .data.specialized import Dexcom, Libreview
from .metrics import ModularGlucoseMetrics
from .metrics.basic import BasicMetrics
from .metrics.pregnancy import GestationalDiabetes
from .metrics.time_in_range import TimeInRangeMetrics
from .metrics.variability import VariabilityMetrics
from .plotting.agp import AGPPlotter
from .plotting.daily_plots import DailyPlotter
from .plotting.statistical_plots import StatisticalPlotter
from .utils.date_utils import parse_date, validate_date_range


# Clases principales con nombres tradicionales (estándar)
class GlucoseData(ModularGlucoseData):
    """
    Clase principal para manejo de datos de glucosa.

    Esta clase proporciona funcionalidad completa para:
    - Carga de datos desde CSV, Parquet o DataFrame
    - Procesamiento y validación de datos
    - Análisis básico de información
    - Exportación a diferentes formatos
    """

    pass


class GlucoseMetrics(ModularGlucoseData, ModularGlucoseMetrics):
    """
    Clase principal para métricas de glucosa.

    Esta clase combina:
    - Manejo de datos (GlucoseData)
    - Métricas básicas (media, mediana, GMI, etc.)
    - Métricas de tiempo en rango (TIR, TAR, TBR)
    - Métricas de variabilidad (MAGE, MODD, CONGA, etc.)
    """


class GlucosePlot(ModularGlucoseData, AGPPlotter, DailyPlotter, StatisticalPlotter):
    """
    Clase principal para visualización de glucosa.

    Esta clase combina:
    - Manejo de datos (GlucoseData)
    - Gráficos de perfil ambulatorio (AGP)
    - Gráficos diarios y variaciones
    - Gráficos estadísticos y análisis
    """

    pass


__all__ = [
    # Clases principales (nombres tradicionales)
    "AGPPlotter",
    "AgataAnalysis",
    "BasicMetrics",
    "DailyPlotter",
    "DataAnalyzer",
    "DataExporter",
    "DataLoader",
    "DataProcessor",
    "Dexcom",
    "GestationalDiabetes",
    "GlucoseAnalysis",
    "GlucoseData",
    "GlucoseMetrics",
    "GlucosePlot",
    "Libreview",
    "ModularGlucoseData",
    "PregnancyData",
    "PregnancyDataHandler",
    "StatisticalPlotter",
    "TimeInRangeMetrics",
    "VariabilityMetrics",
    # Utilidades
    "agata",
    "analysis",
    "data",
    "metrics",
    "parse_date",
    "plotting",
    "utils",
    "validate_date_range",
]


# Información del paquete
__version__ = "0.5.0"
__author__ = "Javier Peñate Arrieta"
__description__ = "Paquete modular para análisis de datos de glucosa de monitoreo continuo (CGM)"
