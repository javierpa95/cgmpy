# cgmpy/__init__.py

# Modular (internal) imports
from . import agata, analysis, data, metrics, plotting, utils

# Agata wrapper
from .agata.metrics import AgataAnalysis
from .analysis.core import GlucoseAnalysis
from .data.analyzer import DataAnalyzer

# Specific imports from refactored modules
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


# Main classes with traditional (standard) names
class GlucoseData(ModularGlucoseData):
    """
    Main class for handling glucose data.

    This class provides complete functionality for:
    - Loading data from CSV, Parquet or DataFrame
    - Data processing and validation
    - Basic information analysis
    - Exporting to different formats
    """

    pass


class GlucoseMetrics(ModularGlucoseData, ModularGlucoseMetrics):
    """
    Main class for glucose metrics.

    This class combines:
    - Data handling (GlucoseData)
    - Basic metrics (mean, median, GMI, etc.)
    - Time in range metrics (TIR, TAR, TBR)
    - Variability metrics (MAGE, MODD, CONGA, etc.)
    """


class GlucosePlot(
    ModularGlucoseData,
    BasicMetrics,
    TimeInRangeMetrics,
    AGPPlotter,
    DailyPlotter,
    StatisticalPlotter,
):
    """
    Main class for glucose visualization.

    This class combines:
    - Data handling (GlucoseData)
    - Basic metrics (mean, GMI, CV, ...)
    - Time in range metrics (TIR, TAR, TBR)
    - Ambulatory glucose profile (AGP) plots
    - Daily plots and variations
    - Statistical plots and analysis
    """

    pass


__all__ = [
    # Main classes (traditional names)
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
    # Utilities
    "agata",
    "analysis",
    "data",
    "metrics",
    "parse_date",
    "plotting",
    "utils",
    "validate_date_range",
]


# Package info
__version__ = "0.5.1"
__author__ = "Javier Peñate Arrieta"
__description__ = "Modular package for continuous glucose monitoring (CGM) data analysis"
