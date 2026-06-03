# cgmpy/__init__.py

# Modular (internal) imports
from . import agata, analysis, data, metrics, plotting, utils

# Agata wrapper
from .agata.metrics import AgataAnalysis
from .analysis.core import GlucoseAnalysis
from .data.analyzer import DataAnalyzer

# Specific imports from refactored modules
from .data.core import GlucoseData
from .data.exporter import DataExporter
from .data.loader import DataLoader
from .data.pregnancy_data import PregnancyData, PregnancyDataHandler
from .data.processor import DataProcessor
from .data.specialized import Dexcom, Libreview, MedtronicCarelink, TandemDiabetes
from .metrics.units import GlucoseUnit
from .metrics.pregnancy import PregnancyAnalysis
from .utils.date_utils import parse_date, validate_date_range


__all__ = [
    # Main classes
    "AgataAnalysis",
    "DataAnalyzer",
    "DataExporter",
    "DataLoader",
    "DataProcessor",
    "Dexcom",
    "GlucoseAnalysis",
    "GlucoseData",
    "GlucoseUnit",
    "Libreview",
    "MedtronicCarelink",
    "PregnancyAnalysis",
    "PregnancyData",
    "PregnancyDataHandler",
    "TandemDiabetes",
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
__version__ = "0.5.2"
__author__ = "Javier Peñate Arrieta"
__description__ = "Modular package for continuous glucose monitoring (CGM) data analysis"
