# cgmpy/__init__.py

from importlib.metadata import PackageNotFoundError, version as _pkg_version

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
from .metrics.pregnancy import PregnancyAnalysis
from .metrics.units import GlucoseUnit
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


# Package info. The version is sourced from the installed package metadata so
# that ``pyproject.toml`` remains the single source of truth (release-please
# bumps it there). Falls back gracefully when running from an uninstalled tree.
try:
    __version__ = _pkg_version("cgmpy")
except PackageNotFoundError:  # pragma: no cover - only when not installed
    __version__ = "0.0.0+unknown"

__author__ = "Javier Peñate Arrieta"
__description__ = "Modular package for continuous glucose monitoring (CGM) data analysis"
