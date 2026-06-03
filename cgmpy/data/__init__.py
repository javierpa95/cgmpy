"""
Refactored data module for cgmpy.

This module provides a modular architecture for glucose data handling:
- DataLoader: Data loading from different sources
- DataProcessor: Data processing and validation
- DataAnalyzer: Basic data analysis
- DataExporter: Export to different formats
- GlucoseData: Main class integrating all modules

Specialized classes by device:
- Dexcom: For Dexcom Clarity data
- Libreview: For Libreview data
- MedtronicCarelink: For Medtronic CareLink data
- TandemDiabetes: For Tandem Diabetes data
"""

# Import main classes
from .analyzer import DataAnalyzer
from .core import GlucoseData
from .exporter import DataExporter
from .loader import DataLoader
from .pregnancy_data import PregnancyData, PregnancyDataHandler
from .processor import DataProcessor

# Import specialized classes
from .specialized import (
    Dexcom,
    Libreview,
    MedtronicCarelink,
    TandemDiabetes,
    create_specialized_loader,
    detect_device_type,
)

__all__ = [
    "DataAnalyzer",
    "DataExporter",
    "DataLoader",
    "DataProcessor",
    "GlucoseData",
    "Dexcom",
    "Libreview",
    "MedtronicCarelink",
    "PregnancyData",
    "PregnancyDataHandler",
    "TandemDiabetes",
    "create_specialized_loader",
    "detect_device_type",
]
