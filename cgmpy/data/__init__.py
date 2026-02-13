"""
Módulo de datos refactorizado para cgmpy.

Este módulo proporciona una arquitectura modular para el manejo de datos de glucosa:
- DataLoader: Carga de datos desde diferentes fuentes
- DataProcessor: Procesamiento y validación de datos
- DataAnalyzer: Análisis básico de datos
- DataExporter: Exportación a diferentes formatos
- ModularGlucoseData: Clase principal que integra todos los módulos

Clases especializadas por dispositivo:
- Dexcom: Para datos de Dexcom Clarity
- Libreview: Para datos de Libreview
- MedtronicCarelink: Para datos de Medtronic CareLink
- TandemDiabetes: Para datos de Tandem Diabetes
"""

# Importar clases principales
from .analyzer import DataAnalyzer
from .core import ModularGlucoseData
from .exporter import DataExporter
from .loader import DataLoader
from .pregnancy_data import PregnancyData, PregnancyDataHandler
from .processor import DataProcessor

# Importar clases especializadas
from .specialized import (
    Dexcom,
    Libreview,
    MedtronicCarelink,
    TandemDiabetes,
    create_specialized_loader,
    detect_device_type,
)

# Mantener compatibilidad hacia atrás
# Los usuarios pueden seguir usando: from cgmpy import GlucoseData
GlucoseData = ModularGlucoseData

__all__ = [
    # Clase principal
    "ModularGlucoseData",
    "GlucoseData",  # Alias para compatibilidad
    # Módulos especializados
    "DataLoader",
    "DataProcessor",
    "DataAnalyzer",
    "DataExporter",
    # Clases especializadas por dispositivo
    "Dexcom",
    "Libreview",
    "MedtronicCarelink",
    "TandemDiabetes",
    # Utilidades
    "detect_device_type",
    "create_specialized_loader",
    "PregnancyData",
    "PregnancyDataHandler",
]
