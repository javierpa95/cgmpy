# cgmpy/agata/__init__.py

"""
Subpaquete de integración con la librería py_agata.

Este subpaquete proporciona las herramientas necesarias para preparar y analizar
datos de cgmpy utilizando la librería externa py_agata, actuando como un puente
entre ambas.

Funciones Públicas:
- prepare_data_for_agata: Adapta un objeto ModularGlucoseData al formato requerido por py_agata.
- analyze_with_agata: Ejecuta el pipeline de análisis completo usando py_agata.
"""

# Importar las funciones que quieres exponer públicamente desde este subpaquete
from .adapter import prepare_data_for_agata
from .metrics import AgataAnalysis, analyze_one_arm, analyze_with_agata

# Definir qué se importa cuando un usuario ejecuta 'from cgmpy.agata import *'
# Es una buena práctica para definir la API pública del subpaquete.
__all__ = ["AgataAnalysis", "analyze_one_arm", "analyze_with_agata", "prepare_data_for_agata"]
