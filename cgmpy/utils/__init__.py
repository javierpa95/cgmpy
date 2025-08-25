"""
Módulo de utilidades para el manejo de datos de glucosa.

Este módulo contiene funciones y clases auxiliares para:
- Utilidades de fecha y hora
- Validadores de datos médicos
- Configuración centralizada
- Funciones de ayuda general
"""

# Importaciones que estarán disponibles cuando se implemente
# from .date_utils import parse_date
# from .validators import DataValidator
# from .config import GlucoseConfig

# Por ahora, importamos desde la ubicación actual para mantener compatibilidad
# Evitamos import circular usando import diferido
def parse_date(*args, **kwargs):
    from ..utils import parse_date as _parse_date
    return _parse_date(*args, **kwargs)

__all__ = ['parse_date'] 