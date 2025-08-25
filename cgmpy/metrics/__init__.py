"""
Módulo de métricas y estadísticas de glucosa.

Este módulo contiene las clases y funciones para calcular:
- Métricas básicas: media, mediana, percentiles, GMI
- Tiempo en rango: TIR, TAR, TBR
- Variabilidad: SD, CV, MAGE, MODD, CONGA
- Métricas avanzadas: GRADE, GRI, M-Value, J-Index
"""

import pandas as pd

# Importaciones que están disponibles ahora
from .basic import BasicMetrics
from .time_in_range import TimeInRangeMetrics
from .variability import VariabilityMetrics

# Importaciones que estarán disponibles cuando se implemente
# from .advanced import AdvancedMetrics

# Clase combinada que integra todas las métricas modulares
class ModularGlucoseMetrics(BasicMetrics, TimeInRangeMetrics, VariabilityMetrics):
    """
    Clase que combina todas las métricas modulares.
    
    Esta clase permite usar las métricas de forma modular mientras
    mantiene compatibilidad con la interfaz existente.
    """

    def all(self) -> dict:
        """
        Calcula todas las métricas disponibles de glucosa.
        
        Esta función integra todas las métricas de los tres módulos:
        - Métricas básicas (BasicMetrics)
        - Métricas de tiempo en rango (TimeInRangeMetrics) 
        - Métricas de variabilidad (VariabilityMetrics)
        
        Returns:
            dict: Diccionario completo con todas las métricas calculadas
        """
        try:
            # Inicializar diccionario de resultados
            all_metrics = {}
            
            # 1. MÉTRICAS BÁSICAS (BasicMetrics)
            try:
                basic_metrics = self.calculate_all_metrics()
                all_metrics['basic'] = basic_metrics
            except Exception as e:
                all_metrics['basic'] = {'error': f'Error en métricas básicas: {str(e)}'}
            
            # 2. MÉTRICAS DE TIEMPO EN RANGO (TimeInRangeMetrics)
            try:
                time_metrics = self.time_range_summary()
                all_metrics['time_in_range'] = time_metrics
            except Exception as e:
                all_metrics['time_in_range'] = {'error': f'Error en tiempo en rango: {str(e)}'}
            
            # 3. MÉTRICAS DE VARIABILIDAD (VariabilityMetrics)
            try:
                # Métricas de desviación estándar
                sd_metrics = {
                    'sd_total': self.sd_total(),
                    'sd_within_day': self.sd_within_day(),
                    'sd_between_timepoints': self.sd_between_timepoints(),
                    'sd_segments': {
                        'noche': self.sd_segment("00:00", 8),
                        'dia': self.sd_segment("08:00", 8),
                        'tarde': self.sd_segment("16:00", 8)
                    },
                    'sd_within_series': {
                        '1h': self.sd_within_series(hours=1),
                        '6h': self.sd_within_series(hours=6),
                        '24h': self.sd_within_series(hours=24)
                    },
                    'sd_daily_mean': self.sd_daily_mean(),
                    'sd_same_timepoint': self.sd_same_timepoint(),
                    'sd_same_timepoint_adjusted': self.sd_same_timepoint_adjusted(),
                    'sd_interaction': self.sd_interaction()
                }
                
                # Métricas de coeficiente de variación
                cv_metrics = self.calculate_all_cv_metrics()
                
                # Métricas de excursiones
                try:
                    mage_metrics = self.MAGE_Baghurst()
                    excursion_metrics = {
                        'mage_baghurst': mage_metrics,
                        'mage_simple': self.MAGE()
                    }
                except Exception as e:
                    excursion_metrics = {'error': f'Error en MAGE: {str(e)}'}
                
                # Métricas de variabilidad inter e intradiaria
                variability_metrics = {
                    'modd': self.MODD(),
                    'conga': {
                        '1h': self.CONGA(hours=1),
                        '2h': self.CONGA(hours=2),
                        '4h': self.CONGA(hours=4),
                        '6h': self.CONGA(hours=6),
                        '24h': self.CONGA(hours=24)
                    },
                    'lability_index': self.Lability_index()
                }
                
                # Métricas de calidad de glucosa
                quality_metrics = {
                    'm_value': self.M_Value(),
                    'j_index': self.j_index(),
                    'grade': self.GRADE(),
                    'lbgi': self.LBGI(),
                    'hbgi': self.HBGI(),
                    'gri': self.GRI(),
                    'gri_pregnancy': self.GRI(pregnancy=True),
                    'adrr': self.ADRR()
                }
                
                all_metrics['variability'] = {
                    'sd_metrics': sd_metrics,
                    'cv_metrics': cv_metrics,
                    'excursion_metrics': excursion_metrics,
                    'variability_metrics': variability_metrics,
                    'quality_metrics': quality_metrics
                }
                
            except Exception as e:
                all_metrics['variability'] = {'error': f'Error en métricas de variabilidad: {str(e)}'}
            
            # 4. RESUMEN GENERAL
            try:
                summary = {
                    'total_metrics': len(all_metrics),
                    'modules': list(all_metrics.keys()),
                    'calculation_timestamp': pd.Timestamp.now().isoformat(),
                    'data_summary': {
                        'total_readings': len(self.data),
                        'date_range': {
                            'start': self.data['time'].min().isoformat(),
                            'end': self.data['time'].min().isoformat()
                        },
                        'data_completeness': self.data_completeness()
                    }
                }
                all_metrics['summary'] = summary
                
            except Exception as e:
                all_metrics['summary'] = {'error': f'Error en resumen: {str(e)}'}
            
            return all_metrics
            
        except Exception as e:
            return {
                'error': f'Error general en cálculo de métricas: {str(e)}',
                'type': 'general_error'
            }
    
    def all_simplified(self) -> dict:
        """
        Versión simplificada de all() que devuelve solo los valores principales.
        
        Returns:
            dict: Diccionario con métricas principales en formato plano
        """
        try:
            # Obtener todas las métricas
            full_metrics = self.all()
            
            # Extraer solo los valores principales
            simplified = {}
            
            # Métricas básicas principales
            if 'basic' in full_metrics and 'error' not in full_metrics['basic']:
                simplified.update({
                    'GMI': full_metrics['basic'].get('GMI'),
                    'Media': full_metrics['basic'].get('Media'),
                    'Mediana': full_metrics['basic'].get('Mediana'),
                    'SD': full_metrics['basic'].get('Desviacion_estandar'),
                    'CV': full_metrics['basic'].get('CV')
                })
            
            # Tiempo en rango principal
            if 'time_in_range' in full_metrics and 'error' not in full_metrics['time_in_range']:
                simplified.update({
                    'TIR': full_metrics['time_in_range'].get('standard_ranges', {}).get('TIR'),
                    'TAR180': full_metrics['time_in_range'].get('standard_ranges', {}).get('TAR180'),
                    'TAR250': full_metrics['time_in_range'].get('standard_ranges', {}).get('TAR250'),
                    'TBR70': full_metrics['time_in_range'].get('standard_ranges', {}).get('TBR70'),
                    'TBR55': full_metrics['time_in_range'].get('standard_ranges', {}).get('TBR55')
                })
            
            # Variabilidad principal
            if 'variability' in full_metrics and 'error' not in full_metrics['variability']:
                sd_metrics = full_metrics['variability'].get('sd_metrics', {})
                simplified.update({
                    'SDw': sd_metrics.get('sd_within_day', {}).get('sd'),
                    'SDdm': sd_metrics.get('sd_daily_mean', {}).get('sd'),
                    'MAGE': full_metrics['variability'].get('excursion_metrics', {}).get('mage_baghurst', {}).get('MAGE_avg'),
                    'CONGA4': full_metrics['variability'].get('variability_metrics', {}).get('conga', {}).get('4h', {}).get('value'),
                    'LBGI': full_metrics['variability'].get('quality_metrics', {}).get('lbgi'),
                    'HBGI': full_metrics['variability'].get('quality_metrics', {}).get('hbgi'),
                    'GRI': full_metrics['variability'].get('quality_metrics', {}).get('gri', {}).get('GRI')
                })
            
            return simplified
            
        except Exception as e:
            return {'error': f'Error en métricas simplificadas: {str(e)}'}

    pass

__all__ = [
    'BasicMetrics', 
    'TimeInRangeMetrics',
    'VariabilityMetrics',
    'ModularGlucoseMetrics'
] 