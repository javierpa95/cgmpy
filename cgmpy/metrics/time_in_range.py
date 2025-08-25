"""
Módulo de métricas de tiempo en rango para datos de glucosa.

Este módulo contiene las métricas relacionadas con el tiempo en diferentes rangos:
- Time in Range (TIR)
- Time Above Range (TAR) 
- Time Below Range (TBR)
- Estadísticas de tiempo específicas
"""

from typing import Union, Dict, Any


class TimeInRangeMetrics:
    """
    Clase para métricas de tiempo en rango de glucosa.
    
    Esta clase debe ser utilizada como mixin con GlucoseData.
    """
    
    def _calculate_data_completeness(self, interval_minutes: Union[float, None] = None) -> dict:
        """
        Calcula el porcentaje de datos disponibles para el DataFrame actual.
        
        Args:
            interval_minutes: Intervalo esperado entre mediciones en minutos.
                             Si es None, se calcula automáticamente.
        
        Returns:
            dict: Información sobre la completitud de datos
        """
        # Si no se especifica el intervalo, calcularlo como la mediana de las diferencias
        if interval_minutes is None:
            interval_minutes = self.typical_interval

        # Crear una copia de los datos y ordenarlos
        data = self.data.sort_values('time').copy()
        
        # Análisis para todo el período
        tiempo_total = (data['time'].max() - data['time'].min()).total_seconds() / 60
        datos_esperados = int(tiempo_total / interval_minutes)
        datos_reales = len(data)
        
        return {
            'inicio': data['time'].min(),
            'fin': data['time'].max(),
            'intervalo': interval_minutes,
            'datos_esperados': datos_esperados,
            'datos_reales': datos_reales,
            'porcentaje': (datos_reales / datos_esperados) * 100 if datos_esperados > 0 else 0
        }
    
    def data_completeness(self, interval_minutes: Union[float, None] = None) -> int:
        """
        Devuelve el porcentaje de datos disponibles.
        
        Args:
            interval_minutes: Intervalo esperado entre mediciones
            
        Returns:
            int: Porcentaje de completitud de datos
        """
        return int(self._calculate_data_completeness(interval_minutes)['porcentaje'])
    
    def calculate_time_in_range(self, low_threshold: float, high_threshold: float) -> float:
        """
        Calcula el tiempo en rango (TIR) de glucemia.
        
        Args:
            low_threshold: Umbral inferior del rango
            high_threshold: Umbral superior del rango
            
        Returns:
            float: Porcentaje de tiempo en rango
        """
        in_range = self.data[(self.data['glucose'] >= low_threshold) & 
                           (self.data['glucose'] <= high_threshold)]
        return (len(in_range) / len(self.data)) * 100
    
    def TAR(self, threshold: float) -> float:
        """
        Calcula el tiempo por encima del rango (TAR).
        
        Args:
            threshold: Umbral de hiperglucemia
            
        Returns:
            float: Porcentaje de lecturas por encima del umbral
        """
        return (len(self.data[self.data['glucose'] > threshold]) / len(self.data)) * 100
    
    def TBR(self, threshold: float) -> float:
        """
        Calcula el tiempo por debajo del rango (TBR).
        
        Args:
            threshold: Umbral de hipoglucemia
            
        Returns:
            float: Porcentaje de lecturas por debajo del umbral
        """
        return (len(self.data[self.data['glucose'] < threshold]) / len(self.data)) * 100

    # Métricas específicas de tiempo en rango
    def TAR250(self) -> float:
        """
        Calcula el tiempo por encima de 250 mg/dL.
        
        Returns:
            float: Porcentaje de tiempo > 250 mg/dL
        """
        return self.TAR(250)
    
    def TAR180(self) -> float:
        """
        Calcula el tiempo en rango entre 180 y 250 mg/dL.
        
        Returns:
            float: Porcentaje de tiempo entre 180-250 mg/dL
        """
        return self.calculate_time_in_range(181, 250)
    
    def TAR140(self) -> float:
        """
        Calcula el tiempo por encima de 140 mg/dL.
        
        Returns:
            float: Porcentaje de tiempo entre 140-250 mg/dL
        """
        return self.calculate_time_in_range(141, 250)
    
    def TIR(self) -> float:
        """
        Calcula el tiempo en rango entre 70 y 180 mg/dL.
        
        Returns:
            float: Porcentaje de tiempo en rango objetivo estándar
        """
        return self.calculate_time_in_range(70, 180)
    
    def TIR_tight(self) -> float:
        """
        Calcula el tiempo en rango estricto entre 70 y 140 mg/dL.
        
        Returns:
            float: Porcentaje de tiempo en rango estricto
        """
        return self.calculate_time_in_range(70, 140)
    
    def TIR_pregnancy(self) -> float:
        """
        Calcula el tiempo en rango para embarazo entre 63 y 140 mg/dL.
        
        Returns:
            float: Porcentaje de tiempo en rango para embarazo
        """
        return self.calculate_time_in_range(63, 140)
    
    def TBR70(self) -> float:
        """
        Calcula el tiempo en rango entre 55 y 70 mg/dL.
        
        Returns:
            float: Porcentaje de tiempo en hipoglucemia leve
        """
        return self.calculate_time_in_range(55, 69)
    
    def TBR63(self) -> float:
        """
        Calcula el tiempo por debajo de 63 mg/dL.
        
        Returns:
            float: Porcentaje de tiempo < 63 mg/dL
        """
        return self.TBR(63)
    
    def TBR55(self) -> float:
        """
        Calcula el tiempo por debajo de 55 mg/dL.
        
        Returns:
            float: Porcentaje de tiempo < 55 mg/dL
        """
        return self.TBR(55)

    def time_statistics(self) -> Dict[str, Any]:
        """
        Calcula las estadísticas de tiempo de glucosa estándar.
        
        Returns:
            dict: Estadísticas completas de tiempo en rango
        """
        return {
            '%Data': self.data_completeness(),
            'TIR': self.TIR(),
            'TIR_tight': self.TIR_tight(),
            'TBR70': self.TBR70(),
            'TBR55': self.TBR55(),
            'TAR250': self.TAR250(),
            'TAR180': self.TAR180(),
            'TAR140': self.TAR140(),
        }
    
    def time_statistics_pregnancy(self) -> Dict[str, Any]:
        """
        Calcula las estadísticas de tiempo específicas para embarazo.
        
        Siguiendo las guías internacionales para diabetes gestacional.
        
        Returns:
            dict: Estadísticas de tiempo en rango para embarazo
        """
        return {
            '%Data': self.data_completeness(),
            'TIR_pregnancy': self.TIR_pregnancy(),  # 63-140 mg/dL
            'TBR63': self.TBR63(),    # < 63 mg/dL
            'TAR140': self.TAR140(),  # > 140 mg/dL 
            'TAR250': self.TAR250(),  # > 250 mg/dL
        }
    
    def time_range_summary(self) -> Dict[str, Any]:
        """
        Resumen completo de todas las métricas de tiempo en rango.
        
        Returns:
            dict: Resumen completo de TIR, TAR y TBR
        """
        return {
            'data_completeness': self.data_completeness(),
            'standard_ranges': {
                'TIR': self.TIR(),
                'TIR_tight': self.TIR_tight(),
                'TAR180': self.TAR180(),
                'TAR250': self.TAR250(),
                'TBR70': self.TBR70(),
                'TBR55': self.TBR55(),
            },
            'pregnancy_ranges': {
                'TIR_pregnancy': self.TIR_pregnancy(),
                'TBR63': self.TBR63(),
                'TAR140': self.TAR140(),
            },
            'custom_thresholds': {
                'TAR140': self.TAR140(),
                'TBR63': self.TBR63(),
            }
        } 