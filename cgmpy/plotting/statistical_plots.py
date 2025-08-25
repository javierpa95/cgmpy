"""
Módulo de gráficos estadísticos para datos de glucosa.

Este módulo contiene las funciones para generar gráficos estadísticos:
- Histogramas de distribución
- Gráficos de tiempo en rango
- Gráficos de correlación
- Análisis de distribución estadística
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Union, Optional, List


class StatisticalPlotter:
    """
    Clase para generar gráficos estadísticos de glucosa.
    
    Esta clase debe ser utilizada como mixin con GlucoseData.
    """
    
    def histogram(self, bin_width: int = 10):
        """
        Genera y muestra el histograma de glucosa con intervalos fijos.
        
        Args:
            bin_width: Ancho de cada intervalo en mg/dL (por defecto 10)
        """
        # Calcular los bordes de los bins
        min_glucose = 0  # O podrías usar self.data['glucose'].min()
        max_glucose = 500  # O podrías usar self.data['glucose'].max()
        bins = range(int(min_glucose), int(max_glucose) + bin_width, bin_width)
        
        # Crear figura
        plt.figure(figsize=(12, 8))
        
        # Crear histograma
        plt.hist(self.data['glucose'], bins=bins, edgecolor='black', alpha=0.7)
        
        # Configurar zonas de glucemia
        plt.axvspan(0, 70, color='#ffcccb', alpha=0.3, label='Hipoglucemia')
        plt.axvspan(70, 180, color='#90ee90', alpha=0.3, label='Rango objetivo')
        plt.axvspan(180, 400, color='#ffcccb', alpha=0.3, label='Hiperglucemia')
        
        # Configurar gráfico
        plt.xlabel('Nivel de Glucosa (mg/dL)', fontsize=12)
        plt.ylabel('Frecuencia', fontsize=12)
        plt.title(f'Histograma de Glucosa (Intervalos de {bin_width} mg/dL)', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def plot_time_in_range(self, pregnancy: bool = False):
        """
        Genera un gráfico circular (pie chart) del tiempo en rango.
        
        Args:
            pregnancy: Si es True, usa los rangos específicos para embarazo
        """
        if pregnancy:
            # Rangos para embarazo
            tir_pregnancy = self.TIR_pregnancy()  # 63-140 mg/dL
            tbr = self.TBR(63)                   # < 63 mg/dL  
            tar = self.TAR(140)                  # > 140 mg/dL
            
            labels = ['TIR Embarazo\n(63-140 mg/dL)', 'TBR\n(< 63 mg/dL)', 'TAR\n(> 140 mg/dL)']
            sizes = [tir_pregnancy, tbr, tar]
            colors = ['#90ee90', '#ffcccb', '#ffa500']
            title = 'Tiempo en Rango - Embarazo'
        else:
            # Rangos estándar
            tir = self.TIR()          # 70-180 mg/dL
            tbr70 = self.TBR70()      # 55-70 mg/dL
            tbr55 = self.TBR55()      # < 55 mg/dL
            tar180 = self.TAR180()    # 180-250 mg/dL
            tar250 = self.TAR250()    # > 250 mg/dL
            
            labels = ['TIR\n(70-180 mg/dL)', 'TBR Nivel 1\n(55-70 mg/dL)', 
                     'TBR Nivel 2\n(< 55 mg/dL)', 'TAR Nivel 1\n(180-250 mg/dL)', 
                     'TAR Nivel 2\n(> 250 mg/dL)']
            sizes = [tir, tbr70, tbr55, tar180, tar250]
            colors = ['#90ee90', '#ffeb9c', '#ffcccb', '#ffa500', '#ff6666']
            title = 'Tiempo en Rango - Estándar'
        
        # Filtrar valores mayores que 0 para el gráfico
        non_zero_data = [(label, size, color) for label, size, color in zip(labels, sizes, colors) if size > 0]
        if non_zero_data:
            labels, sizes, colors = zip(*non_zero_data)
        
        # Crear figura
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Gráfico circular
        wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                          startangle=90, textprops={'fontsize': 10})
        
        ax1.set_title(title, fontsize=14, fontweight='bold')
        
        # Gráfico de barras horizontal
        y_pos = np.arange(len(labels))
        bars = ax2.barh(y_pos, sizes, color=colors, alpha=0.7)
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(labels, fontsize=10)
        ax2.set_xlabel('Porcentaje (%)', fontsize=12)
        ax2.set_title('Distribución Detallada', fontsize=14, fontweight='bold')
        
        # Añadir valores en las barras
        for i, (bar, size) in enumerate(zip(bars, sizes)):
            ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                    f'{size:.1f}%', ha='left', va='center', fontsize=10)
        
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.show()

    def plot_distribution_comparison(self, target_ranges: Optional[List[tuple]] = None):
        """
        Compara la distribución actual con rangos objetivo.
        
        Args:
            target_ranges: Lista de tuplas (min, max, label, color) para comparar
        """
        if target_ranges is None:
            target_ranges = [
                (70, 180, 'Rango Objetivo', '#90ee90'),
                (0, 70, 'Hipoglucemia', '#ffcccb'),
                (180, 400, 'Hiperglucemia', '#ffa500')
            ]
        
        # Crear figura con subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Histograma con densidad
        ax1.hist(self.data['glucose'], bins=50, density=True, alpha=0.7, 
                color='skyblue', edgecolor='black')
        
        # Añadir zonas objetivo
        for min_val, max_val, label, color in target_ranges:
            ax1.axvspan(min_val, max_val, alpha=0.3, color=color, label=label)
        
        ax1.set_xlabel('Glucosa (mg/dL)')
        ax1.set_ylabel('Densidad')
        ax1.set_title('Distribución de Glucosa')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Box plot
        ax2.boxplot(self.data['glucose'], vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightblue'))
        ax2.set_ylabel('Glucosa (mg/dL)')
        ax2.set_title('Box Plot de Glucosa')
        ax2.grid(True, alpha=0.3)
        
        # 3. Q-Q plot (comparación con distribución normal)
        from scipy import stats
        stats.probplot(self.data['glucose'], dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot (Normalidad)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Estadísticas resumidas
        ax4.axis('off')
        stats_text = self._generate_statistics_text()
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.show()

    def plot_correlation_matrix(self, time_segments: Optional[List[str]] = None):
        """
        Genera una matriz de correlación entre diferentes segmentos temporales.
        
        Args:
            time_segments: Lista de segmentos de tiempo para analizar
        """
        if time_segments is None:
            time_segments = ['00:00-06:00', '06:00-12:00', '12:00-18:00', '18:00-24:00']
        
        # Preparar datos por segmentos
        data_copy = self.data.copy()
        data_copy['hour'] = data_copy['time'].dt.hour
        data_copy['date'] = data_copy['time'].dt.date
        
        # Crear DataFrame con promedios por segmento y día
        segment_data = {}
        
        for segment in time_segments:
            start_hour, end_hour = segment.split('-')
            start_h = int(start_hour.split(':')[0])
            end_h = int(end_hour.split(':')[0])
            
            if end_h == 0:  # Caso especial para 24:00
                end_h = 24
            
            if start_h < end_h:
                mask = (data_copy['hour'] >= start_h) & (data_copy['hour'] < end_h)
            else:  # Caso de segmento que cruza medianoche
                mask = (data_copy['hour'] >= start_h) | (data_copy['hour'] < end_h)
            
            segment_glucose = data_copy[mask].groupby('date')['glucose'].mean()
            segment_data[segment] = segment_glucose
        
        # Crear DataFrame de correlación
        correlation_df = pd.DataFrame(segment_data)
        correlation_matrix = correlation_df.corr()
        
        # Crear figura
        plt.figure(figsize=(10, 8))
        
        # Mapa de calor
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
        
        plt.title('Matriz de Correlación entre Segmentos Temporales', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def _generate_statistics_text(self) -> str:
        """Genera texto con estadísticas resumidas."""
        glucose_data = self.data['glucose']
        
        stats_text = f"""ESTADÍSTICAS DESCRIPTIVAS

Media:           {glucose_data.mean():.1f} mg/dL
Mediana:         {glucose_data.median():.1f} mg/dL
Desv. Estándar:  {glucose_data.std():.1f} mg/dL
CV:              {(glucose_data.std()/glucose_data.mean()*100):.1f}%

Percentiles:
P5:              {glucose_data.quantile(0.05):.1f} mg/dL
P25:             {glucose_data.quantile(0.25):.1f} mg/dL
P75:             {glucose_data.quantile(0.75):.1f} mg/dL
P95:             {glucose_data.quantile(0.95):.1f} mg/dL

Tiempo en Rango:
TIR (70-180):    {self.TIR():.1f}%
TBR (<70):       {self.TBR(70):.1f}%
TAR (>180):      {self.TAR(180):.1f}%

GMI:             {self.gmi():.1f}%
"""
        return stats_text 