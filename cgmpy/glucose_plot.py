from .glucose_data import GlucoseData
import matplotlib.pyplot as plt
from typing import Union
import seaborn as sns
import pandas as pd
import numpy as np
import datetime

class GlucosePlot(GlucoseData):
    def __init__(self, data_source: Union[str, pd.DataFrame], date_col: str="time", glucose_col: str="glucose", delimiter: Union[str, None] = None, header: int = 0, start_date: Union[str, datetime.datetime, None] = None,end_date: Union[str, datetime.datetime, None] = None):
        super().__init__(data_source, date_col, glucose_col, delimiter, header, start_date, end_date)

     # GRÁFICOS
    def generate_agp(self, smoothing_window: int = 15):
        """
        Genera y muestra el Perfil de Glucosa Ambulatoria (AGP) mejorado.
        
        :param smoothing_window: Ventana de suavizado en minutos (por defecto 15)
        """
        self.data['time_decimal'] = (self.data['time'].dt.hour + 
                                    self.data['time'].dt.minute / 60.0).round(2)
        
        # Primero agrupamos y calculamos los percentiles
        percentiles = (self.data.groupby('time_decimal')['glucose']
                      .agg([
                          lambda x: np.percentile(x, 5),
                          lambda x: np.percentile(x, 25),
                          lambda x: np.percentile(x, 50),
                          lambda x: np.percentile(x, 75),
                          lambda x: np.percentile(x, 95)
                      ]))
        
        # Renombramos las columnas
        percentiles.columns = [0.05, 0.25, 0.5, 0.75, 0.95]
        
        # Aplicamos el suavizado después de calcular los percentiles
        for col in percentiles.columns:
            percentiles[col] = percentiles[col].rolling(window=smoothing_window, center=True, min_periods=1).mean()
        
        # Asegurar que los datos están ordenados
        percentiles = percentiles.sort_index()
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Zonas de glucemia
        ax.axhspan(0, 70, facecolor='#ffcccb', alpha=0.3, label='Hipoglucemia')
        ax.axhspan(70, 180, facecolor='#90ee90', alpha=0.3, label='Rango objetivo')
        ax.axhspan(180, 400, facecolor='#ffcccb', alpha=0.3, label='Hiperglucemia')
        
        # Líneas de percentiles
        ax.plot(percentiles.index, percentiles[0.5], label='Mediana', color='blue', linewidth=2)
        ax.fill_between(percentiles.index, percentiles[0.25], percentiles[0.75], color='blue', alpha=0.3, label='Rango Intercuartil')
        ax.fill_between(percentiles.index, percentiles[0.05], percentiles[0.95], color='lightblue', alpha=0.2, label='Percentiles 5-95%')
        
        # Líneas horizontales en 70 y 180 mg/dL
        ax.axhline(y=70, color='red', linestyle='--', linewidth=1)
        ax.axhline(y=180, color='red', linestyle='--', linewidth=1)
        
        # Configuración de ejes y etiquetas
        ax.set_xlabel('Hora del Día', fontsize=12)
        ax.set_ylabel('Nivel de Glucosa (mg/dL)', fontsize=12)
        ax.set_title('Perfil de Glucosa Ambulatoria (AGP)', fontsize=16, fontweight='bold')
        
        # Configuración de la leyenda
        ax.legend(title="Leyenda", loc="upper left", fontsize=10)
        
        # Configuración de la cuadrícula
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Configuración de los ticks del eje x
        ax.set_xticks(range(0, 25, 3))
        ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 25, 3)])
        
        # Ajuste de los límites del eje y
        ax.set_ylim(0, 400)
        
        # Ajustes finales
        plt.tight_layout()
        plt.show()


    def day_graph(self, fecha=None):
        """
        Genera y muestra el gráfico de glucosa para un día específico.
        
        :param fecha: Fecha opcional en formato 'YYYY-MM-DD'. Si no se proporciona, se usa el primer día del DataFrame.
        """
        # Si no se proporciona fecha, usar el primer día del DataFrame
        if fecha is None:
            fecha = self.data['time'].dt.date.min()
        else:
            fecha = pd.to_datetime(fecha).date()
        
        # Filtrar datos para el día específico
        day_data = self.data[self.data['time'].dt.date == fecha].copy()
        
        if day_data.empty:
            print(f"No hay datos para la fecha {fecha}")
            return
        
        # Convertir la hora a un formato numérico para el gráfico
        day_data['hours'] = day_data['time'].dt.hour + day_data['time'].dt.minute / 60.0
        
        # Configurar el estilo de seaborn para un aspecto más profesional
        sns.set_style("whitegrid")
        sns.set_context("notebook", font_scale=1.1)

        fig, ax = plt.subplots(figsize=(16, 9))

        # Zonas de glucemia con colores más suaves y transparencia
        ax.axhspan(0, 70, facecolor='#FF9999', alpha=0.2, label='Hipoglucemia')
        ax.axhspan(70, 180, facecolor='#90EE90', alpha=0.2, label='Rango objetivo')
        ax.axhspan(180, 400, facecolor='#FFB266', alpha=0.2, label='Hiperglucemia')

        # Gráfico de línea con marcadores para los puntos de datos
        ax.plot(day_data['hours'], day_data['glucose'], label='Glucosa', 
                color='#3366CC', linewidth=2, marker='o', markersize=4)

        # Líneas horizontales en 70 y 180 mg/dL con etiquetas
        ax.axhline(y=70, color='#FF6666', linestyle='--', linewidth=1)
        ax.axhline(y=180, color='#FF6666', linestyle='--', linewidth=1)
        ax.text(24, 72, '70 mg/dL', va='bottom', ha='right', color='#FF6666')
        ax.text(24, 182, '180 mg/dL', va='bottom', ha='right', color='#FF6666')

        # Configuración de ejes y etiquetas
        ax.set_xlabel('Hora del Día', fontsize=12, fontweight='bold')
        ax.set_ylabel('Nivel de Glucosa (mg/dL)', fontsize=12, fontweight='bold')
        ax.set_title(f'Niveles de Glucosa - {fecha}', fontsize=16, fontweight='bold')

        # Configuración de la leyenda
        ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)

        # Ajuste de los límites del eje y
        ax.set_ylim(0, 400)

        # Configuración del eje x
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 3))
        ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 25, 3)])

        # Rotar las etiquetas del eje x para mejor legibilidad
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # Añadir una cuadrícula suave
        ax.grid(True, linestyle=':', alpha=0.6)

        # Ajustar los márgenes
        plt.tight_layout()

        # Mostrar el gráfico
        plt.show()


    def histogram(self, bin_width=10):
        """
        Genera y muestra el histograma de glucosa con intervalos fijos.
        
        :param bin_width: Ancho de cada intervalo en mg/dL (por defecto 10)
        """
        # Calcular los bordes de los bins
        min_glucose = 0  # O podrías usar self.data['glucose'].min()
        max_glucose = 500  # O podrías usar self.data['glucose'].max()
        bins = range(int(min_glucose), int(max_glucose) + bin_width, bin_width)
        
        plt.hist(self.data['glucose'], bins=bins, edgecolor='black')
        plt.xlabel('Nivel de Glucosa (mg/dL)')
        plt.ylabel('Frecuencia')
        plt.title(f'Histograma de Glucosa (Intervalos de {bin_width} mg/dL)')
        plt.axvspan(0, 70, color='#ffcccb', alpha=0.3, label='Hipoglucemia')
        plt.axvspan(70, 180, color='#90ee90', alpha=0.3, label='Rango objetivo')
        plt.axvspan(180, 400, color='#ffcccb', alpha=0.3, label='Hiperglucemia')
        plt.legend()
        plt.show()
