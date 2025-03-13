from .glucose_metrics import GlucoseMetrics
import matplotlib.pyplot as plt
from typing import Union
import seaborn as sns
import pandas as pd
import numpy as np
import datetime
import scipy.stats as stats
import os

class GlucosePlot(GlucoseMetrics):
    def __init__(self, data_source: Union[str, pd.DataFrame], 
                 date_col: str="time", 
                 glucose_col: str="glucose", 
                 delimiter: Union[str, None] = None, 
                 header: int = 0, 
                 start_date: Union[str, datetime.datetime, None] = None,
                 end_date: Union[str, datetime.datetime, None] = None,
                 log: bool = False):
        
        # Verificar si GlucoseData ya ha sido inicializado
        if not hasattr(self, 'data'):
            super().__init__(data_source, date_col, glucose_col, delimiter, header, 
                           start_date, end_date, log)

     # GRÁFICOS
    def plot_agp(self, smoothing_window: int = 15):
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

    def generate_week_agp(self, smoothing_window: int = 15, combined: bool = True):
        """
        Genera y muestra el Perfil de Glucosa Ambulatoria (AGP) por días de la semana.
        
        :param smoothing_window: Ventana de suavizado en minutos (por defecto 15)
        :param combined: Si es True, muestra todos los días en un solo gráfico. Si es False,
                    muestra un subplot para cada día.
        """
        # Preparar los datos
        self.data['time_decimal'] = (self.data['time'].dt.hour + 
                                    self.data['time'].dt.minute / 60.0).round(2)
        self.data['weekday'] = self.data['time'].dt.day_name(locale='es_ES')
        
        # Orden de los días y colores
        dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
        colores = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD', '#D4A5A5', '#9B59B6']

        if combined:
            # Crear un solo gráfico
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # Zonas de glucemia
            ax.axhspan(0, 70, facecolor='#ffcccb', alpha=0.3, label='Hipoglucemia')
            ax.axhspan(70, 180, facecolor='#90ee90', alpha=0.3, label='Rango objetivo')
            ax.axhspan(180, 400, facecolor='#ffcccb', alpha=0.3, label='Hiperglucemia')
            
            for dia, color in zip(dias, colores):
                # Filtrar datos para el día específico
                dia_data = self.data[self.data['weekday'] == dia]
                
                if not dia_data.empty:
                    # Calcular percentiles
                    percentiles = (dia_data.groupby('time_decimal')['glucose']
                                 .agg([
                                     lambda x: np.percentile(x, 25),
                                     lambda x: np.percentile(x, 50),
                                     lambda x: np.percentile(x, 75)
                                 ]))
                    
                    # Renombrar columnas
                    percentiles.columns = [0.25, 0.5, 0.75]
                    
                    # Aplicar suavizado
                    for col in percentiles.columns:
                        percentiles[col] = percentiles[col].rolling(
                            window=smoothing_window, center=True, min_periods=1).mean()
                    
                    # Graficar línea mediana
                    ax.plot(percentiles.index, percentiles[0.5], 
                           label=f'{dia} (n={len(dia_data["time"].dt.date.unique())} días)', 
                           color=color, linewidth=2)
                    
                    # Área del IQR con transparencia
                    ax.fill_between(percentiles.index, 
                                  percentiles[0.25], 
                                  percentiles[0.75], 
                                  color=color, alpha=0.1)
            
            # Líneas horizontales
            ax.axhline(y=70, color='red', linestyle='--', linewidth=1)
            ax.axhline(y=180, color='red', linestyle='--', linewidth=1)
            
            # Configuración del gráfico
            ax.set_title('Perfil de Glucosa Ambulatoria (AGP) por Día de la Semana', 
                        fontsize=14, pad=20)
            ax.set_xlabel('Hora del Día', fontsize=12)
            ax.set_ylabel('Nivel de Glucosa (mg/dL)', fontsize=12)
            ax.set_ylim(0, 400)
            ax.grid(True, linestyle=':', alpha=0.6)
            
            # Configuración del eje x
            ax.set_xticks(range(0, 25, 3))
            ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 25, 3)])
            
            # Leyenda
            ax.legend(title="Días de la semana", 
                     loc='center left', 
                     bbox_to_anchor=(1, 0.5),
                     fontsize=10)
            
            plt.tight_layout()
            plt.show()
        else:
            # Crear subplots
            fig, axes = plt.subplots(7, 1, figsize=(15, 20), sharex=True)
            fig.suptitle('Perfil de Glucosa Ambulatoria (AGP) por Día de la Semana', 
                        fontsize=16, fontweight='bold', y=0.92)
            
            for ax, dia in zip(axes, dias):
                # Filtrar datos para el día específico
                dia_data = self.data[self.data['weekday'] == dia]
                
                if not dia_data.empty:
                    # Calcular percentiles
                    percentiles = (dia_data.groupby('time_decimal')['glucose']
                                 .agg([
                                     lambda x: np.percentile(x, 5),
                                     lambda x: np.percentile(x, 25),
                                     lambda x: np.percentile(x, 50),
                                     lambda x: np.percentile(x, 75),
                                     lambda x: np.percentile(x, 95)
                                 ]))
                    
                    # Renombrar columnas
                    percentiles.columns = [0.05, 0.25, 0.5, 0.75, 0.95]
                    
                    # Aplicar suavizado
                    for col in percentiles.columns:
                        percentiles[col] = percentiles[col].rolling(
                            window=smoothing_window, center=True, min_periods=1).mean()
                    
                    # Zonas de glucemia
                    ax.axhspan(0, 70, facecolor='#ffcccb', alpha=0.3)
                    ax.axhspan(70, 180, facecolor='#90ee90', alpha=0.3)
                    ax.axhspan(180, 400, facecolor='#ffcccb', alpha=0.3)
                    
                    # Líneas de percentiles
                    ax.plot(percentiles.index, percentiles[0.5], 
                           label='Mediana', color='blue', linewidth=2)
                    ax.fill_between(percentiles.index, percentiles[0.25], percentiles[0.75], 
                                  color='blue', alpha=0.3)
                    ax.fill_between(percentiles.index, percentiles[0.05], percentiles[0.95], 
                                  color='lightblue', alpha=0.2)
                    
                    # Líneas horizontales
                    ax.axhline(y=70, color='red', linestyle='--', linewidth=1)
                    ax.axhline(y=180, color='red', linestyle='--', linewidth=1)
                
                # Configuración del eje
                ax.set_ylim(0, 400)
                ax.set_ylabel('Glucosa (mg/dL)')
                ax.set_title(dia)
                ax.grid(True, linestyle=':', alpha=0.6)
                
                # Añadir el número de días analizados
                n_dias = len(dia_data['time'].dt.date.unique())
                ax.text(0.02, 0.95, f'n={n_dias} días', 
                        transform=ax.transAxes, fontsize=10)
            
            # Configuración del eje x (solo visible en el último subplot)
            axes[-1].set_xticks(range(0, 25, 3))
            axes[-1].set_xticklabels([f'{h:02d}:00' for h in range(0, 25, 3)])
            axes[-1].set_xlabel('Hora del Día')
            
            # Leyenda (solo en el primer subplot)
            handles = [
                plt.Rectangle((0,0),1,1, facecolor='#ffcccb', alpha=0.3, label='Fuera de rango'),
                plt.Rectangle((0,0),1,1, facecolor='#90ee90', alpha=0.3, label='Rango objetivo'),
                plt.Line2D([0], [0], color='blue', linewidth=2, label='Mediana'),
                plt.Rectangle((0,0),1,1, facecolor='blue', alpha=0.3, label='IQR (25-75%)'),
                plt.Rectangle((0,0),1,1, facecolor='lightblue', alpha=0.2, label='5-95%')
            ]
            axes[0].legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))
            
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


    def plot_overlapping_days(self):
        """
        Genera un gráfico con los perfiles de glucosa de múltiples días superpuestos.
        Cada línea representa un día diferente.
        """
        # Crear columnas para hora del día en formato decimal
        self.data['time_decimal'] = (self.data['time'].dt.hour + 
                                    self.data['time'].dt.minute / 60.0)
        self.data['date'] = self.data['time'].dt.date
        
        # Configurar el estilo del gráfico
        plt.figure(figsize=(12, 8))
        
        # Graficar cada día con un color diferente y menor opacidad
        dates = self.data['date'].unique()
        
        # Calcular el perfil medio
        mean_profile = (self.data.groupby('time_decimal')['glucose']
                       .mean()
                       .rolling(window=15, center=True, min_periods=1)
                       .mean())
        
        # Graficar cada día individual con líneas más claras
        for date in dates:
            day_data = self.data[self.data['date'] == date]
            plt.plot(day_data['time_decimal'], day_data['glucose'], 
                    color='gray', alpha=0.2, linewidth=1)
        
        # Graficar el perfil medio con una línea más gruesa
        plt.plot(mean_profile.index, mean_profile.values, 
                color='black', linewidth=2, label='Perfil medio')
        
        # Configuración del gráfico
        plt.xlabel('Hora del Día', fontsize=12)
        plt.ylabel('Nivel de Glucosa (mg/dL)', fontsize=12)
        plt.title('Perfiles de Glucosa Superpuestos', fontsize=14)
        
        # Configurar el eje x para mostrar las horas
        plt.xticks(range(0, 25, 3), 
                   [f'{h:02d}:00' for h in range(0, 25, 3)])
        
        # Añadir líneas horizontales para los rangos objetivo
        plt.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        plt.axhline(y=180, color='red', linestyle='--', alpha=0.5)
        
        # Añadir zonas coloreadas para los rangos
        plt.axhspan(0, 70, facecolor='#ffcccb', alpha=0.2)
        plt.axhspan(70, 180, facecolor='#90ee90', alpha=0.2)
        plt.axhspan(180, 400, facecolor='#ffcccb', alpha=0.2)
        
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Ajustar los límites del eje y
        plt.ylim(0, 400)
        
        plt.tight_layout()
        plt.show()

    def plot_week_boxplots(self):
        """
        Genera un gráfico de boxplots para visualizar la distribución de glucosa
        por día de la semana, incluyendo el número de días para cada día.
        """
        # Añadir columna con el nombre del día de la semana y la fecha
        self.data['weekday'] = self.data['time'].dt.day_name(locale='es_ES')
        self.data['date'] = self.data['time'].dt.date
        
        # Definir el orden correcto de los días
        orden_dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
        
        # Calcular el número de días únicos para cada día de la semana
        dias_unicos = self.data.groupby('weekday')['date'].nunique()
        
        # Crear etiquetas personalizadas con el número de días
        etiquetas = [f'{dia}\n(n={dias_unicos[dia]} días)' for dia in orden_dias]
        
        # Crear la figura
        plt.figure(figsize=(12, 8))
        
        # Crear el boxplot
        sns.boxplot(x='weekday', y='glucose', data=self.data, 
                    order=orden_dias,
                    whis=1.5,  # Define los bigotes como 1.5 * IQR
                    medianprops=dict(color="red", linewidth=1.5),
                    flierprops=dict(marker='o', markerfacecolor='gray', markersize=4))
        
        # Zonas de glucemia
        plt.axhspan(0, 70, color='#ffcccb', alpha=0.2, label='Hipoglucemia')
        plt.axhspan(70, 180, color='#90ee90', alpha=0.2, label='Rango objetivo')
        plt.axhspan(180, 400, color='#ffcccb', alpha=0.2, label='Hiperglucemia')
        
        # Líneas horizontales en 70 y 180 mg/dL
        plt.axhline(y=70, color='red', linestyle='--', linewidth=1)
        plt.axhline(y=180, color='red', linestyle='--', linewidth=1)
        
        # Personalizar el gráfico
        plt.title('Distribución de Glucosa por Día de la Semana', fontsize=14, pad=20)
        plt.xlabel('Día de la Semana', fontsize=12)
        plt.ylabel('Nivel de Glucosa (mg/dL)', fontsize=12)
        
        # Actualizar las etiquetas del eje x con la información del número de días
        plt.xticks(range(len(orden_dias)), etiquetas, rotation=45, ha='right')
        
        # Ajustar los límites del eje y
        plt.ylim(0, 400)
        
        # Añadir leyenda
        plt.legend(title='Rangos', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Ajustar los márgenes
        plt.tight_layout()
        
        plt.show()

    def plot_daily_variations(self):
        """
        Genera dos histogramas: uno para las desviaciones estándar diarias y otro para las medias diarias,
        con líneas verticales que marcan los promedios.
        """
        # Calcular estadísticas diarias
        daily_stats = self.data.groupby(self.data['time'].dt.date)['glucose'].agg(['std', 'mean'])
        
        # Crear la figura con dos subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Histograma de desviaciones estándar
        ax1.hist(daily_stats['std'], bins=20, edgecolor='black', alpha=0.7)
        ax1.axvline(daily_stats['std'].mean(), color='red', linestyle='--', 
                    label=f'Media: {daily_stats["std"].mean():.1f} mg/dL')
        ax1.set_title('Distribución de Desviaciones Estándar Diarias', fontsize=12)
        ax1.set_xlabel('Desviación Estándar (mg/dL)')
        ax1.set_ylabel('Frecuencia')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Histograma de medias
        ax2.hist(daily_stats['mean'], bins=20, edgecolor='black', alpha=0.7)
        ax2.axvline(daily_stats['mean'].mean(), color='red', linestyle='--',
                    label=f'Media: {daily_stats["mean"].mean():.1f} mg/dL')
        ax2.set_title('Distribución de Medias Diarias', fontsize=12)
        ax2.set_xlabel('Glucosa Media (mg/dL)')
        ax2.set_ylabel('Frecuencia')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Añadir rangos objetivo en el gráfico de medias
        ax2.axvspan(70, 180, color='#90ee90', alpha=0.2, label='Rango objetivo')
        
        # Ajustar el diseño
        plt.tight_layout()
        plt.show()

    ## GRÁFICOS PARA LA VARIABILIDAD

    def plot_sd_components(self):
        """
        Genera una visualización de todos los componentes de la desviación estándar.
        """
        # Obtener las métricas
        metrics = self.calculate_all_sd_metrics()
        
        # Preparar los datos para el gráfico
        metrics_df = pd.DataFrame({
            'Componente': list(metrics.keys()),
            'Valor': list(metrics.values())
        })
        
        # Crear el gráfico
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Crear barras horizontales
        bars = ax.barh(metrics_df['Componente'], metrics_df['Valor'])
        
        # Personalizar colores según el tipo de métrica
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', 
                  '#99FFCC', '#FFB366', '#B266FF', '#66FFB3', '#FF66B2', 
                  '#66B2FF', '#FFB366', '#B366FF']
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Añadir valores en las barras
        for i, v in enumerate(metrics_df['Valor']):
            ax.text(v + 1, i, f'{v:.1f}', va='center')
        
        # Personalizar el gráfico
        ax.set_title('Componentes de la Desviación Estándar de Glucosa', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Valor (mg/dL)', fontsize=12)
        
        # Añadir una cuadrícula suave
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Añadir anotaciones explicativas
        explicaciones = {
            'SDT': 'Desviación estándar total',
            'SDw': 'DE dentro del día',
            'SDhh:mm': 'DE del patrón por tiempo del día',
            'Noche': 'DE segmento nocturno (00:00-08:00)',
            'Día': 'DE segmento diurno (08:00-16:00)',
            'Tarde': 'DE segmento tarde (16:00-00:00)',
            'SDws_1h': 'DE dentro de series de 1 hora',
            'SDws_6h': 'DE dentro de series de 6 horas',
            'SDws_24h': 'DE dentro de series de 24 horas',
            'SDdm': 'DE de medias diarias',
            'SDbhh:mm': 'DE entre días por punto temporal',
            'SDbhh:mm_dm': 'DE entre días ajustada',
            'SDI': 'DE de interacción'
        }
        
        # Añadir un texto explicativo
        explanation_text = '\n'.join([f'{k}: {v}' for k, v in explicaciones.items()])
        plt.figtext(1.02, 0.5, explanation_text, 
                    fontsize=8, ha='left', va='center',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        # Ajustar los márgenes para acomodar la leyenda
        plt.subplots_adjust(right=0.75)
        
        plt.show()

    def plot_sd_between_timepoints(self, min_count_threshold: float = 0.5, filter_outliers: bool = True,
                              agrupar_por_intervalos: bool = False, intervalo_minutos: int = 5):
        """
        Genera un gráfico que muestra el patrón promedio de glucosa por hora del día
        y el valor de SDhh:mm (desviación estándar del patrón promedio).
        
        Este gráfico visualiza cómo los niveles de glucosa varían a lo largo del día
        en promedio, y muestra la variabilidad de este patrón mediante SDhh:mm.
        También identifica puntos con pocos datos que podrían ser atípicos.
        
        Versión optimizada para grandes conjuntos de datos.
        
        :param min_count_threshold: Umbral para considerar una marca temporal como válida
                               (proporción de la mediana de conteos). Por defecto 0.5 (50%).
        :param filter_outliers: Si es True, filtra las marcas temporales con pocos datos
                               antes de calcular SDhh:mm.
        :param agrupar_por_intervalos: Si es True, agrupa los datos en intervalos regulares de tiempo.
        :param intervalo_minutos: Tamaño del intervalo en minutos para la agrupación (por defecto 5 min).
        """
        # Calcular SDhh:mm y obtener estadísticas
        sd_result = self.sd_between_timepoints(min_count_threshold, filter_outliers, 
                                              agrupar_por_intervalos, intervalo_minutos)
        sd_value = sd_result['sd']
        
        # Preparar datos para graficar de manera más eficiente
        # Crear una copia simplificada de los datos con solo las columnas necesarias
        df = self.data[['time', 'glucose']].copy()
        
        # Extraer características de tiempo de forma vectorizada
        df['hour'] = df['time'].dt.hour
        df['minute'] = df['time'].dt.minute
        df['time_decimal'] = df['hour'] + df['minute']/60
        df['day'] = df['time'].dt.date
        
        if agrupar_por_intervalos:
            # Agrupar por intervalo de tiempo
            df['interval_minutes'] = (df['hour'] * 60 + df['minute']) // intervalo_minutos * intervalo_minutos
            df['time_decimal'] = df['interval_minutes'] / 60
            # Calcular promedios por día e intervalo
            df_means = df.groupby(['day', 'time_decimal'])['glucose'].mean().reset_index()
            
            # Calcular promedios globales por intervalo
            result = df_means.groupby('time_decimal')['glucose'].agg(['mean', 'count', 'std']).reset_index()
            
            # Crear listas para gráficos
            time_decimal = result['time_decimal'].tolist()
            means = result['mean'].tolist()
            counts = result['count'].tolist()
        else:
            # Calcular promedios por día y tiempo (hora:minuto)
            df_means = df.groupby(['day', 'hour', 'minute'])['glucose'].mean().reset_index()
            df_means['time_decimal'] = df_means['hour'] + df_means['minute']/60
            
            # Calcular promedios globales por punto temporal
            result = df_means.groupby('time_decimal')['glucose'].agg(['mean', 'count', 'std']).reset_index()
            
            # Crear listas para gráficos
            time_decimal = result['time_decimal'].tolist()
            means = result['mean'].tolist()
            counts = result['count'].tolist()
        
        # Calcular umbral para puntos de pocos datos
        median_count = np.median(counts)
        threshold = median_count * min_count_threshold
        
        # Identificar puntos con pocos datos
        low_data_points = [i for i, count in enumerate(counts) if count < threshold]
        
        # Configurar tamaño de figura
        plt.figure(figsize=(12, 6))
        
        # Crear subplots
        ax1 = plt.subplot(111)
        
        # Graficar la línea de glucosa promedio
        ax1.plot(time_decimal, means, 'b-', linewidth=2, label='Glucosa promedio')
        
        # Marcar puntos con pocos datos en rojo
        if low_data_points and not filter_outliers:
            low_data_times = [time_decimal[i] for i in low_data_points]
            low_data_means = [means[i] for i in low_data_points]
            ax1.plot(low_data_times, low_data_means, 'ro', label='Puntos con pocos datos')
        
        # Configurar eje x
        ax1.set_xticks(range(0, 25, 3))
        ax1.set_xlim(0, 24)
        ax1.set_xlabel('Hora del día')
        
        # Configurar eje y izquierdo (glucosa)
        ax1.set_ylabel('Glucosa mg/dL')
        ax1.grid(True)
        
        # Crear el título y añadir información de SDhh:mm
        plt.title(f'Patrón glucémico diario promedio\nSDhh:mm = {sd_value:.2f} mg/dL')
        
        # Añadir leyenda
        plt.legend(loc='upper left')
        
        # Ajustar diseño
        plt.tight_layout()
        
        # Mostrar gráfico
        plt.show()

    def plot_sd_same_timepoint(self, min_count_threshold: float = 0.5, filter_outliers: bool = True,
                              agrupar_por_intervalos: bool = False, intervalo_minutos: int = 5):
        """
        Genera un gráfico que muestra la desviación estándar entre días para cada punto temporal (SDbhh:mm).
        
        Este gráfico visualiza cómo varía la desviación estándar de los niveles de glucosa 
        para cada hora específica del día entre diferentes días, mostrando la consistencia
        del patrón glucémico día a día.
        
        Versión optimizada para grandes conjuntos de datos.
        
        :param min_count_threshold: Umbral para considerar una marca temporal como válida
                               (proporción de la mediana de conteos). Por defecto 0.5 (50%).
        :param filter_outliers: Si es True, filtra las marcas temporales con pocos datos
                           antes de calcular SDbhh:mm.
        :param agrupar_por_intervalos: Si es True, agrupa los datos en intervalos regulares de tiempo.
        :param intervalo_minutos: Tamaño del intervalo en minutos para la agrupación (por defecto 5 min).
        """
        # Calcular SDbhh:mm y obtener estadísticas de forma optimizada
        sd_result = self.sd_same_timepoint(min_count_threshold, filter_outliers, 
                                          agrupar_por_intervalos, intervalo_minutos)
        sd_value = sd_result['sd']
        
        # Preparar datos para graficar de forma eficiente
        # Convertir los diccionarios de resultados a DataFrames para procesamiento vectorizado
        df_results = pd.DataFrame({
            'time_str': list(sd_result['sd_por_marca'].keys()),
            'sd': list(sd_result['sd_por_marca'].values()),
            'mean': list(sd_result['valores_por_marca'].values()),
            'count': list(sd_result['conteo_por_marca'].values())
        })
        
        # Convertir marca de tiempo a decimal para gráfico
        df_results['hour'] = df_results['time_str'].apply(lambda x: int(x.split(':')[0]))
        df_results['minute'] = df_results['time_str'].apply(lambda x: int(x.split(':')[1]))
        df_results['time_decimal'] = df_results['hour'] + df_results['minute']/60.0
        
        # Ordenar por tiempo para un gráfico continuo
        df_results = df_results.sort_values('time_decimal')
        
        # Calcular umbral para puntos con pocos datos
        threshold = sd_result['threshold']
        
        # Preparar listas para gráfico
        time_decimal = df_results['time_decimal'].tolist()
        sd_values = df_results['sd'].tolist()
        means = df_results['mean'].tolist()
        counts = df_results['count'].tolist()
        
        # Identificar puntos con pocos datos
        low_data_mask = df_results['count'] < threshold
        
        # Crear figura y ejes
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [1, 1]})
        
        # Gráfico 1: Desviación estándar por hora del día
        norm = plt.Normalize(df_results['count'].min(), df_results['count'].max())
        scatter1 = ax1.scatter(
            df_results['time_decimal'], 
            df_results['sd'], 
            s=df_results['count'].apply(lambda x: max(20, x/2)), 
            c=df_results['count'], 
            cmap='viridis', 
            alpha=0.7, 
            edgecolors='k', 
            linewidths=0.5
        )
        
        # Añadir línea de tendencia suavizada usando rolling window
        window_size = min(15, len(time_decimal) // 10 + 1)  # Ajustar ventana según la cantidad de datos
        if len(time_decimal) > window_size:
            # Crear DataFrame temporal para calcular la media móvil
            temp_df = pd.DataFrame({'time': time_decimal, 'sd': sd_values})
            temp_df = temp_df.sort_values('time')
            temp_df['smooth_sd'] = temp_df['sd'].rolling(window=window_size, center=True).mean()
            
            # Graficar línea suavizada
            valid_mask = temp_df['smooth_sd'].notna()
            ax1.plot(
                temp_df.loc[valid_mask, 'time'], 
                temp_df.loc[valid_mask, 'smooth_sd'], 
                'r-', 
                linewidth=2, 
                label=f'Tendencia (ventana={window_size})'
            )
        
        # Configurar eje x
        ax1.set_xticks(range(0, 25, 3))
        ax1.set_xlim(0, 24)
        ax1.set_xlabel('Hora del día')
        
        # Configurar eje y
        ax1.set_ylabel('SDbhh:mm (mg/dL)')
        
        # Añadir título e información
        ax1.set_title(f'Desviación estándar entre días por punto temporal\nSDbhh:mm = {sd_value:.2f} mg/dL')
        
        # Añadir leyenda de colores para número de días
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('Número de días')
        
        # Añadir línea de referencia para la media global de SDbhh:mm
        ax1.axhline(y=sd_value, color='gray', linestyle='--', alpha=0.7, label=f'Media: {sd_value:.2f} mg/dL')
        
        # Añadir leyenda
        ax1.legend(loc='upper left')
        
        # Gráfico 2: Patrón promedio diario
        norm2 = plt.Normalize(df_results['count'].min(), df_results['count'].max())
        scatter2 = ax2.scatter(
            df_results['time_decimal'], 
            df_results['mean'], 
            s=df_results['count'].apply(lambda x: max(20, x/2)), 
            c=df_results['count'], 
            cmap='viridis', 
            alpha=0.7, 
            edgecolors='k', 
            linewidths=0.5
        )
        
        # Añadir línea de tendencia suavizada
        if len(time_decimal) > window_size:
            # Crear DataFrame temporal para calcular la media móvil
            temp_df = pd.DataFrame({'time': time_decimal, 'mean': means})
            temp_df = temp_df.sort_values('time')
            temp_df['smooth_mean'] = temp_df['mean'].rolling(window=window_size, center=True).mean()
            
            # Graficar línea suavizada
            valid_mask = temp_df['smooth_mean'].notna()
            ax2.plot(
                temp_df.loc[valid_mask, 'time'], 
                temp_df.loc[valid_mask, 'smooth_mean'], 
                'r-', 
                linewidth=2, 
                label=f'Tendencia (ventana={window_size})'
            )
        
        # Configurar eje x
        ax2.set_xticks(range(0, 25, 3))
        ax2.set_xlim(0, 24)
        ax2.set_xlabel('Hora del día')
        
        # Configurar eje y
        ax2.set_ylabel('Glucosa promedio (mg/dL)')
        
        # Añadir título
        ax2.set_title('Patrón glucémico diario promedio')
        
        # Añadir leyenda de colores para número de días
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label('Número de días')
        
        # Añadir leyenda
        ax2.legend(loc='upper left')
        
        # Ajustar diseño
        plt.tight_layout()
        
        # Mostrar gráfico
        plt.show()

    # Modificación para evitar el error de colorbar y layout
    def plot_variability_dashboard(self, min_count_threshold=0.5, filter_outliers=True,
                            agrupar_por_intervalos=False, intervalo_minutos=5):
        """
        Versión modificada de plot_variability_dashboard que evita el error de colorbar y layout.
        """
        # Calcular todas las métricas de variabilidad
        metrics = self.calculate_all_sd_metrics()
        
        # Obtener datos para los gráficos
        sd_between = self.sd_between_timepoints(min_count_threshold, filter_outliers, agrupar_por_intervalos, intervalo_minutos)
        sd_same = self.sd_same_timepoint(min_count_threshold, filter_outliers, agrupar_por_intervalos, intervalo_minutos)
        
        # Crear figura con subplots sin constrained_layout
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.6])
        
        # Gráfico 1: Patrón glucémico diario promedio
        ax1 = fig.add_subplot(gs[0, 0])
        # Calcular el promedio por hora (solo para columnas numéricas)
        hourly_data = self.data[['time', 'glucose']].copy()  # Solo incluir columnas necesarias
        hourly_data['hour'] = hourly_data['time'].dt.hour + hourly_data['time'].dt.minute / 60
        
        # Agrupar y calcular la media solo para 'glucose' con observed=True para evitar el warning
        hourly_grouped = hourly_data.groupby(pd.cut(hourly_data['hour'], bins=24), observed=True)
        hourly_means = hourly_grouped['glucose'].mean()
        hourly_counts = hourly_grouped['glucose'].count()
        
        # Crear el eje x para las horas
        x_hours = np.linspace(0, 24, len(hourly_means))
        
        # Obtener detalles sobre puntos con pocos datos
        median_count = hourly_counts.median()
        threshold = median_count * min_count_threshold
        low_data_hours = hourly_counts < threshold
        
        # Graficar línea principal
        ax1.plot(x_hours, hourly_means, color='blue', linewidth=2, label='Glucosa promedio')
        
        # Marcar puntos con pocos datos si existen
        if low_data_hours.any():
            low_data_x = x_hours[low_data_hours]
            low_data_y = hourly_means[low_data_hours]
            ax1.scatter(low_data_x, low_data_y, color='red', s=30, marker='x', 
                       label='Puntos con pocos datos')
        
        # Añadir información sobre puntos con pocos datos
        low_data_count = low_data_hours.sum()
        total_hours = len(hourly_means)
        
        ax1.set_xlim(0, 24)
        ax1.set_xlabel('Hora del día')
        ax1.set_ylabel('Glucosa (mg/dL)')
        
        # Actualizar título con información sobre puntos filtrados
        title = f'Patrón glucémico diario promedio\nSDhh:mm = {metrics["SDhh:mm"]:.2f} mg/dL'
        if low_data_count > 0:
            title += f'\nHoras con pocos datos: {low_data_count} de {total_hours}'
        ax1.set_title(title)
        
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Gráfico 2: Desviación estándar entre días por punto temporal
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Convertir los datos del diccionario sd_same a un DataFrame para facilitar su uso
        sd_same_df = pd.DataFrame({
            'time_str': list(sd_same['sd_por_marca'].keys()),
            'sd': list(sd_same['sd_por_marca'].values()),
            'mean': list(sd_same['valores_por_marca'].values()),
            'count': list(sd_same['conteo_por_marca'].values())
        })
        
        # Obtener información sobre puntos filtrados
        filtered_timepoints = sd_same['total_timepoints'] - len(sd_same_df) if 'total_timepoints' in sd_same else 0
        
        # Convertir marca de tiempo a decimal para gráfico
        sd_same_df['hour'] = sd_same_df['time_str'].apply(lambda x: int(x.split(':')[0]))
        sd_same_df['minute'] = sd_same_df['time_str'].apply(lambda x: int(x.split(':')[1]))
        sd_same_df['time_decimal'] = sd_same_df['hour'] + sd_same_df['minute']/60.0
        
        # Ordenar por tiempo para un gráfico continuo
        sd_same_df = sd_same_df.sort_values('time_decimal')
        
        # Identificar puntos con pocos datos (pero no filtrados)
        threshold = sd_same['threshold'] if 'threshold' in sd_same else (sd_same_df['count'].median() * min_count_threshold)
        low_data_points = sd_same_df['count'] < threshold
        
        # Crear scatter plot con los datos principales
        scatter = ax2.scatter(
            sd_same_df['time_decimal'], 
            sd_same_df['sd'], 
            c=sd_same_df['mean'],
            cmap='plasma',
            alpha=0.7,
            s=30
        )
        
        # Marcar puntos con pocos datos si existen y no fueron filtrados
        if low_data_points.any() and not filter_outliers:
            low_data_df = sd_same_df[low_data_points]
            ax2.scatter(
                low_data_df['time_decimal'],
                low_data_df['sd'],
                facecolors='none',
                edgecolors='red',
                s=50,
                linewidths=1.5,
                label='Puntos con pocos datos'
            )
        
        # Añadir línea de tendencia
        window = 15  # Tamaño de la ventana para suavizado
        if len(sd_same_df) > window:
            y_smoothed = sd_same_df['sd'].rolling(window=window, center=True).mean()
            ax2.plot(sd_same_df['time_decimal'], y_smoothed, color='red', linewidth=2, label=f'Tendencia (ventana={window})')
        
        # Añadir línea de media
        mean_sd = sd_same_df['sd'].mean()
        ax2.axhline(mean_sd, color='gray', linestyle='--', label=f'Media: {mean_sd:.2f} mg/dL')
        
        # Configurar el gráfico
        ax2.set_xlim(0, 24)
        ax2.set_xlabel('Hora del día')
        ax2.set_ylabel('SDbhh:mm (mg/dL)')
        
        # Actualizar título con información sobre puntos filtrados
        title = f'Desviación estándar entre días por punto temporal\nSDbhh:mm = {metrics["SDbhh:mm"]:.2f} mg/dL'
        if filtered_timepoints > 0:
            title += f'\nPuntos temporales filtrados: {filtered_timepoints}'
        ax2.set_title(title)
        
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Añadir barra de color
        cbar = fig.colorbar(scatter, ax=ax2)
        cbar.set_label('Glucosa media (mg/dL)')
        
        # Gráfico 3: Desviación estándar por día (SDw)
        ax3 = fig.add_subplot(gs[1, 0])
        
        # Calcular estadísticas diarias directamente
        daily_stats = self.data.groupby(self.data['time'].dt.date)['glucose'].agg(['std', 'mean', 'count']).reset_index()
        daily_stats.columns = ['date', 'sd', 'mean', 'count']
        
        # Obtener detalles sobre días con pocos datos
        sdw_details = self.sd_within_day(min_count_threshold)
        valid_days_sdw = set(sdw_details.get('daily_sds', {}).keys())
        all_days = set(daily_stats['date'])
        invalid_days_sdw = all_days - valid_days_sdw
        
        # Crear el gráfico de barras para todos los días
        bars = ax3.bar(daily_stats['date'].astype(str), daily_stats['sd'], alpha=0.7, color='skyblue')
        
        # Marcar días excluidos del cálculo de SDw
        for i, date in enumerate(daily_stats['date']):
            if date in invalid_days_sdw:
                bars[i].set_color('lightcoral')
                bars[i].set_hatch('///')  # Añadir patrón de rayado
        
        # Resaltar días con valores extremos entre los válidos
        valid_stats = daily_stats[daily_stats['date'].isin(valid_days_sdw)]
        if len(valid_stats) > 0:
            max_indices = valid_stats['sd'].nlargest(3).index
            for idx in max_indices:
                if idx < len(bars):
                    bars[idx].set_color('darkblue')
        
        # Añadir línea de media (solo de días válidos)
        ax3.axhline(sdw_details['sd'], color='blue', linestyle='--', label=f'SDw = {sdw_details["sd"]:.2f} mg/dL')
        
        # Configurar el gráfico
        ax3.set_xlabel('Fecha')
        ax3.set_ylabel('Desviación estándar (mg/dL)')
        
        # Actualizar título con información sobre días excluidos
        excluded_days_sdw = len(invalid_days_sdw)
        title = 'Desviación estándar por día (SDw)'
        if excluded_days_sdw > 0:
            title += f'\nDías excluidos: {excluded_days_sdw} de {len(all_days)}'
        ax3.set_title(title)
        
        # Añadir leyenda personalizada
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='skyblue', label='Día incluido'),
            Patch(facecolor='darkblue', label='Valor extremo'),
        ]
        if excluded_days_sdw > 0:
            legend_elements.append(Patch(facecolor='lightcoral', hatch='///', label='Día excluido'))
        legend_elements.append(plt.Line2D([0], [0], color='blue', linestyle='--', 
                                         label=f'SDw = {sdw_details["sd"]:.2f} mg/dL'))
        
        ax3.legend(handles=legend_elements)
        ax3.grid(True, alpha=0.3)
        
        # Rotar etiquetas del eje x para mejor visualización y mostrar menos etiquetas
        if len(daily_stats) > 10:
            # Mostrar solo algunas etiquetas para evitar solapamiento
            n_ticks = min(10, len(daily_stats))
            step = max(1, len(daily_stats) // n_ticks)
            ax3.set_xticks(range(0, len(daily_stats), step))
            ax3.set_xticklabels([daily_stats['date'].astype(str).iloc[i] for i in range(0, len(daily_stats), step)])
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Gráfico 4: Media de glucosa por día (SDdm)
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Obtener datos detallados de SDdm para identificar días excluidos
        sdm_details = self.sd_daily_mean(min_count_threshold)
        
        # Usar los mismos datos diarios calculados anteriormente
        daily_means = daily_stats[['date', 'mean']]
        
        # Identificar días válidos e inválidos
        valid_days_set = set(sdm_details['daily_means'].keys())
        all_days = set(daily_means['date'])
        invalid_days = all_days - valid_days_set
        
        # Crear el gráfico de dispersión para días válidos
        valid_means = daily_means[daily_means['date'].isin(valid_days_set)]
        ax4.scatter(valid_means['date'].astype(str), valid_means['mean'], 
                   alpha=0.7, color='blue', s=30, label='Media diaria (incluida)')
        
        # Crear el gráfico de dispersión para días inválidos (si hay alguno)
        if invalid_days:
            invalid_means = daily_means[daily_means['date'].isin(invalid_days)]
            ax4.scatter(invalid_means['date'].astype(str), invalid_means['mean'], 
                       alpha=0.7, color='red', s=30, marker='x', label='Media diaria (excluida)')
        
        # Añadir líneas de referencia
        overall_mean = valid_means['mean'].mean()  # Solo usar días válidos para la media
        ax4.axhline(overall_mean, color='blue', linestyle='--', label=f'Media: {overall_mean:.2f} mg/dL')
        ax4.axhline(overall_mean + 20, color='blue', linestyle=':', alpha=0.5)
        ax4.axhline(overall_mean - 20, color='blue', linestyle=':', alpha=0.5)
        
        # Añadir línea de tendencia (solo para días válidos)
        if len(valid_means) > 7:
            valid_means_sorted = valid_means.sort_values('date')
            x_numeric = np.arange(len(valid_means_sorted))
            z = np.polyfit(x_numeric, valid_means_sorted['mean'], 1)
            p = np.poly1d(z)
            ax4.plot(valid_means_sorted['date'].astype(str), p(x_numeric), "r--", alpha=0.7)
        
        # Configurar el gráfico
        ax4.set_xlabel('Fecha')
        ax4.set_ylabel('Glucosa (mg/dL)')
        
        # Añadir información sobre días excluidos en el título
        excluded_days = sdm_details['total_days'] - sdm_details['valid_days']
        title = f'Media de glucosa por día (SDdm = {metrics["SDdm"]:.2f} mg/dL)'
        if excluded_days > 0:
            title += f'\nDías excluidos: {excluded_days} de {sdm_details["total_days"]}'
        ax4.set_title(title)
        
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Rotar etiquetas del eje x para mejor visualización y mostrar menos etiquetas
        if len(daily_means) > 10:
            # Mostrar solo algunas etiquetas para evitar solapamiento
            n_ticks = min(10, len(daily_means))
            step = max(1, len(daily_means) // n_ticks)
            ax4.set_xticks(range(0, len(daily_means), step))
            ax4.set_xticklabels([daily_means['date'].astype(str).iloc[i] for i in range(0, len(daily_means), step)])
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Gráfico 5: Componentes de la variabilidad glucémica (todos)
        ax5 = fig.add_subplot(gs[2, :])
        
        # Preparar todos los componentes para el gráfico de barras
        components = []
        for key, value in metrics.items():
            # Incluir solo componentes numéricos (excluir diccionarios y otros objetos)
            if isinstance(value, (int, float)):
                components.append((key, value))
        
        # Ordenar componentes de mayor a menor para mejor visualización
        components.sort(key=lambda x: x[1], reverse=True)
        
        # Crear el gráfico de barras con todos los componentes
        colors = plt.cm.tab20(np.linspace(0, 1, len(components)))
        bars = ax5.bar(
            [comp[0] for comp in components],
            [comp[1] for comp in components],
            color=colors
        )
        
        # Añadir etiquetas de valor encima de cada barra
        for bar in bars:
            height = bar.get_height()
            ax5.text(
                bar.get_x() + bar.get_width()/2.,
                height + 1,
                f'{height:.2f}',
                ha='center', 
                va='bottom',
                fontweight='bold',
                fontsize=9
            )
        
        # Configurar el gráfico
        ax5.set_ylabel('Desviación estándar (mg/dL)')
        ax5.set_title('Componentes de la variabilidad glucémica')
        
        # Ajustar tamaño de fuente en eje x si hay muchos componentes
        if len(components) > 8:
            plt.setp(ax5.xaxis.get_majorticklabels(), fontsize=8)
        
        # Añadir leyenda explicativa
        legend_text = (
            'SDT: DE total\n'
            'SDw: DE dentro del día\n'
            'SDbhh:mm: DE entre días por punto temporal\n'
            'SDdm: DE de medias diarias\n'
            'SDbhh:mm,dm: DE entre días ajustada\n'
            'SDI: DE de interacción\n'
            'Noche: DE segmento nocturno (00:00-08:00)\n'
            'Día: DE segmento diurno (08:00-16:00)\n'
            'Tarde: DE segmento tarde (16:00-00:00)\n'
            'SDws_1h: DE dentro de series de 1 hora\n'
            'SDws_6h: DE dentro de series de 6 horas\n'
            'SDws_24h: DE dentro de series de 24 horas'
        )
        ax5.text(
            1.02, 0.5, legend_text,
            transform=ax5.transAxes,
            fontsize=9,
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )
        
        # Ajustar espaciado entre subplots
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.2, hspace=0.3)
        
        # Mostrar gráfico
        plt.show()

    def plot_variance_components(self):
        """
        Genera un gráfico con tres subplots para visualizar la descomposición de la variabilidad glucémica:
        1. Componentes principales: interdía vs intradía
        2. Desglose de varianza intradía: patrón, interacción, residual
        3. Varianza por segmentos del día: noche (00-08h), día (08-16h), tarde (16-00h)
        """
        # Obtener componentes de varianza
        components = self.variance_components()
        
        # Crear figura con tres subplots
        fig, axs = plt.subplots(1, 3, figsize=(18, 6), dpi=100)
        
        # 1. PRIMER SUBPLOT: INTERDÍA VS INTRADÍA
        # Preparar datos para el gráfico de barras
        labels1 = ['Interdía', 'Intradía']
        values1 = [components['porcentaje_interdía'], components['porcentaje_intradía']]
        colors1 = ['#ff9999', '#66b3ff']
        
        # Crear gráfico de barras
        axs[0].bar(labels1, values1, color=colors1, width=0.6)
        
        # Personalizar gráfico
        axs[0].set_title('Componentes Principales de Varianza', fontweight='bold')
        axs[0].set_ylabel('Porcentaje de varianza total (%)')
        axs[0].grid(axis='y', alpha=0.3)
        
        # Añadir etiquetas con valores
        for i, v in enumerate(values1):
            axs[0].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
        
        # Añadir información de SD
        sd_info = f"SD Total: {components['sd_total']:.2f} mg/dL\n"
        sd_info += f"SD Interdía: {components['sd_interdía']:.2f} mg/dL\n"
        sd_info += f"SD Intradía: {components['sd_intradía']:.2f} mg/dL"
        
        axs[0].text(0.5, -0.15, sd_info, transform=axs[0].transAxes, 
                    ha='center', va='center', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # 2. SEGUNDO SUBPLOT: DESGLOSE DE INTRADÍA
        labels2 = ['Patrón', 'Interacción', 'Residual']
        values2 = [
            components['porcentaje_patrón'], 
            components['porcentaje_interacción'],
            components['porcentaje_residual']
        ]
        colors2 = ['#99ff99', '#ffcc99', '#c2c2f0']
        
        # Crear gráfico de barras
        axs[1].bar(labels2, values2, color=colors2, width=0.6)
        
        # Personalizar gráfico
        axs[1].set_title('Desglose de Varianza Intradía', fontweight='bold')
        axs[1].set_ylabel('Porcentaje de varianza intradía (%)')  # Cambiado de "varianza total" a "varianza intradía"
        axs[1].grid(axis='y', alpha=0.3)
        
        # Añadir etiquetas con valores
        for i, v in enumerate(values2):
            axs[1].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
        
        # Verificar que la suma de porcentajes es cercana a 100%
        sum_intraday_pcts = sum(values2)
        if abs(sum_intraday_pcts - 100) > 0.1:  # Tolerancia de 0.1%
            axs[1].text(0.5, 0.95, f"⚠️ Suma: {sum_intraday_pcts:.1f}%", transform=axs[1].transAxes, 
                        ha='center', color='red', fontweight='bold')
        else:
            axs[1].text(0.5, 0.95, f"Suma: {sum_intraday_pcts:.1f}%", transform=axs[1].transAxes, 
                        ha='center', color='green', fontweight='bold')
        
        # Añadir información de SD
        sd_info2 = f"SD Patrón: {components['sd_patrón']:.2f} mg/dL\n"
        sd_info2 += f"SD Interacción: {components['sd_interacción']:.2f} mg/dL\n"
        sd_info2 += f"SD Residual: {components['sd_residual']:.2f} mg/dL"
        
        axs[1].text(0.5, -0.15, sd_info2, transform=axs[1].transAxes, 
                    ha='center', va='center', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # 3. TERCER SUBPLOT: VARIANZA POR SEGMENTOS DEL DÍA
        segment_vars = components['varianza_segmentos']
        segment_pcts = components['porcentaje_segmentos']
        segment_sds = components['sd_segmentos']
        
        labels3 = list(segment_vars.keys())
        values3 = [segment_pcts[seg] for seg in labels3]
        colors3 = ['#c2c2f0', '#99ff99', '#ffcc99']  # Noche, Día, Tarde
        
        # Crear gráfico de barras
        axs[2].bar(labels3, values3, color=colors3, width=0.6)
        
        # Personalizar gráfico
        axs[2].set_title('Distribución de Varianza Intradía por Segmento', fontweight='bold')
        axs[2].set_ylabel('Porcentaje de varianza intradía (%)')
        axs[2].grid(axis='y', alpha=0.3)
        
        # Añadir etiquetas con valores
        for i, v in enumerate(values3):
            axs[2].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
        
        # Verificar que la suma de porcentajes es cercana a 100%
        sum_segment_pcts = sum(values3)
        if abs(sum_segment_pcts - 100) > 5:
            axs[2].text(0.5, 0.95, f"⚠️ Suma: {sum_segment_pcts:.1f}%", transform=axs[2].transAxes, 
                        ha='center', color='red', fontweight='bold')
        else:
            axs[2].text(0.5, 0.95, f"Suma: {sum_segment_pcts:.1f}%", transform=axs[2].transAxes, 
                        ha='center', color='green', fontweight='bold')
        
        # Añadir información de SD
        sd_info3 = ""
        for seg in labels3:
            sd_info3 += f"SD {seg.capitalize()}: {segment_sds[seg]:.2f} mg/dL\n"
        
        axs[2].text(0.5, -0.15, sd_info3.strip(), transform=axs[2].transAxes, 
                    ha='center', va='center', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Título general
        fig.suptitle('Descomposición Extendida de la Variabilidad Glucémica', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Añadir leyenda explicativa general
        explanation = (
            "• La varianza total se puede descomponer en dos componentes: interdía (entre diferentes días) e intradía (dentro del mismo día).\n"
            "• La varianza intradía se desglosa en: patrón (efecto predecible de la hora), interacción (variabilidad en la consistencia del patrón) y residual.\n"
            "• El tercer gráfico muestra cómo se distribuye la variabilidad intradía entre los segmentos del día: noche (00:00-08:00), día (08:00-16:00) y tarde (16:00-00:00)."
        )
        
        plt.figtext(0.5, 0.01, explanation, wrap=True, horizontalalignment='center', 
                    fontsize=9, bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
        
        # Ajustar layout
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        
        # Mostrar gráfico
        plt.show()
        
        return components

    

    def plot_time_in_range(self, pregnancy: bool = False):
        """
        Genera un gráfico de barras con los tiempos en rango en estilo AGP.
        
        :param pregnancy: Si es True, muestra las métricas específicas para embarazo.
                 Si es False, muestra las métricas estándar y estrictas en dos subplots.
        """
        # Obtener estadísticas según el modo
        if pregnancy:
            stats = self.time_statistics_pregnancy()
            # Definir rangos para embarazo (orden invertido)
            ranges = [
                {'name': 'Bajo', 'range': '< 63 mg/dL', 'value': stats['TBR63'], 'color': '#FF0000'},
                {'name': 'Rango objetivo', 'range': '63 - 140 mg/dL', 'value': stats['TIR_pregnancy'], 'color': '#00CC00'},
                {'name': 'Alto', 'range': '140 - 250 mg/dL', 'value': stats['TAR140'], 'color': '#FFCC00'},
                {'name': 'Muy alto', 'range': '> 250 mg/dL', 'value': stats['TAR250'], 'color': '#FF0000'}
            ]
            
            # Crear figura con un solo subplot para embarazo
            fig, ax = plt.subplots(figsize=(6, 10))
            axes = [ax]  # Lista con un solo eje para reutilizar código
            subplot_titles = [f"TIEMPO EN RANGOS (EMBARAZO) - Datos: {stats['%Data']}%"]
            
        else:
            stats = self.time_statistics()
            # Definir rangos estándar (orden invertido)
            ranges_standard = [
                {'name': 'Muy bajo', 'range': '< 54 mg/dL', 'value': stats['TBR55'], 'color': '#990000'},
                {'name': 'Bajo', 'range': '54 - 69 mg/dL', 'value': stats['TBR70'], 'color': '#FF0000'},
                {'name': 'Rango objetivo', 'range': '70 - 180 mg/dL', 'value': stats['TIR'], 'color': '#00CC00'},
                {'name': 'Alto', 'range': '181 - 250 mg/dL', 'value': stats['TAR180'], 'color': '#FFCC00'},
                {'name': 'Muy alto', 'range': '> 250 mg/dL', 'value': stats['TAR250'], 'color': '#FF0000'}
            ]
            
            # Definir rangos estrictos (orden invertido)
            ranges_strict = [
                {'name': 'Muy bajo', 'range': '< 54 mg/dL', 'value': stats['TBR55'], 'color': '#990000'},
                {'name': 'Bajo', 'range': '54 - 69 mg/dL', 'value': stats['TBR70'], 'color': '#FF0000'},
                {'name': 'Rango objetivo', 'range': '70 - 140 mg/dL', 'value': stats['TIR_tight'], 'color': '#00CC00'},
                {'name': 'Alto', 'range': '141 - 250 mg/dL', 'value': stats['TAR140'], 'color': '#FFCC00'},
                {'name': 'Muy alto', 'range': '> 250 mg/dL', 'value': stats['TAR250'], 'color': '#FF0000'}
            ]
            
            # Crear figura con dos subplots para criterios estándar y estrictos
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 10))
            axes = [ax1, ax2]  # Lista con dos ejes para reutilizar código
            ranges = [ranges_standard, ranges_strict]  # Lista con ambos conjuntos de rangos
            subplot_titles = [
                f"TIEMPO EN RANGOS (ESTÁNDAR) - Datos: {stats['%Data']}%", 
                f"TIEMPO EN RANGOS (ESTRICTO) - Datos: {stats['%Data']}%"
            ]
        
        # Iterar sobre cada subplot
        for i, ax in enumerate(axes):
            # Configurar el eje Y
            ax.set_ylim(0, 100)
            
            # Eliminar ejes y ticks
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Crear un título con recuadro negro
            title_box = dict(
                boxstyle="square,pad=0.5",
                facecolor="black",
                edgecolor="black",
            )
            ax.text(0.5, -0.05, subplot_titles[i], 
                    transform=ax.transAxes,
                    color="white",
                    fontsize=12,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    bbox=title_box)
            
            # Variables para seguimiento de posición y ancho de barra
            current_position = 0
            bar_width = 0.4
            bar_x_pos = 0.5  # Posición central de la barra
            
            # Seleccionar el conjunto de rangos adecuado
            current_ranges = ranges[i] if not pregnancy else ranges
            
            # Dibujar cada segmento de la barra
            for j, r in enumerate(current_ranges):
                # Altura del segmento
                height = r['value']
                
                # Dibujar el segmento de la barra
                bar = ax.bar(bar_x_pos, height, bottom=current_position, width=bar_width, 
                        color=r['color'], edgecolor='white', linewidth=1)
                
                # Añadir líneas horizontales de separación entre segmentos
                if j > 0:
                    ax.axhline(y=current_position, color='white', linestyle='-', linewidth=1, 
                           xmin=0.3, xmax=0.7)
                
                # Posición para la etiqueta del rango (izquierda)
                ax.text(0.05, current_position + height/2, f"{r['range']}", 
                        ha='left', va='center', fontsize=10)
                
                # Posición para el porcentaje (derecha)
                ax.text(0.95, current_position + height/2, f"{r['value']:.0f}%", 
                        ha='right', va='center', fontsize=10, fontweight='bold')
                
                # Actualizar posición para el siguiente segmento
                current_position += height
            
            # Añadir etiquetas de nombre debajo de la barra
            y_positions = []
            current_position = 0
            
            for j, r in enumerate(current_ranges):
                height = r['value']
                mid_point = current_position + height/2
                y_positions.append(mid_point)
                current_position += height
            
            # Añadir las etiquetas de nombre debajo de la barra
            for j, (y_pos, r) in enumerate(zip(y_positions, current_ranges)):
                # Calcular posición vertical para la etiqueta
                ax.text(bar_x_pos, y_pos, r['name'], 
                        ha='center', va='center', fontsize=10, fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Ajustar para dejar espacio para el título
        plt.show()

    def plot_conga_profile(self, max_hours: int = 24, step: int = 1):
        """
        Genera un gráfico que muestra los valores de CONGA calculados para diferentes
        intervalos de tiempo, desde 1 hasta max_hours horas.
        
        :param max_hours: Número máximo de horas para calcular CONGA (por defecto 24)
        :param step: Incremento en horas entre cálculos consecutivos (por defecto 1)
        """
        # Crear listas para almacenar resultados
        hours_list = list(range(1, max_hours + 1, step))
        conga_values = []
        valid_percentages = []
        
        # Calcular CONGA para cada intervalo de horas
        for hours in hours_list:
            conga_result = self.CONGA(hours=hours)
            
            # Verificar si el cálculo fue exitoso
            if conga_result['value'] is not None:
                conga_values.append(conga_result['value'])
                valid_percentages.append(conga_result['percent_valid'])
            else:
                conga_values.append(np.nan)
                valid_percentages.append(0)
        
        # Crear figura con dos subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Gráfico principal: valores de CONGA
        line1 = ax1.plot(hours_list, conga_values, 'o-', color='#3366CC', linewidth=2, markersize=8)
        ax1.set_xlabel('Intervalo de tiempo (horas)', fontsize=12)
        ax1.set_ylabel('CONGA (mg/dL)', fontsize=12, color='#3366CC')
        ax1.tick_params(axis='y', labelcolor='#3366CC')
        ax1.grid(True, alpha=0.3)
        
        # Añadir etiquetas de valor en puntos clave
        for i, (x, y) in enumerate(zip(hours_list, conga_values)):
            if i % 4 == 0 or i == len(hours_list) - 1:  # Etiquetar cada 4 puntos y el último
                if not np.isnan(y):
                    ax1.annotate(f'{y:.1f}', 
                                xy=(x, y), 
                                xytext=(0, 10),
                                textcoords='offset points',
                                ha='center',
                                fontweight='bold',
                                fontsize=9)
        
        # Gráfico secundario: porcentaje de comparaciones válidas
        ax2.bar(hours_list, valid_percentages, color='#FF9966', alpha=0.7)
        ax2.set_xlabel('Intervalo de tiempo (horas)', fontsize=12)
        ax2.set_ylabel('% Comparaciones válidas', fontsize=10)
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        
        # Añadir línea de referencia en 80%
        ax2.axhline(y=80, color='red', linestyle='--', alpha=0.7)
        ax2.text(max_hours, 82, '80%', color='red', ha='right', fontsize=9)
        
        # Título general
        plt.suptitle('Perfil de CONGA a diferentes intervalos de tiempo', fontsize=14, fontweight='bold')
        
        # Añadir explicación
        explanation = (
            "CONGA (Continuous Overlapping Net Glycemic Action) mide la variabilidad intradiaria\n"
            "calculando la desviación estándar de las diferencias entre valores actuales y valores\n"
            "de n horas antes. Valores más altos indican mayor variabilidad glucémica."
        )
        plt.figtext(0.5, 0.01, explanation, ha='center', fontsize=9, 
                    bbox=dict(facecolor='#f0f0f0', alpha=0.5, boxstyle='round'))
        
        # Ajustar diseño
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.show()
        
        # Devolver los datos para posible uso posterior
        return {'hours': hours_list, 'conga_values': conga_values, 'valid_percentages': valid_percentages}

    def plot_m_value_curve(self):
        """
        Genera un gráfico que muestra la curva del valor M para diferentes niveles de glucosa
        y marca el valor M del paciente.
        """
        # Generar valores de glucosa de 40 a 500
        glucose_range = np.linspace(40, 500, 1000)
        
        # Calcular el valor M para cada punto de la curva
        m_values = np.abs(10 * np.log10(glucose_range/120))**3
        
        # Obtener el valor M del paciente y la media de glucosa
        patient_m_value = self.M_Value()
        patient_mean_glucose = self.data['glucose'].mean()
        
        # Crear la figura
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Graficar la curva del valor M
        ax.plot(glucose_range, m_values, 'b-', label='Curva del Valor M', linewidth=2)
        
        # Graficar el punto del valor M del paciente
        ax.scatter(patient_mean_glucose, patient_m_value, 
                  color='red', s=100, marker='*',
                  label=f'Valor M del paciente: {patient_m_value:.1f}')
        
        # Añadir líneas de referencia
        ax.axvline(x=120, color='green', linestyle='--', alpha=0.5, 
                   label='Valor de referencia (120 mg/dL)')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # Añadir zonas de interpretación
        ax.axhspan(0, 18, color='green', alpha=0.1, label='Control bueno (<18)')
        ax.axhspan(18, 30, color='yellow', alpha=0.1, label='Control regular (18-30)')
        ax.axhspan(30, m_values.max(), color='red', alpha=0.1, label='Control pobre (>30)')
        
        # Configurar ejes
        ax.set_xlabel('Nivel de Glucosa (mg/dL)', fontsize=12)
        ax.set_ylabel('Valor M', fontsize=12)
        ax.set_title('Curva del Valor M y Valor del Paciente', fontsize=14, pad=20)
        
        # Añadir grid
        ax.grid(True, alpha=0.3)
        
        # Configurar límites
        ax.set_xlim(40, 500)
        ax.set_ylim(0, min(max(m_values.max(), patient_m_value * 1.2), 200))
        
        # Añadir leyenda
        ax.legend(loc='upper left')
        
        # Determinar la interpretación del valor M
        if patient_m_value < 18:
            interpretation = "Control bueno"
        elif patient_m_value < 30:
            interpretation = "Control regular"
        else:
            interpretation = "Control pobre"
        
        # Añadir texto explicativo
        explanation = (
            "El Valor M penaliza más las hipoglucemias que las hiperglucemias.\n"
            "La fórmula utilizada es: M = |10 × log₁₀(BS/120)|³\n"
            f"Interpretación del valor {patient_m_value:.1f}: {interpretation}"
        )
        plt.figtext(0.5, 0.02, explanation, ha='center', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
        
        plt.tight_layout()
        plt.show()

    def plot_gri(self):
        """
        Genera un gráfico de dispersión que muestra la relación entre los componentes
        de hipoglucemia y hiperglucemia del GRI, con zonas de riesgo.
        
        El gráfico incluye:
        - Zonas de riesgo (A-E)
        - Punto que representa la posición del paciente
        - Fórmula del GRI
        - Componentes del GRI del paciente
        """
        # Obtener los datos del GRI del paciente
        gri_data = self.GRI()
        
        # Crear figura
        fig, ax = plt.subplots(figsize=(8, 12))
        
        # Definir los límites del gráfico
        x_max = 30
        y_max = 60
        
        # Crear las zonas de riesgo
        # Zona E (81-100)
        x_e = np.linspace(0, x_max, 100)
        y_e = (100 - 3.0*x_e) / 1.6
        ax.fill_between(x_e, y_e, y_max, color='#e6b3b3', alpha=0.5, label='Zona E (81-100)')
        
        # Zona D (61-80)
        y_d = (80 - 3.0*x_e) / 1.6
        ax.fill_between(x_e, y_d, y_e, color='#ffb3b3', alpha=0.5, label='Zona D (61-80)')
        
        # Zona C (41-60)
        y_c = (60 - 3.0*x_e) / 1.6
        ax.fill_between(x_e, y_c, y_d, color='#ffd9b3', alpha=0.5, label='Zona C (41-60)')
        
        # Zona B (21-40)
        y_b = (40 - 3.0*x_e) / 1.6
        ax.fill_between(x_e, y_b, y_c, color='#ffff99', alpha=0.5, label='Zona B (21-40)')
        
        # Zona A (0-20)
        y_a = (20 - 3.0*x_e) / 1.6
        ax.fill_between(x_e, 0, y_b, color='#b3ffb3', alpha=0.5, label='Zona A (0-20)')
        
        # Plotear el punto del paciente
        hypo_comp = gri_data['components']['VLow'] + gri_data['components']['Low']
        hyper_comp = gri_data['components']['VHigh'] + gri_data['components']['High']
        ax.scatter(hypo_comp, hyper_comp, color='black', s=100, marker='*', 
                  label=f'Paciente (GRI={gri_data["GRI"]})')
        
        # Añadir texto con los componentes del paciente
        info_text = (
            f'T1D MDI\n'
            f'GRI = {gri_data["GRI"]}\n'
            f'VLow = {gri_data["components"]["VLow"]}%\n'
            f'Low = {gri_data["components"]["Low"]}%\n'
            f'High = {gri_data["components"]["High"]}%\n'
            f'VHigh = {gri_data["components"]["VHigh"]}%\n'
            f'TIR = {gri_data["derived_metrics"]["TIR"]}%'
        )
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
        
        # Añadir la fórmula del GRI
        formula_text = (
            'GRI = (3.0 x VLow) + (2.4 x Low) + (1.6 x VHigh) + (0.8 X High)\n'
            'GRI = 3.0 x Hypo Component + 1.6 x Hyper Component'
        )
        ax.text(0.98, 0.98, formula_text, transform=ax.transAxes,
                horizontalalignment='right', verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8))
        
        # Configurar el gráfico
        ax.set_xlabel('Hypo Component (%)')
        ax.set_ylabel('Hyperglycemia Component (%)')
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='center right')
        
        plt.tight_layout()
        plt.show()

    def plot_gri_array(self, gri_dict: dict):
        """
        Genera un único gráfico de GRI con múltiples puntos para comparar diferentes
        pacientes o períodos.
        
        :param gri_dict: Diccionario donde las claves son los identificadores (ej: 'Paciente 1')
                        y los valores son los resultados del método GRI()
        """
        # Crear figura
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Definir los límites del gráfico
        x_max = 30
        y_max = 60
        
        # Crear las zonas de riesgo
        x_e = np.linspace(0, x_max, 100)
        
        # Zona E (81-100)
        y_e = (100 - 3.0*x_e) / 1.6
        ax.fill_between(x_e, y_e, y_max, color='#e6b3b3', alpha=0.5, label='Zona E (81-100)')
        
        # Zona D (61-80)
        y_d = (80 - 3.0*x_e) / 1.6
        ax.fill_between(x_e, y_d, y_e, color='#ffb3b3', alpha=0.5, label='Zona D (61-80)')
        
        # Zona C (41-60)
        y_c = (60 - 3.0*x_e) / 1.6
        ax.fill_between(x_e, y_c, y_d, color='#ffd9b3', alpha=0.5, label='Zona C (41-60)')
        
        # Zona B (21-40)
        y_b = (40 - 3.0*x_e) / 1.6
        ax.fill_between(x_e, y_b, y_c, color='#ffff99', alpha=0.5, label='Zona B (21-40)')
        
        # Zona A (0-20)
        ax.fill_between(x_e, 0, y_b, color='#b3ffb3', alpha=0.5, label='Zona A (0-20)')
        
        # Definir marcadores y colores para los diferentes puntos
        markers = ['*', 'o', 's', 'D', '^', 'v', '<', '>', 'p', 'h']
        colors = plt.cm.tab10(np.linspace(0, 1, len(gri_dict)))
        
        
        # Plotear los puntos para cada paciente/período
        for idx, (identifier, gri_data) in enumerate(gri_dict.items()):
            hypo_comp = gri_data['components']['VLow'] + gri_data['components']['Low']
            hyper_comp = gri_data['components']['VHigh'] + gri_data['components']['High']
            
            # Plotear punto de forma más simple
            ax.scatter(hypo_comp, hyper_comp, 
                    color=colors[idx],
                    marker='o',  # Usar círculos simples
                    s=100,      # Tamaño más pequeño
                    alpha=0.7)  # Algo de transparencia
            
            
        
        # Añadir la fórmula del GRI
        formula_text = (
            r'$GRI = (3.0 \times VLow) + (2.4 \times Low) + (1.6 \times VHigh) + (0.8 \times High)$' + '\n' +
            r'$GRI = 3.0 \times \text{Hypo Component} + 1.6 \times \text{Hyper Component}$'
        )
        plt.text(0.5, 1.05, formula_text, 
                horizontalalignment='center',
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8))
        
        # Configurar el gráfico
        ax.set_xlabel('Hypo Component (%)')
        ax.set_ylabel('Hyperglycemia Component (%)')
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)
        ax.grid(True, alpha=0.3)
        
        # Ajustar la leyenda
        ax.legend(bbox_to_anchor=(1.02, 0.8), loc='upper left')
        
        # Ajustar el layout para acomodar la tabla y la leyenda
        plt.subplots_adjust(right=0.7)
        
        plt.show()