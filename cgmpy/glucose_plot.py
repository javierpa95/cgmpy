from .glucose_metrics import GlucoseMetrics
import matplotlib.pyplot as plt
from typing import Union
import seaborn as sns
import pandas as pd
import numpy as np
import datetime

class GlucosePlot(GlucoseMetrics):
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
        
        :param min_count_threshold: Umbral para considerar una marca temporal como válida
                                   (proporción de la mediana de conteos). Por defecto 0.5 (50%).
        :param filter_outliers: Si es True, filtra las marcas temporales con pocos datos
                               antes de calcular SDbhh:mm.
        :param agrupar_por_intervalos: Si es True, agrupa los datos en intervalos regulares de tiempo.
        :param intervalo_minutos: Tamaño del intervalo en minutos para la agrupación (por defecto 5 min).
        """
        # Calcular SDbhh:mm y obtener estadísticas
        sd_result = self.sd_same_timepoint(min_count_threshold, filter_outliers, 
                                          agrupar_por_intervalos, intervalo_minutos)
        sd_value = sd_result['sd']
        
        # Preparar datos para graficar
        time_decimal = []
        values = []  # Valores de SD para cada marca temporal
        means = []   # Valores medios para cada marca temporal
        counts = []
        
        # Obtener los datos por marca temporal
        for time_str in sd_result['sd_por_marca'].keys():
            h, m = map(int, time_str.split(':'))
            time_decimal.append(h + m/60.0)
            
            # Obtener SD y media para esta marca temporal
            values.append(sd_result['sd_por_marca'][time_str])
            means.append(sd_result['valores_por_marca'][time_str])
            counts.append(sd_result['conteo_por_marca'][time_str])
        
        # Crear figura y ejes
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [1, 1]})
        
        # Gráfico 1: Desviación estándar por hora del día
        norm = plt.Normalize(min(counts), max(counts))
        scatter1 = ax1.scatter(time_decimal, values, s=[max(20, c/2) for c in counts], 
                             c=counts, cmap='viridis', alpha=0.7, edgecolors='k', linewidths=0.5)
        
        # Añadir línea de tendencia para SD
        ax1.plot(time_decimal, values, 'r-', alpha=0.5, label=f'SDbhh:mm = {sd_value:.2f} mg/dL')
        
        # Gráfico 2: Valor medio por hora del día
        scatter2 = ax2.scatter(time_decimal, means, s=[max(20, c/2) for c in counts], 
                             c=counts, cmap='viridis', alpha=0.7, edgecolors='k', linewidths=0.5)
        
        # Añadir línea de tendencia para medias
        ax2.plot(time_decimal, means, 'b-', alpha=0.5, label=f'Media = {sd_result["mean"]:.2f} mg/dL')
        
        # Marcar puntos con pocos datos en ambos gráficos
        for time_str, info in sd_result['marcas_con_pocos_datos'].items():
            # Buscar el valor de SD para esta marca temporal
            if time_str in sd_result['sd_por_marca']:
                sd_value_point = sd_result['sd_por_marca'][time_str]
                mean_value_point = sd_result['valores_por_marca'][time_str]
                
                # Anotar en el gráfico de SD
                ax1.annotate(f"{time_str}\n(n={info['conteo']})", 
                           (info['hora_decimal'], sd_value_point),
                           textcoords="offset points", 
                           xytext=(0,10), 
                           ha='center',
                           fontsize=8,
                           bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))
                
                # Anotar en el gráfico de medias
            ax2.annotate(f"{time_str}\n(n={info['conteo']})", 
                           (info['hora_decimal'], mean_value_point),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center',
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))
        
        # Añadir descripción con información sobre SDbhh:mm
        descripcion = f"SDbhh:mm = {sd_value:.2f} mg/dL\nMedia = {sd_result['mean']:.2f} mg/dL"
        
        if agrupar_por_intervalos:
            descripcion += f"\nDatos agrupados en intervalos de {intervalo_minutos} minutos."
        
        if filter_outliers and sd_result['marcas_con_pocos_datos']:
            descripcion += f"\nSe han filtrado {len(sd_result['marcas_con_pocos_datos'])} marcas temporales con menos de {sd_result['umbral_conteo']:.1f} datos."
        
        ax1.text(0.02, 0.89, descripcion, transform=ax1.transAxes, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        # Añadir barra de color para mostrar el número de datos
        cbar = plt.colorbar(scatter1, ax=ax1)
        cbar.set_label('Número de días con datos')
        
        # Configuración de ejes y etiquetas para el gráfico de SD
        ax1.set_ylabel('Desviación Estándar (mg/dL)', fontsize=12)
        
        # Título con información sobre la agrupación
        titulo = 'Desviación Estándar Entre Días por Hora del Día (SDbhh:mm)'
        if agrupar_por_intervalos:
            titulo += f' (Intervalos de {intervalo_minutos} min)'
        ax1.set_title(titulo, fontsize=16, fontweight='bold')
        
        # Configuración de la leyenda
        ax1.legend(loc="upper right", fontsize=10)
        
        # Configuración de la cuadrícula
        ax1.grid(True, linestyle=':', alpha=0.6)
        
        # Configuración de los ticks del eje x
        ax1.set_xticks(range(0, 25, 3))
        ax1.set_xticklabels([f'{h:02d}:00' for h in range(0, 25, 3)])
        ax1.set_xlim(0, 24)
        
        # Configuración para el gráfico de medias
        ax2.set_xlabel('Hora del Día', fontsize=12)
        ax2.set_ylabel('Nivel de Glucosa (mg/dL)', fontsize=12)
        ax2.set_title('Valor Medio por Hora del Día', fontsize=14)
        ax2.legend(loc="upper right", fontsize=10)
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.set_xticks(range(0, 25, 3))
        ax2.set_xticklabels([f'{h:02d}:00' for h in range(0, 25, 3)])
        ax2.set_xlim(0, 24)
        ax2.set_ylim(0, 400)
        
        # Añadir barra de color para el segundo gráfico
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label('Número de días con datos')
        
        # Ajustes finales
        plt.tight_layout()
        plt.show()

    def plot_sd_within_day(self, min_count_threshold: float = 0.5):
        """
        Genera un gráfico que muestra la desviación estándar dentro del día (SDw) para cada día
        y la media de todas estas SD.
        
        Este gráfico visualiza cómo varía la desviación estándar de los niveles de glucosa
        dentro de cada día individual, permitiendo identificar días con mayor o menor variabilidad.
        
        :param min_count_threshold: Umbral para considerar un día como válido
                                   (proporción de la mediana de conteos). Por defecto 0.5 (50%).
        """
        # Calcular la SD dentro del día para cada día individual
        daily_stats = self.data.groupby(self.data['time'].dt.date)['glucose'].agg(['std', 'mean', 'count'])
        
        # Identificar días con pocos datos
        count_threshold = daily_stats['count'].median() * min_count_threshold
        
        # Filtrar días con suficientes datos para el cálculo principal
        filtered_stats = daily_stats[daily_stats['count'] >= count_threshold]
        
        # Días con pocos datos (para mostrarlos de forma diferente)
        low_count_days = daily_stats[daily_stats['count'] < count_threshold]
        
        # Calcular la media de las SD diarias (SDw) solo con los días filtrados
        sdw = filtered_stats['std'].mean()
        
        # Preparar datos para graficar (días con suficientes datos)
        dates = [date.strftime('%d-%m-%Y') for date in filtered_stats.index]
        sd_values = filtered_stats['std'].values
        mean_values = filtered_stats['mean'].values
        counts = filtered_stats['count'].values
        
        # Preparar datos para días con pocos datos
        low_dates = [date.strftime('%d-%m-%Y') for date in low_count_days.index]
        low_sd_values = low_count_days['std'].values
        low_mean_values = low_count_days['mean'].values
        low_counts = low_count_days['count'].values
        
        # Crear figura y ejes
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [1, 1]})
        
        # Gráfico 1: Desviación estándar por día (días con suficientes datos)
        norm = plt.Normalize(min(counts) if counts.size > 0 else 0, max(counts) if counts.size > 0 else 100)
        scatter1 = ax1.scatter(range(len(dates)), sd_values, s=[max(30, c/5) for c in counts], 
                             c=counts, cmap='viridis', alpha=0.7, edgecolors='k', linewidths=0.5)
        
        # Añadir días con pocos datos con un estilo diferente
        if len(low_dates) > 0:
            low_scatter1 = ax1.scatter(
                [len(dates) + i for i in range(len(low_dates))], 
                low_sd_values, 
                s=[max(20, c/5) for c in low_counts], 
                c='gray', alpha=0.5, edgecolors='red', linewidths=1.0, marker='x'
            )
        
        # Añadir línea horizontal para la media de SD (SDw)
        ax1.axhline(y=sdw, color='r', linestyle='--', linewidth=1.5, 
                   label=f'SDw = {sdw:.2f} mg/dL')
        
        # Gráfico 2: Valor medio de glucosa por día
        scatter2 = ax2.scatter(range(len(dates)), mean_values, s=[max(30, c/5) for c in counts], 
                             c=counts, cmap='viridis', alpha=0.7, edgecolors='k', linewidths=0.5)
        
        # Añadir días con pocos datos al gráfico de medias
        if len(low_dates) > 0:
            low_scatter2 = ax2.scatter(
                [len(dates) + i for i in range(len(low_dates))], 
                low_mean_values, 
                s=[max(20, c/5) for c in low_counts], 
                c='gray', alpha=0.5, edgecolors='red', linewidths=1.0, marker='x'
            )
        
        # Añadir línea horizontal para la media global (solo de días con suficientes datos)
        mean_global = filtered_stats['mean'].mean() if not filtered_stats.empty else self.mean()
        ax2.axhline(y=mean_global, color='b', linestyle='--', linewidth=1.5, 
                   label=f'Media global = {mean_global:.2f} mg/dL')
        
        # Añadir etiquetas para días con valores extremos en el gráfico de SD
        if len(dates) > 0:
            # Identificar los 3 días con mayor SD y los 3 con menor SD
            top_indices = np.argsort(sd_values)[-3:] if len(sd_values) >= 3 else np.argsort(sd_values)
            bottom_indices = np.argsort(sd_values)[:3] if len(sd_values) >= 3 else []
            
            for idx in np.concatenate([top_indices, bottom_indices]):
                ax1.annotate(f"{dates[idx]}\n(n={counts[idx]})", 
                           (idx, sd_values[idx]),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center',
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))
        
        # Añadir etiquetas para todos los días con pocos datos
        for i, idx in enumerate(range(len(low_dates))):
            pos = len(dates) + idx
            ax1.annotate(f"{low_dates[idx]}\n(n={low_counts[idx]})", 
                       (pos, low_sd_values[idx]),
                       textcoords="offset points", 
                       xytext=(0,10), 
                       ha='center',
                       fontsize=8,
                       bbox=dict(boxstyle="round,pad=0.3", fc="pink", alpha=0.7))
        
        # Añadir descripción con información sobre SDw
        descripcion = (f"SDw = {sdw:.2f} mg/dL (media de las SD diarias)\n"
                      f"Días analizados: {len(dates)} (filtrados) + {len(low_dates)} (pocos datos)\n"
                      f"Umbral de filtrado: {count_threshold:.0f} mediciones/día\n"
                      f"Rango de SD: {min(sd_values):.2f} - {max(sd_values):.2f} mg/dL\n"
                      f"Media global: {mean_global:.2f} mg/dL")
        
        ax1.text(0.02, 0.89, descripcion, transform=ax1.transAxes, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        # Añadir barra de color para mostrar el número de datos
        if len(dates) > 0:
            cbar = plt.colorbar(scatter1, ax=ax1)
            cbar.set_label('Número de mediciones por día')
        
        # Configuración de ejes y etiquetas para el gráfico de SD
        ax1.set_ylabel('Desviación Estándar (mg/dL)', fontsize=12)
        ax1.set_title('Desviación Estándar Dentro del Día (SDw) por Día', fontsize=16, fontweight='bold')
        ax1.legend(loc="upper right", fontsize=10)
        ax1.grid(True, linestyle=':', alpha=0.6)
        
        # Configurar el eje x para mostrar fechas
        all_dates = dates + low_dates
        if len(all_dates) > 20:
            # Si hay muchos días, mostrar solo algunos
            step = max(1, len(all_dates) // 10)
            ticks = list(range(0, len(dates), step))
            if len(low_dates) > 0:
                ticks += [len(dates) + i for i in range(0, len(low_dates), max(1, len(low_dates) // 3))]
            ax1.set_xticks(ticks)
            ax1.set_xticklabels([all_dates[i] if i < len(all_dates) else "" for i in ticks], rotation=45, ha='right')
        else:
            ax1.set_xticks(range(len(all_dates)))
            ax1.set_xticklabels(all_dates, rotation=45, ha='right')
        
        # Configuración para el gráfico de medias
        ax2.set_xlabel('Fecha', fontsize=12)
        ax2.set_ylabel('Nivel de Glucosa (mg/dL)', fontsize=12)
        ax2.set_title('Valor Medio de Glucosa por Día', fontsize=14)
        ax2.legend(loc="upper right", fontsize=10)
        ax2.grid(True, linestyle=':', alpha=0.6)
        
        # Configurar el eje x para mostrar fechas (igual que en el gráfico 1)
        if len(all_dates) > 20:
            step = max(1, len(all_dates) // 10)
            ticks = list(range(0, len(dates), step))
            if len(low_dates) > 0:
                ticks += [len(dates) + i for i in range(0, len(low_dates), max(1, len(low_dates) // 3))]
            ax2.set_xticks(ticks)
            ax2.set_xticklabels([all_dates[i] if i < len(all_dates) else "" for i in ticks], rotation=45, ha='right')
        else:
            ax2.set_xticks(range(len(all_dates)))
            ax2.set_xticklabels(all_dates, rotation=45, ha='right')
        
        # Añadir barra de color para el segundo gráfico
        if len(dates) > 0:
            cbar2 = plt.colorbar(scatter2, ax=ax2)
            cbar2.set_label('Número de mediciones por día')
        
        # Ajustes finales
        plt.tight_layout()
        plt.show()

    def plot_sd_daily_mean(self, min_count_threshold: float = 0.5):
        """
        Genera un gráfico que muestra la desviación estándar de las medias diarias (SDdm).
        
        Este gráfico visualiza la variabilidad entre los promedios diarios de glucosa,
        mostrando qué tan consistentes son los niveles medios de glucosa de un día a otro.
        
        :param min_count_threshold: Umbral para considerar un día como válido
                                   (proporción de la mediana de conteos). Por defecto 0.5 (50%).
        """
        # Calcular estadísticas para cada día
        daily_stats = self.data.groupby(self.data['time'].dt.date)['glucose'].agg(['mean', 'count'])
        
        # Si no hay datos, mostrar mensaje y salir
        if daily_stats.empty:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "No hay datos suficientes para calcular SDdm", 
                    ha='center', va='center', fontsize=14)
            plt.tight_layout()
            plt.show()
            return
        
        # Identificar días con pocos datos
        count_threshold = daily_stats['count'].median() * min_count_threshold
        
        # Filtrar días con suficientes datos para el cálculo principal
        filtered_stats = daily_stats[daily_stats['count'] >= count_threshold]
        low_count_days = daily_stats[daily_stats['count'] < count_threshold]
        
        # Si después de filtrar no quedan días, usar todos los datos
        if filtered_stats.empty:
            filtered_stats = daily_stats
            advertencia = "No hay días con suficientes datos, se usaron todos los días disponibles"
        else:
            advertencia = None
        
        # Calcular SDdm con los días filtrados
        sdm = filtered_stats['mean'].std()
        mean_global = filtered_stats['mean'].mean()
        
        # Preparar datos para graficar
        dates = [date.strftime('%d-%m-%Y') for date in filtered_stats.index]
        mean_values = filtered_stats['mean'].values
        counts = filtered_stats['count'].values
        
        # Preparar datos para días con pocos datos
        low_dates = [date.strftime('%d-%m-%Y') for date in low_count_days.index]
        low_mean_values = low_count_days['mean'].values
        low_counts = low_count_days['count'].values
        
        # Crear figura
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Graficar medias diarias (días con suficientes datos)
        scatter = ax.scatter(range(len(dates)), mean_values, s=[max(30, c/5) for c in counts], 
                            c=counts, cmap='viridis', alpha=0.7, edgecolors='k', linewidths=0.5)
        
        # Añadir días con pocos datos con un estilo diferente
        if len(low_dates) > 0:
            low_scatter = ax.scatter(
                [len(dates) + i for i in range(len(low_dates))], 
                low_mean_values, 
                s=[max(20, c/5) for c in low_counts], 
                c='gray', alpha=0.5, edgecolors='red', linewidths=1.0, marker='x'
            )
        
        # Añadir línea horizontal para la media global
        ax.axhline(y=mean_global, color='b', linestyle='--', linewidth=1.5, 
                  label=f'Media global = {mean_global:.2f} mg/dL')
        
        # Añadir líneas horizontales para media ± SDdm
        ax.axhline(y=mean_global + sdm, color='r', linestyle=':', linewidth=1.0, 
                  label=f'Media + SDdm = {mean_global + sdm:.2f} mg/dL')
        ax.axhline(y=mean_global - sdm, color='r', linestyle=':', linewidth=1.0, 
                  label=f'Media - SDdm = {mean_global - sdm:.2f} mg/dL')
        
        # Sombrear el área entre media ± SDdm
        ax.fill_between(
            [-1, len(dates) + len(low_dates)], 
            mean_global - sdm, 
            mean_global + sdm, 
            color='red', alpha=0.1
        )
        
        # Añadir etiquetas para días con valores extremos
        if len(dates) > 0:
            # Identificar los 3 días con mayor media y los 3 con menor media
            top_indices = np.argsort(mean_values)[-3:] if len(mean_values) >= 3 else np.argsort(mean_values)
            bottom_indices = np.argsort(mean_values)[:3] if len(mean_values) >= 3 else []
            
            for idx in np.concatenate([top_indices, bottom_indices]):
                ax.annotate(f"{dates[idx]}\n(n={counts[idx]})", 
                          (idx, mean_values[idx]),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center',
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7))
        
        # Añadir etiquetas para todos los días con pocos datos
        for i, idx in enumerate(range(len(low_dates))):
            pos = len(dates) + idx
            ax.annotate(f"{low_dates[idx]}\n(n={low_counts[idx]})", 
                       (pos, low_mean_values[idx]),
                       textcoords="offset points", 
                       xytext=(0,10), 
                       ha='center',
                       fontsize=8,
                       bbox=dict(boxstyle="round,pad=0.3", fc="pink", alpha=0.7))
        
        # Añadir descripción con información sobre SDdm
        descripcion = (f"SDdm = {sdm:.2f} mg/dL (SD de las medias diarias)\n"
                      f"Días analizados: {len(dates)} (filtrados) + {len(low_dates)} (pocos datos)\n"
                      f"Umbral de filtrado: {count_threshold:.0f} mediciones/día\n"
                      f"Rango de medias: {min(mean_values):.2f} - {max(mean_values):.2f} mg/dL\n"
                      f"Media global: {mean_global:.2f} mg/dL")
        
        if advertencia:
            descripcion += f"\n⚠️ {advertencia}"
        
        ax.text(0.02, 0.95, descripcion, transform=ax.transAxes, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        # Añadir barra de color para mostrar el número de datos
        if len(dates) > 0:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Número de mediciones por día')
        
        # Configuración de ejes y etiquetas
        ax.set_ylabel('Nivel de Glucosa (mg/dL)', fontsize=12)
        ax.set_title('Desviación Estándar de Medias Diarias (SDdm)', fontsize=16, fontweight='bold')
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Configurar el eje x para mostrar fechas
        all_dates = dates + low_dates
        if len(all_dates) > 20:
            # Si hay muchos días, mostrar solo algunos
            step = max(1, len(all_dates) // 10)
            ticks = list(range(0, len(dates), step))
            if len(low_dates) > 0:
                ticks += [len(dates) + i for i in range(0, len(low_dates), max(1, len(low_dates) // 3))]
            ax.set_xticks(ticks)
            ax.set_xticklabels([all_dates[i] if i < len(all_dates) else "" for i in ticks], rotation=45, ha='right')
        else:
            ax.set_xticks(range(len(all_dates)))
            ax.set_xticklabels(all_dates, rotation=45, ha='right')
        
        # Ajustes finales
        plt.tight_layout()
        plt.show()

    def plot_variability_dashboard(self, min_count_threshold: float = 0.5, filter_outliers: bool = True,
                             agrupar_por_intervalos: bool = False, intervalo_minutos: int = 5):
        """
        Genera un panel de control que muestra las principales métricas de variabilidad
        en un solo gráfico para una visión general completa.
        
        Este dashboard incluye:
        1. SDhh:mm - Desviación estándar entre puntos temporales (variabilidad a lo largo del día)
        2. SDbhh:mm - Desviación estándar entre días para cada punto temporal (consistencia día a día)
        3. SDw - Desviación estándar dentro del día (variabilidad diaria)
        4. SDdm - Desviación estándar de medias diarias (variabilidad entre días)
        5. Componentes de la variabilidad - Desglose de la varianza total en sus componentes
        
        :param min_count_threshold: Umbral para considerar un punto como válido
                               (proporción de la mediana de conteos). Por defecto 0.5 (50%).
        :param filter_outliers: Si es True, filtra los puntos con pocos datos.
        :param agrupar_por_intervalos: Si es True, agrupa los datos en intervalos regulares de tiempo.
        :param intervalo_minutos: Tamaño del intervalo en minutos para la agrupación (por defecto 5 min).
        """
        print("Iniciando dashboard de variabilidad...")
        
        # Obtener todas las métricas de variabilidad
        print("Calculando métricas generales...")
        metrics = self.calculate_all_sd_metrics()
        print(f"Métricas calculadas: SDT={metrics['SDT']:.2f}, SDw={metrics['SDw']:.2f}, SDhh:mm={metrics['SDhh:mm']:.2f}, SDdm={metrics['SDdm']:.2f}")
        
        # Obtener los datos detallados llamando directamente a los métodos individuales
        print("Calculando SD entre puntos temporales (SDhh:mm)...")
        sd_bt_result = self.sd_between_timepoints(min_count_threshold, filter_outliers, 
                                            agrupar_por_intervalos, intervalo_minutos)
        print(f"SDhh:mm calculado: {sd_bt_result['sd']:.2f} mg/dL")
        
        print("Calculando SD en el mismo punto temporal (SDbhh:mm)...")
        sd_st_result = self.sd_same_timepoint(min_count_threshold, filter_outliers, 
                                         agrupar_por_intervalos, intervalo_minutos)
        print(f"SDbhh:mm calculado: {sd_st_result['sd']:.2f} mg/dL")
        
        print("Calculando SD dentro del día (SDw)...")
        sd_wd_result = self.sd_within_day(min_count_threshold)
        print(f"SDw calculado: {sd_wd_result['sd']:.2f} mg/dL")
        
        print("Calculando SD de medias diarias (SDdm)...")
        sd_dm_result = self.sd_daily_mean(min_count_threshold)
        print(f"SDdm calculado: {sd_dm_result['sd']:.2f} mg/dL")
        
        print("Creando figura principal...")
        # Crear figura con 5 subplots (2x2 para las métricas principales y 1 para componentes)
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.8], hspace=0.4, wspace=0.3)
        
        print("Preparando gráfico 1: SDhh:mm...")
        # 1. SDhh:mm - Desviación estándar entre puntos temporales (arriba izquierda)
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Preparar datos para SDhh:mm
        time_decimal_bt = []
        values_bt = []
        counts_bt = []
        
        # Obtener los datos por marca temporal para SDhh:mm
        print(f"Procesando {len(sd_bt_result['conteo_por_marca'])} marcas temporales para SDhh:mm...")
        for time_str, count in sd_bt_result['conteo_por_marca'].items():
            h, m = map(int, time_str.split(':'))
            time_decimal_bt.append(h + m/60.0)
            
            # Para los valores, necesitamos recalcularlos o extraerlos del resultado
            if agrupar_por_intervalos:
                # Si estamos agrupando, los valores ya están en el resultado
                data_copy = self.data.copy()
                minutos_del_dia = data_copy['time'].dt.hour * 60 + data_copy['time'].dt.minute
                intervalo_redondeado = (minutos_del_dia // intervalo_minutos) * intervalo_minutos
                horas = intervalo_redondeado // 60
                minutos = intervalo_redondeado % 60
                data_copy['time_interval'] = horas.astype(str).str.zfill(2) + ':' + minutos.astype(str).str.zfill(2)
                
                # Filtrar solo los datos para este intervalo de tiempo
                interval_data = data_copy[data_copy['time_interval'] == time_str]
                values_bt.append(interval_data['glucose'].mean())
            else:
                # Si no estamos agrupando, filtramos por la hora exacta
                hour_data = self.data[self.data['time'].dt.strftime("%H:%M") == time_str]
                values_bt.append(hour_data['glucose'].mean())
            
            counts_bt.append(count)
        
        print("Dibujando gráfico 1: SDhh:mm...")
        # Graficar SDhh:mm
        scatter1 = ax1.scatter(time_decimal_bt, values_bt, s=[max(20, c/3) for c in counts_bt], 
                         c=counts_bt, cmap='viridis', alpha=0.7, edgecolors='k', linewidths=0.5)
        ax1.plot(time_decimal_bt, values_bt, 'r-', alpha=0.5)
        
        # Configuración para SDhh:mm
        ax1.set_title(f'SDhh:mm = {metrics["SDhh:mm"]:.2f} mg/dL\nPatrón Promedio de Glucosa', 
                 fontsize=12, fontweight='bold')
        ax1.set_xlabel('Hora del Día', fontsize=10)
        ax1.set_ylabel('Glucosa (mg/dL)', fontsize=10)
        ax1.set_xticks(range(0, 25, 6))
        ax1.set_xticklabels([f'{h:02d}:00' for h in range(0, 25, 6)])
        ax1.set_xlim(0, 24)
        ax1.grid(True, linestyle=':', alpha=0.6)
        
        print("Preparando gráfico 2: SDbhh:mm...")
        # 2. SDbhh:mm - Desviación estándar entre días para cada punto temporal (arriba derecha)
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Preparar datos para SDbhh:mm
        time_decimal_st = []
        values_st = []
        counts_st = []
        
        # Obtener los datos por marca temporal para SDbhh:mm
        print(f"Procesando {len(sd_st_result['sd_por_marca'])} marcas temporales para SDbhh:mm...")
        for time_str in sd_st_result['sd_por_marca'].keys():
            h, m = map(int, time_str.split(':'))
            time_decimal_st.append(h + m/60.0)
            values_st.append(sd_st_result['sd_por_marca'][time_str])
            counts_st.append(sd_st_result['conteo_por_marca'][time_str])
        
        print("Dibujando gráfico 2: SDbhh:mm...")
        # Graficar SDbhh:mm
        scatter2 = ax2.scatter(time_decimal_st, values_st, s=[max(20, c/3) for c in counts_st], 
                         c=counts_st, cmap='viridis', alpha=0.7, edgecolors='k', linewidths=0.5)
        ax2.plot(time_decimal_st, values_st, 'r-', alpha=0.5)
        
        # Configuración para SDbhh:mm
        ax2.set_title(f'SDbhh:mm = {metrics["SDbhh:mm"]:.2f} mg/dL\nVariabilidad Entre Días por Hora', 
                 fontsize=12, fontweight='bold')
        ax2.set_xlabel('Hora del Día', fontsize=10)
        ax2.set_ylabel('Desviación Estándar (mg/dL)', fontsize=10)
        ax2.set_xticks(range(0, 25, 6))
        ax2.set_xticklabels([f'{h:02d}:00' for h in range(0, 25, 6)])
        ax2.set_xlim(0, 24)
        ax2.grid(True, linestyle=':', alpha=0.6)
        
        # 3. SDw - Desviación estándar dentro del día (abajo izquierda)
        ax3 = fig.add_subplot(gs[1, 0])
        
        # Preparar datos para SDw
        # Verificamos si tenemos los detalles del cálculo o necesitamos recalcularlos
        if 'estadisticas_diarias' in sd_wd_result:
            filtered_stats = pd.DataFrame.from_dict(sd_wd_result['estadisticas_diarias'])
        else:
            # Recalcular desde los datos originales
            daily_stats = self.data.groupby(self.data['time'].dt.date)['glucose'].agg(['std', 'count'])
            count_threshold = daily_stats['count'].median() * min_count_threshold
            filtered_stats = daily_stats[daily_stats['count'] >= count_threshold]
        
        # Datos para graficar
        dates_wd = [date.strftime('%d-%m-%Y') for date in filtered_stats.index]
        sd_values = filtered_stats['std'].values
        counts_wd = filtered_stats['count'].values
        
        # Graficar SDw
        scatter3 = ax3.scatter(range(len(dates_wd)), sd_values, s=[max(20, c/5) for c in counts_wd], 
                         c=counts_wd, cmap='viridis', alpha=0.7, edgecolors='k', linewidths=0.5)
        
        # Añadir línea horizontal para SDw
        sdw_value = metrics["SDw"]
        ax3.axhline(y=sdw_value, color='r', linestyle='--', linewidth=1.5)
        
        # Configuración para SDw
        ax3.set_title(f'SDw = {sdw_value:.2f} mg/dL\nVariabilidad Dentro del Día', 
                 fontsize=12, fontweight='bold')
        ax3.set_xlabel('Días', fontsize=10)
        ax3.set_ylabel('Desviación Estándar (mg/dL)', fontsize=10)
        ax3.grid(True, linestyle=':', alpha=0.6)
        
        # Configurar el eje x para mostrar algunas fechas
        if len(dates_wd) > 10:
            step = max(1, len(dates_wd) // 5)
            ax3.set_xticks(range(0, len(dates_wd), step))
            ax3.set_xticklabels([dates_wd[i] for i in range(0, len(dates_wd), step)], rotation=45, ha='right', fontsize=8)
        else:
            ax3.set_xticks(range(len(dates_wd)))
            ax3.set_xticklabels(dates_wd, rotation=45, ha='right', fontsize=8)
        
        # 4. SDdm - Desviación estándar de medias diarias (abajo derecha)
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Preparar datos para SDdm
        if 'estadisticas_diarias' in sd_dm_result:
            filtered_means = pd.DataFrame.from_dict(sd_dm_result['estadisticas_diarias'])
        else:
            # Recalcular desde los datos originales
            daily_stats_dm = self.data.groupby(self.data['time'].dt.date)['glucose'].agg(['mean', 'count'])
            count_threshold_dm = daily_stats_dm['count'].median() * min_count_threshold
            filtered_means = daily_stats_dm[daily_stats_dm['count'] >= count_threshold_dm]
        
        # Datos para graficar
        dates_dm = [date.strftime('%d-%m-%Y') for date in filtered_means.index]
        mean_values = filtered_means['mean'].values
        counts_dm = filtered_means['count'].values
        
        # Graficar SDdm
        scatter4 = ax4.scatter(range(len(dates_dm)), mean_values, s=[max(20, c/5) for c in counts_dm], 
                         c=counts_dm, cmap='viridis', alpha=0.7, edgecolors='k', linewidths=0.5)
        
        # Añadir líneas horizontales para media y media ± SDdm
        mean_global = filtered_means['mean'].mean()
        sdm_value = metrics["SDdm"]
        ax4.axhline(y=mean_global, color='b', linestyle='--', linewidth=1.5)
        ax4.axhline(y=mean_global + sdm_value, color='r', linestyle=':', linewidth=1.0)
        ax4.axhline(y=mean_global - sdm_value, color='r', linestyle=':', linewidth=1.0)
        
        # Sombrear el área entre media ± SDdm
        ax4.fill_between(
            [-1, len(dates_dm) + 1], 
            mean_global - sdm_value, 
            mean_global + sdm_value, 
            color='red', alpha=0.1
        )
        
        # Configuración para SDdm
        ax4.set_title(f'SDdm = {sdm_value:.2f} mg/dL\nVariabilidad Entre Días', 
                 fontsize=12, fontweight='bold')
        ax4.set_xlabel('Días', fontsize=10)
        ax4.set_ylabel('Glucosa Media (mg/dL)', fontsize=10)
        ax4.grid(True, linestyle=':', alpha=0.6)
        
        # Configurar el eje x para mostrar algunas fechas
        if len(dates_dm) > 10:
            step = max(1, len(dates_dm) // 5)
            ax4.set_xticks(range(0, len(dates_dm), step))
            ax4.set_xticklabels([dates_dm[i] for i in range(0, len(dates_dm), step)], rotation=45, ha='right', fontsize=8)
        else:
            ax4.set_xticks(range(len(dates_dm)))
            ax4.set_xticklabels(dates_dm, rotation=45, ha='right', fontsize=8)
        
        print("Preparando gráfico 5: Componentes de SD...")
        # 5. Componentes de la SD (abajo, ocupando todo el ancho)
        ax5 = fig.add_subplot(gs[2, :])
        
        # Generar el gráfico de componentes similar a plot_sd_components
        # Primero guardamos la figura actual para poder restaurarla después
        current_fig = plt.gcf()
        current_ax = plt.gca()
        
        print("Recopilando componentes para el gráfico 5...")
        # Crear los datos para el gráfico de componentes
        componentes = [
            ('SDT', metrics['SDT'], 'crimson'),
            ('SDw', metrics['SDw'], 'royalblue'),
            ('SDhh:mm', metrics['SDhh:mm'], 'limegreen'),
            ('Noche', self.sd_by_segment('22:00', 8) if 'Noche' not in metrics else metrics['Noche'], 'orange'),
            ('Día', self.sd_by_segment('06:00', 16) if 'Día' not in metrics else metrics['Día'], 'hotpink'),
            ('Tarde', self.sd_by_segment('14:00', 8) if 'Tarde' not in metrics else metrics['Tarde'], 'mediumspringgreen'),
            ('SDws_1h', self.sd_within_segment('01:00') if 'SDws_1h' not in metrics else metrics['SDws_1h'], 'darkorange'),
            ('SDws_6h', self.sd_within_segment('06:00') if 'SDws_6h' not in metrics else metrics['SDws_6h'], 'mediumpurple'),
            ('SDws_24h', self.sd_within_day() if 'SDws_24h' not in metrics else metrics['SDws_24h'], 'palegreen'),
            ('SDdm', metrics['SDdm'], 'magenta'),
            ('SDbhh:mm', metrics['SDbhh:mm'], 'skyblue'),
            ('SDbhh:mm_dm', metrics['SDbhh:mm_dm'] if 'SDbhh:mm_dm' in metrics else np.nan, 'sandybrown')
        ]
        
        # Añadir componentes adicionales si están disponibles
        print("Verificando componentes adicionales...")
        
        # Noche
        try:
            noche_value = metrics['Noche'] if 'Noche' in metrics else self.sd_by_segment('22:00', 8)
            print(f"Componente Noche: {noche_value:.2f} mg/dL")
            componentes.append(('Noche', noche_value, 'orange'))
        except Exception as e:
            print(f"Error al calcular Noche: {e}")
        
        # Día
        try:
            dia_value = metrics['Día'] if 'Día' in metrics else self.sd_by_segment('06:00', 16)
            print(f"Componente Día: {dia_value:.2f} mg/dL")
            componentes.append(('Día', dia_value, 'hotpink'))
        except Exception as e:
            print(f"Error al calcular Día: {e}")
        
        # Tarde
        try:
            tarde_value = metrics['Tarde'] if 'Tarde' in metrics else self.sd_by_segment('14:00', 8)
            print(f"Componente Tarde: {tarde_value:.2f} mg/dL")
            componentes.append(('Tarde', tarde_value, 'mediumspringgreen'))
        except Exception as e:
            print(f"Error al calcular Tarde: {e}")
        
        # SDws_1h
        try:
            sdws_1h = metrics['SDws_1h'] if 'SDws_1h' in metrics else self.sd_within_segment('01:00')
            print(f"Componente SDws_1h: {sdws_1h:.2f} mg/dL")
            componentes.append(('SDws_1h', sdws_1h, 'darkorange'))
        except Exception as e:
            print(f"Error al calcular SDws_1h: {e}")
        
        # SDws_6h
        try:
            sdws_6h = metrics['SDws_6h'] if 'SDws_6h' in metrics else self.sd_within_segment('06:00')
            print(f"Componente SDws_6h: {sdws_6h:.2f} mg/dL")
            componentes.append(('SDws_6h', sdws_6h, 'mediumpurple'))
        except Exception as e:
            print(f"Error al calcular SDws_6h: {e}")
        
        # SDws_24h
        try:
            sdws_24h = metrics['SDws_24h'] if 'SDws_24h' in metrics else self.sd_within_day()
            print(f"Componente SDws_24h: {sdws_24h:.2f} mg/dL")
            componentes.append(('SDws_24h', sdws_24h, 'palegreen'))
        except Exception as e:
            print(f"Error al calcular SDws_24h: {e}")
        
        # Añadir componentes básicos restantes
        componentes.append(('SDdm', metrics['SDdm'], 'magenta'))
        componentes.append(('SDbhh:mm', metrics['SDbhh:mm'], 'skyblue'))
        
        # SDbhh:mm_dm
        if 'SDbhh:mm_dm' in metrics:
            componentes.append(('SDbhh:mm_dm', metrics['SDbhh:mm_dm'], 'sandybrown'))
            print(f"Componente SDbhh:mm_dm: {metrics['SDbhh:mm_dm']:.2f} mg/dL")
        
        # Añadir SDI si está disponible
        if 'SDI' in metrics:
            componentes.append(('SDI', metrics['SDI'], 'mediumpurple'))
            print(f"Componente SDI: {metrics['SDI']:.2f} mg/dL")
        
        print(f"Total de componentes a graficar: {len(componentes)}")
        
        # Ordenar componentes por valor para mejor visualización
        print("Ordenando componentes...")
        componentes.sort(key=lambda x: x[1] if not np.isnan(x[1]) else 0)
        
        # Eliminar componentes con valores NaN
        componentes = [c for c in componentes if not np.isnan(c[1])]
        print(f"Componentes válidos después de filtrar NaN: {len(componentes)}")
        
        # Extraer nombres, valores y colores
        nombres = [c[0] for c in componentes]
        valores = [c[1] for c in componentes]
        colores = [c[2] for c in componentes]
        
        print("Dibujando gráfico 5: Componentes de SD...")
        # Crear barras horizontales
        y_pos = np.arange(len(componentes))
        barras = ax5.barh(y_pos, valores, color=colores, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Añadir valores numéricos al final de cada barra
        for i, barra in enumerate(barras):
            ax5.text(barra.get_width() + 1, barra.get_y() + barra.get_height()/2, 
                    f'{valores[i]:.1f}', va='center', fontsize=9)
        
        # Configuración del gráfico
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels(nombres)
        ax5.set_xlabel('Valor (mg/dL)', fontsize=12)
        ax5.set_title('Componentes de la Desviación Estándar de Glucosa', fontsize=14, fontweight='bold')
        ax5.grid(True, axis='x', linestyle=':', alpha=0.6)
        
        # Restaurar la figura original
        plt.figure(current_fig.number)
        
        print("Finalizando dashboard...")
        # Añadir título general
        fig.suptitle('Panel Completo de Métricas de Variabilidad Glucémica', fontsize=18, fontweight='bold', y=0.98)
        
        # Añadir información general
        info_text = (
            f"Periodo analizado: {min(self.data['time']).strftime('%d-%m-%Y')} a {max(self.data['time']).strftime('%d-%m-%Y')}\n"
            f"Total de mediciones: {len(self.data)}\n"
            f"Media global: {self.mean():.2f} mg/dL\n"
            f"CV global: {self.cv():.2f}%\n"
            f"SD global: {metrics['SDT']:.2f} mg/dL"
        )
        fig.text(0.5, 0.01, info_text, ha='center', fontsize=10, 
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        # Ajustes finales
        print("Ajustando layout...")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        print("Mostrando dashboard completo...")
        plt.show()
        print("Dashboard mostrado con éxito.")