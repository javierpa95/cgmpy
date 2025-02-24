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

    