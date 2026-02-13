"""
Módulo de gráficos diarios para datos de glucosa.

Este módulo contiene las funciones para generar gráficos relacionados con patrones diarios:
- Gráficos de días específicos
- Superposición de múltiples días
- Boxplots por día de la semana
- Análisis de variaciones diarias
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class DailyPlotter:
    """
    Clase para generar gráficos diarios de glucosa.

    Esta clase debe ser utilizada como mixin con GlucoseData.
    """

    def day_graph(self, fecha: Optional[str] = None):
        """
        Genera y muestra el gráfico de glucosa para un día específico.

        Args:
            fecha: Fecha opcional en formato 'YYYY-MM-DD'.
                  Si no se proporciona, se usa el primer día del DataFrame.
        """
        # Si no se proporciona fecha, usar el primer día del DataFrame
        if fecha is None:
            fecha = self.data["time"].dt.date.min()
        else:
            fecha = pd.to_datetime(fecha).date()

        # Filtrar datos para el día específico
        day_data = self.data[self.data["time"].dt.date == fecha].copy()

        if day_data.empty:
            print(f"No hay datos para la fecha {fecha}")
            return

        # Convertir la hora a un formato numérico para el gráfico
        day_data["hours"] = day_data["time"].dt.hour + day_data["time"].dt.minute / 60.0

        # Configurar el estilo
        sns.set_style("whitegrid")
        sns.set_context("notebook", font_scale=1.1)

        fig, ax = plt.subplots(figsize=(16, 9))

        # Configurar zonas de glucemia
        self._add_glucose_zones(ax)

        # Gráfico de línea con marcadores
        ax.plot(
            day_data["hours"],
            day_data["glucose"],
            label="Glucosa",
            color="#3366CC",
            linewidth=2,
            marker="o",
            markersize=4,
        )

        # Configurar referencias
        self._add_reference_lines(ax)

        # Configurar el gráfico
        self._configure_daily_plot(ax, f"Niveles de Glucosa - {fecha}")

        plt.tight_layout()
        plt.show()

    def plot_overlapping_days(self):
        """
        Genera un gráfico con los perfiles de glucosa de múltiples días superpuestos.
        Cada línea representa un día diferente.
        """
        # Preparar datos
        data_copy = self.data.copy()
        data_copy["time_decimal"] = data_copy["time"].dt.hour + data_copy["time"].dt.minute / 60.0
        data_copy["date"] = data_copy["time"].dt.date

        # Configurar figura
        plt.figure(figsize=(12, 8))

        # Calcular el perfil medio
        mean_profile = (
            data_copy.groupby("time_decimal")["glucose"].mean().rolling(window=15, center=True, min_periods=1).mean()
        )

        # Graficar cada día individual
        dates = data_copy["date"].unique()
        for date in dates:
            day_data = data_copy[data_copy["date"] == date]
            plt.plot(
                day_data["time_decimal"],
                day_data["glucose"],
                color="gray",
                alpha=0.2,
                linewidth=1,
            )

        # Graficar el perfil medio
        plt.plot(
            mean_profile.index,
            mean_profile.values,
            color="black",
            linewidth=2,
            label="Perfil medio",
        )

        # Configurar el gráfico
        self._configure_overlapping_plot()

        plt.tight_layout()
        plt.show()

    def plot_week_boxplots(self):
        """
        Genera un gráfico de boxplots para visualizar la distribución de glucosa
        por día de la semana, incluyendo el número de días para cada día.
        """
        # Preparar datos
        data_copy = self.data.copy()
        data_copy["weekday"] = data_copy["time"].dt.day_name(locale="es_ES")
        data_copy["date"] = data_copy["time"].dt.date

        # Definir el orden de los días
        orden_dias = [
            "Lunes",
            "Martes",
            "Miércoles",
            "Jueves",
            "Viernes",
            "Sábado",
            "Domingo",
        ]

        # Calcular el número de días únicos para cada día de la semana
        dias_unicos = data_copy.groupby("weekday")["date"].nunique()

        # Crear etiquetas con el número de días
        etiquetas = [f"{dia}\n(n={dias_unicos.get(dia, 0)} días)" for dia in orden_dias]

        # Crear la figura
        plt.figure(figsize=(12, 8))

        # Configurar zonas de glucemia
        plt.axhspan(0, 70, color="#ffcccb", alpha=0.2, label="Hipoglucemia")
        plt.axhspan(70, 180, color="#90ee90", alpha=0.2, label="Rango objetivo")
        plt.axhspan(180, 400, color="#ffcccb", alpha=0.2, label="Hiperglucemia")

        # Crear el boxplot
        sns.boxplot(
            x="weekday",
            y="glucose",
            data=data_copy,
            order=orden_dias,
            whis=1.5,
            medianprops=dict(color="red", linewidth=1.5),
            flierprops=dict(marker="o", markerfacecolor="gray", markersize=4),
        )

        # Líneas de referencia
        plt.axhline(y=70, color="red", linestyle="--", linewidth=1)
        plt.axhline(y=180, color="red", linestyle="--", linewidth=1)

        # Configurar el gráfico
        plt.title("Distribución de Glucosa por Día de la Semana", fontsize=14, pad=20)
        plt.xlabel("Día de la Semana", fontsize=12)
        plt.ylabel("Nivel de Glucosa (mg/dL)", fontsize=12)

        # Actualizar etiquetas del eje x
        plt.xticks(range(len(orden_dias)), etiquetas, rotation=45, ha="right")
        plt.ylim(0, 400)

        # Leyenda
        plt.legend(title="Rangos", bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        plt.show()

    def plot_daily_variations(self):
        """
        Genera un gráfico que muestra las variaciones diarias promedio
        con bandas de confianza.
        """
        # Preparar datos
        data_copy = self.data.copy()
        data_copy["time_decimal"] = data_copy["time"].dt.hour + data_copy["time"].dt.minute / 60.0

        # Calcular estadísticas por hora del día
        hourly_stats = (
            data_copy.groupby("time_decimal")["glucose"]
            .agg(
                [
                    "mean",
                    "std",
                    "count",
                    lambda x: np.percentile(x, 25),
                    lambda x: np.percentile(x, 75),
                ]
            )
            .reset_index()
        )

        hourly_stats.columns = ["time_decimal", "mean", "std", "count", "p25", "p75"]

        # Aplicar suavizado
        window_size = 15
        for col in ["mean", "std", "p25", "p75"]:
            hourly_stats[col] = hourly_stats[col].rolling(window=window_size, center=True, min_periods=1).mean()

        # Crear figura
        fig, ax = plt.subplots(figsize=(14, 8))

        # Configurar zonas de glucemia
        self._add_glucose_zones(ax)

        # Plotear media con banda de confianza
        ax.plot(
            hourly_stats["time_decimal"],
            hourly_stats["mean"],
            color="blue",
            linewidth=2,
            label="Media",
        )

        # Banda de desviación estándar
        ax.fill_between(
            hourly_stats["time_decimal"],
            hourly_stats["mean"] - hourly_stats["std"],
            hourly_stats["mean"] + hourly_stats["std"],
            alpha=0.3,
            color="blue",
            label="± 1 SD",
        )

        # Rango intercuartil
        ax.fill_between(
            hourly_stats["time_decimal"],
            hourly_stats["p25"],
            hourly_stats["p75"],
            alpha=0.2,
            color="green",
            label="Rango intercuartil",
        )

        # Configurar el gráfico
        ax.set_xlabel("Hora del Día", fontsize=12)
        ax.set_ylabel("Nivel de Glucosa (mg/dL)", fontsize=12)
        ax.set_title("Variaciones Diarias Promedio de Glucosa", fontsize=14)

        # Configurar eje x
        ax.set_xticks(range(0, 25, 3))
        ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 3)])
        ax.set_xlim(0, 24)
        ax.set_ylim(0, 400)

        # Referencias
        ax.axhline(y=70, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax.axhline(y=180, color="red", linestyle="--", linewidth=1, alpha=0.7)

        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.show()

    def _add_glucose_zones(self, ax):
        """Añade las zonas de glucemia al gráfico."""
        ax.axhspan(0, 70, facecolor="#FF9999", alpha=0.2, label="Hipoglucemia")
        ax.axhspan(70, 180, facecolor="#90EE90", alpha=0.2, label="Rango objetivo")
        ax.axhspan(180, 400, facecolor="#FFB266", alpha=0.2, label="Hiperglucemia")

    def _add_reference_lines(self, ax):
        """Añade líneas de referencia al gráfico."""
        ax.axhline(y=70, color="#FF6666", linestyle="--", linewidth=1)
        ax.axhline(y=180, color="#FF6666", linestyle="--", linewidth=1)
        ax.text(24, 72, "70 mg/dL", va="bottom", ha="right", color="#FF6666")
        ax.text(24, 182, "180 mg/dL", va="bottom", ha="right", color="#FF6666")

    def _configure_daily_plot(self, ax, title: str):
        """Configura los elementos comunes del gráfico diario."""
        ax.set_xlabel("Hora del Día", fontsize=12, fontweight="bold")
        ax.set_ylabel("Nivel de Glucosa (mg/dL)", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=16, fontweight="bold")

        ax.legend(loc="upper left", frameon=True, fancybox=True, shadow=True)
        ax.set_ylim(0, 400)
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 3))
        ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 3)])

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        ax.grid(True, linestyle=":", alpha=0.6)

    def _configure_overlapping_plot(self):
        """Configura el gráfico de días superpuestos."""
        plt.xlabel("Hora del Día", fontsize=12)
        plt.ylabel("Nivel de Glucosa (mg/dL)", fontsize=12)
        plt.title("Perfiles de Glucosa Superpuestos", fontsize=14)

        # Configurar eje x
        plt.xticks(range(0, 25, 3), [f"{h:02d}:00" for h in range(0, 25, 3)])

        # Líneas de referencia
        plt.axhline(y=70, color="red", linestyle="--", alpha=0.5)
        plt.axhline(y=180, color="red", linestyle="--", alpha=0.5)

        # Zonas coloreadas
        plt.axhspan(0, 70, facecolor="#ffcccb", alpha=0.2)
        plt.axhspan(70, 180, facecolor="#90ee90", alpha=0.2)
        plt.axhspan(180, 400, facecolor="#ffcccb", alpha=0.2)

        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(0, 400)
