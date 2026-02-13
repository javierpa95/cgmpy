"""
Módulo de gráficos de perfil ambulatorio de glucosa (AGP).

Este módulo contiene las funciones para generar perfiles ambulatorios:
- AGP estándar
- AGP por días de la semana
- Funciones auxiliares para cálculo de percentiles
"""

import matplotlib.pyplot as plt
import numpy as np


class AGPPlotter:
    """
    Clase para generar gráficos de perfil ambulatorio de glucosa (AGP).

    Esta clase debe ser utilizada como mixin con GlucoseData.
    """

    def plot_agp(self, smoothing_window: int = 15):
        """
        Genera y muestra el Perfil de Glucosa Ambulatoria (AGP) mejorado.

        Args:
            smoothing_window: Ventana de suavizado en minutos (por defecto 15)
        """
        # Preparar datos
        data_copy = self.data.copy()
        data_copy["time_decimal"] = (data_copy["time"].dt.hour + data_copy["time"].dt.minute / 60.0).round(2)

        # Calcular percentiles
        percentiles = data_copy.groupby("time_decimal")["glucose"].agg(
            [
                lambda x: np.percentile(x, 5),
                lambda x: np.percentile(x, 25),
                lambda x: np.percentile(x, 50),
                lambda x: np.percentile(x, 75),
                lambda x: np.percentile(x, 95),
            ]
        )

        # Renombrar columnas
        percentiles.columns = [0.05, 0.25, 0.5, 0.75, 0.95]

        # Aplicar suavizado
        for col in percentiles.columns:
            percentiles[col] = percentiles[col].rolling(window=smoothing_window, center=True, min_periods=1).mean()

        # Asegurar que los datos están ordenados
        percentiles = percentiles.sort_index()

        # Crear figura
        fig, ax = plt.subplots(figsize=(14, 8))

        # Configurar zonas de glucemia
        self._add_glucose_zones(ax)

        # Plotear percentiles
        self._plot_percentiles(ax, percentiles)

        # Configurar gráfico
        self._configure_agp_plot(ax, "Perfil de Glucosa Ambulatoria (AGP)")

        plt.tight_layout()
        plt.show()

    def generate_week_agp(self, smoothing_window: int = 15, combined: bool = True):
        """
        Genera y muestra el Perfil de Glucosa Ambulatoria (AGP) por días de la semana.

        Args:
            smoothing_window: Ventana de suavizado en minutos (por defecto 15)
            combined: Si es True, muestra todos los días en un solo gráfico.
                     Si es False, muestra un subplot para cada día.
        """
        # Preparar los datos
        data_copy = self.data.copy()
        data_copy["time_decimal"] = (data_copy["time"].dt.hour + data_copy["time"].dt.minute / 60.0).round(2)
        data_copy["weekday"] = data_copy["time"].dt.day_name(locale="es_ES")

        if combined:
            self._plot_combined_week_agp(data_copy, smoothing_window)
        else:
            self._plot_separate_week_agp(data_copy, smoothing_window)

    def _add_glucose_zones(self, ax):
        """Añade las zonas de glucemia al gráfico."""
        ax.axhspan(0, 70, facecolor="#ffcccb", alpha=0.3, label="Hipoglucemia")
        ax.axhspan(70, 180, facecolor="#90ee90", alpha=0.3, label="Rango objetivo")
        ax.axhspan(180, 400, facecolor="#ffcccb", alpha=0.3, label="Hiperglucemia")

        # Líneas horizontales en 70 y 180 mg/dL
        ax.axhline(y=70, color="red", linestyle="--", linewidth=1)
        ax.axhline(y=180, color="red", linestyle="--", linewidth=1)

    def _plot_percentiles(self, ax, percentiles):
        """Plotea las líneas de percentiles."""
        # Línea mediana
        ax.plot(
            percentiles.index,
            percentiles[0.5],
            label="Mediana",
            color="blue",
            linewidth=2,
        )

        # Rango intercuartil
        ax.fill_between(
            percentiles.index,
            percentiles[0.25],
            percentiles[0.75],
            color="blue",
            alpha=0.3,
            label="Rango Intercuartil",
        )

        # Percentiles 5-95%
        ax.fill_between(
            percentiles.index,
            percentiles[0.05],
            percentiles[0.95],
            color="lightblue",
            alpha=0.2,
            label="Percentiles 5-95%",
        )

    def _configure_agp_plot(self, ax, title: str):
        """Configura los elementos comunes del gráfico AGP."""
        # Etiquetas y título
        ax.set_xlabel("Hora del Día", fontsize=12)
        ax.set_ylabel("Nivel de Glucosa (mg/dL)", fontsize=12)
        ax.set_title(title, fontsize=16, fontweight="bold")

        # Leyenda
        ax.legend(title="Leyenda", loc="upper left", fontsize=10)

        # Cuadrícula
        ax.grid(True, linestyle=":", alpha=0.6)

        # Configuración del eje x
        ax.set_xticks(range(0, 25, 3))
        ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 3)])

        # Límites del eje y
        ax.set_ylim(0, 400)

    def _plot_combined_week_agp(self, data_copy, smoothing_window: int):
        """Plotea AGP combinado para todos los días de la semana."""
        # Orden de los días y colores
        dias = [
            "Lunes",
            "Martes",
            "Miércoles",
            "Jueves",
            "Viernes",
            "Sábado",
            "Domingo",
        ]
        colores = [
            "#FF6B6B",
            "#4ECDC4",
            "#45B7D1",
            "#96CEB4",
            "#FFEEAD",
            "#D4A5A5",
            "#9B59B6",
        ]

        # Crear figura
        fig, ax = plt.subplots(figsize=(15, 8))

        # Configurar zonas de glucemia
        self._add_glucose_zones(ax)

        for dia, color in zip(dias, colores):
            # Filtrar datos para el día específico
            dia_data = data_copy[data_copy["weekday"] == dia]

            if not dia_data.empty:
                # Calcular percentiles
                percentiles = self._calculate_day_percentiles(dia_data, smoothing_window)

                # Graficar línea mediana
                ax.plot(
                    percentiles.index,
                    percentiles[0.5],
                    label=f"{dia} (n={len(dia_data['time'].dt.date.unique())} días)",
                    color=color,
                    linewidth=2,
                )

                # Área del IQR con transparencia
                ax.fill_between(
                    percentiles.index,
                    percentiles[0.25],
                    percentiles[0.75],
                    color=color,
                    alpha=0.1,
                )

        # Configuración del gráfico
        ax.set_title(
            "Perfil de Glucosa Ambulatoria (AGP) por Día de la Semana",
            fontsize=14,
            pad=20,
        )
        ax.set_xlabel("Hora del Día", fontsize=12)
        ax.set_ylabel("Nivel de Glucosa (mg/dL)", fontsize=12)
        ax.set_ylim(0, 400)
        ax.grid(True, linestyle=":", alpha=0.6)

        # Configuración del eje x
        ax.set_xticks(range(0, 25, 3))
        ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 3)])

        # Leyenda
        ax.legend(
            title="Días de la semana",
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fontsize=10,
        )

        plt.tight_layout()
        plt.show()

    def _plot_separate_week_agp(self, data_copy, smoothing_window: int):
        """Plotea AGP separado para cada día de la semana."""
        dias = [
            "Lunes",
            "Martes",
            "Miércoles",
            "Jueves",
            "Viernes",
            "Sábado",
            "Domingo",
        ]

        # Crear subplots
        fig, axes = plt.subplots(7, 1, figsize=(15, 20), sharex=True)
        fig.suptitle(
            "Perfil de Glucosa Ambulatoria (AGP) por Día de la Semana",
            fontsize=16,
            fontweight="bold",
            y=0.92,
        )

        for ax, dia in zip(axes, dias):
            # Filtrar datos para el día específico
            dia_data = data_copy[data_copy["weekday"] == dia]

            if not dia_data.empty:
                # Calcular percentiles completos
                percentiles = self._calculate_full_day_percentiles(dia_data, smoothing_window)

                # Configurar zonas de glucemia
                self._add_glucose_zones(ax)

                # Plotear percentiles
                self._plot_percentiles(ax, percentiles)

                # Configurar subplot
                ax.set_title(
                    f"{dia} (n={len(dia_data['time'].dt.date.unique())} días)",
                    fontsize=12,
                    pad=10,
                )
                ax.set_ylabel("Glucosa (mg/dL)", fontsize=10)
                ax.set_ylim(0, 400)
                ax.grid(True, linestyle=":", alpha=0.6)

        # Configurar eje x solo en el último subplot
        axes[-1].set_xlabel("Hora del Día", fontsize=12)
        axes[-1].set_xticks(range(0, 25, 3))
        axes[-1].set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 3)])

        plt.tight_layout()
        plt.show()

    def _calculate_day_percentiles(self, dia_data, smoothing_window: int):
        """Calcula percentiles para datos de un día específico (25, 50, 75)."""
        percentiles = dia_data.groupby("time_decimal")["glucose"].agg(
            [
                lambda x: np.percentile(x, 25),
                lambda x: np.percentile(x, 50),
                lambda x: np.percentile(x, 75),
            ]
        )

        # Renombrar columnas
        percentiles.columns = [0.25, 0.5, 0.75]

        # Aplicar suavizado
        for col in percentiles.columns:
            percentiles[col] = percentiles[col].rolling(window=smoothing_window, center=True, min_periods=1).mean()

        return percentiles

    def _calculate_full_day_percentiles(self, dia_data, smoothing_window: int):
        """Calcula percentiles completos para datos de un día específico (5, 25, 50, 75, 95)."""
        percentiles = dia_data.groupby("time_decimal")["glucose"].agg(
            [
                lambda x: np.percentile(x, 5),
                lambda x: np.percentile(x, 25),
                lambda x: np.percentile(x, 50),
                lambda x: np.percentile(x, 75),
                lambda x: np.percentile(x, 95),
            ]
        )

        # Renombrar columnas
        percentiles.columns = [0.05, 0.25, 0.5, 0.75, 0.95]

        # Aplicar suavizado
        for col in percentiles.columns:
            percentiles[col] = percentiles[col].rolling(window=smoothing_window, center=True, min_periods=1).mean()

        return percentiles
