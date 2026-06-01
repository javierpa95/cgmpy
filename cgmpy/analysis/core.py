"""
Módulo principal de análisis de glucosa.

Este módulo combina todas las funcionalidades de análisis:
- Manejo de datos (ModularGlucoseData)
- Métricas y estadísticas (módulos de metrics)
- Visualización (módulos de plotting)
"""

import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..data.core import ModularGlucoseData
from ..metrics.time_in_range import TimeInRangeMetrics
from ..metrics.variability import VariabilityMetrics
from ..plotting.agp import AGPPlotter
from ..plotting.daily_plots import DailyPlotter
from ..plotting.statistical_plots import StatisticalPlotter


class GlucoseAnalysis(
    ModularGlucoseData,
    TimeInRangeMetrics,
    VariabilityMetrics,
    AGPPlotter,
    DailyPlotter,
    StatisticalPlotter,
):
    """
    Clase que combina todas las funcionalidades de análisis de glucosa.

    Esta clase hereda de:
    - ModularGlucoseData: Manejo de datos
    - BasicMetrics: Métricas básicas (media, mediana, GMI, etc.)
    - TimeInRangeMetrics: Métricas de tiempo en rango (TIR, TAR, TBR)
    - VariabilityMetrics: Métricas de variabilidad (MAGE, MODD, CONGA, etc.)
    - AGPPlotter: Gráficos de perfil ambulatorio
    - DailyPlotter: Gráficos diarios
    - StatisticalPlotter: Gráficos estadísticos
    """

    def __init__(
        self,
        data_source: str | pd.DataFrame,
        date_col: str = "time",
        glucose_col: str = "glucose",
        delimiter: str | None = None,
        header: int = 0,
        start_date: str | datetime.datetime | None = None,
        end_date: str | datetime.datetime | None = None,
        log: bool = False,
    ):
        """
        Inicializa el análisis completo de glucosa.

        Args:
            data_source: Fuente de datos (archivo o DataFrame)
            date_col: Nombre de la columna de fecha
            glucose_col: Nombre de la columna de glucosa
            delimiter: Delimitador del archivo
            header: Número de fila que contiene los encabezados
            start_date: Fecha de inicio para filtrar datos
            end_date: Fecha de fin para filtrar datos
            log: Si activar logs detallados
        """
        # Inicializar ModularGlucoseData
        super().__init__(
            data_source=data_source,
            date_col=date_col,
            glucose_col=glucose_col,
            delimiter=delimiter,
            header=header,
            start_date=start_date,
            end_date=end_date,
            log=log,
        )

    def get_comprehensive_report(self) -> dict[str, Any]:
        """
        Genera un reporte completo con todas las métricas disponibles.

        Returns:
            dict: Reporte completo con todas las métricas
        """
        report = {
            "informacion_basica": self.info(),
            "metricas_basicas": self.basic_statistics_summary(),
            "estadisticas_tiempo": self.time_statistics(),
            "metricas_variabilidad": self.calculate_all_variability_metrics(),
            "calidad_datos": self.get_data_quality_metrics(),
        }

        return report

    def get_summary_string(self) -> str:
        """
        Genera un resumen en texto del análisis.

        Returns:
            str: Resumen del análisis
        """
        summary = []
        summary.append("=== ANÁLISIS COMPLETO DE GLUCOSA ===")
        summary.append("")

        # Información básica
        info = self.info()
        summary.append("📊 DATOS:")
        summary.append(f"  - Registros: {info['num_datos']:,}")
        summary.append(
            f"  - Período: {info['fecha_inicio'].strftime('%d/%m/%Y')} - {info['fecha_fin'].strftime('%d/%m/%Y')}"
        )
        summary.append(f"  - Disponibilidad: {info['porcentaje_disponibilidad']:.1f}%")
        summary.append("")

        # Métricas básicas
        basic = self.basic_statistics_summary()
        summary.append("📈 MÉTRICAS BÁSICAS:")
        summary.append(f"  - GMI: {basic['GMI']:.1f}%")
        summary.append(f"  - Media: {basic['Media']:.1f} mg/dL")
        summary.append(f"  - Mediana: {basic['Mediana']:.1f} mg/dL")
        summary.append(f"  - Desviación estándar: {basic['Desviacion_estandar']:.1f} mg/dL")
        summary.append(f"  - CV: {basic['CV']:.1f}%")
        summary.append("")

        # Tiempo en rango
        time_stats = self.time_statistics()
        summary.append("⏰ TIEMPO EN RANGO:")
        summary.append(f"  - TIR (70-180): {time_stats['TIR']:.1f}%")
        summary.append(f"  - TIR estricto (70-140): {time_stats['TIR_tight']:.1f}%")
        summary.append(f"  - TBR70 (55-70): {time_stats['TBR70']:.1f}%")
        summary.append(f"  - TBR55 (<55): {time_stats['TBR55']:.1f}%")
        summary.append(f"  - TAR140 (140-250): {time_stats['TAR140']:.1f}%")
        summary.append(f"  - TAR180 (180-250): {time_stats['TAR180']:.1f}%")
        summary.append(f"  - TAR250 (>250): {time_stats['TAR250']:.1f}%")
        summary.append("")

        # Variabilidad
        variability = self.calculate_all_variability_metrics()
        summary.append("📊 VARIABILIDAD:")
        summary.append(f"  - MAGE: {variability.get('MAGE', 'N/A')}")
        summary.append(f"  - MODD: {variability.get('MODD', 'N/A')}")
        summary.append(f"  - CONGA: {variability.get('CONGA', 'N/A')}")
        summary.append(f"  - SD total: {variability.get('SD_total', 'N/A')}")
        summary.append(f"  - SD within-day: {variability.get('SD_within_day', 'N/A')}")
        summary.append(f"  - SD between-day: {variability.get('SD_between_day', 'N/A')}")

        return "\n".join(summary)

    def plot_comprehensive_dashboard(self, figsize: tuple = (20, 12)):
        """
        Genera un dashboard completo con múltiples gráficos.

        Args:
            figsize: Tamaño de la figura
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle("Dashboard Completo de Análisis de Glucosa", fontsize=16, fontweight="bold")

        # Gráfico 1: AGP
        self.plot_agp()
        axes[0, 0].set_title("Perfil Ambulatorio (AGP)")

        # Gráfico 2: Distribución
        self.histogram()
        axes[0, 1].set_title("Distribución de Glucosa")

        # Gráfico 3: Tiempo en rango
        self.plot_time_in_range()
        axes[0, 2].set_title("Tiempo en Rango")

        # Gráfico 4: Variabilidad
        self.plot_variability_dashboard()
        axes[1, 0].set_title("Análisis de Variabilidad")

        # Gráfico 5: Días superpuestos
        self.plot_overlapping_days()
        axes[1, 1].set_title("Días Superpuestos")

        # Gráfico 6: Boxplots semanales
        self.plot_week_boxplots()
        axes[1, 2].set_title("Boxplots Semanales")

        plt.tight_layout()
        plt.show()

    def export_report(self, file_path: str, format: str = "json"):
        """
        Exporta el reporte completo a un archivo.

        Args:
            file_path: Ruta del archivo de salida
            format: Formato de exportación ('json', 'csv', 'excel')
        """
        report = self.get_comprehensive_report()

        if format.lower() == "json":
            import json

            with Path(file_path).open("w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str, ensure_ascii=False)

        elif format.lower() == "csv":
            # Convertir reporte a DataFrame plano
            flat_report = self._flatten_report(report)
            flat_report.to_csv(file_path, index=False)

        elif format.lower() == "excel":
            # Crear Excel con múltiples hojas
            with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
                # Hoja 1: Información básica
                pd.DataFrame([report["informacion_basica"]]).to_excel(
                    writer, sheet_name="Info_Basica", index=False
                )

                # Hoja 2: Métricas básicas
                pd.DataFrame([report["metricas_basicas"]]).to_excel(
                    writer, sheet_name="Metricas_Basicas", index=False
                )

                # Hoja 3: Tiempo en rango
                pd.DataFrame([report["estadisticas_tiempo"]]).to_excel(
                    writer, sheet_name="Tiempo_Rango", index=False
                )

        else:
            raise ValueError(f"Formato no soportado: {format}")

    def _flatten_report(self, report: dict[str, Any]) -> pd.DataFrame:
        """
        Convierte el reporte anidado a un DataFrame plano.

        Args:
            report: Reporte anidado

        Returns:
            pd.DataFrame: Reporte plano
        """
        flat_data = {}

        for section, data in report.items():
            if isinstance(data, dict):
                for key, value in data.items():
                    flat_data[f"{section}_{key}"] = value
            else:
                flat_data[section] = data

        return pd.DataFrame([flat_data])
