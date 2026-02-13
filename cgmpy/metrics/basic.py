"""
Módulo de métricas básicas para datos de glucosa.

Este módulo contiene las métricas fundamentales para el análisis de datos de glucosa:
- Estadísticas descriptivas básicas (media, mediana, percentiles)
- Desviación estándar y coeficiente de variación
- Glucose Management Index (GMI)
- Análisis de distribución
"""

from typing import Any, Dict


class BasicMetrics:
    """
    Clase base para métricas básicas de glucosa.

    Esta clase debe ser utilizada como mixin con GlucoseData.
    """

    # Métricas descriptivas básicas
    def mean(self) -> float:
        """
        Calcula la glucemia media.

        Returns:
            float: Media de glucosa en mg/dL
        """
        return self.data["glucose"].mean()

    def median(self) -> float:
        """
        Calcula la mediana de la glucemia.

        Returns:
            float: Mediana de glucosa en mg/dL
        """
        return self.data["glucose"].median()

    def percentile(self, percentile: float) -> float:
        """
        Calcula el percentil de la glucemia.

        Args:
            percentile: Percentil a calcular (0-100)

        Returns:
            float: Valor del percentil en mg/dL
        """
        return self.data["glucose"].quantile(percentile / 100)

    def sd(self) -> float:
        """
        Calcula la desviación estándar de la glucemia.

        Returns:
            float: Desviación estándar en mg/dL
        """
        return self.data["glucose"].std()

    def cv(self) -> float:
        """
        Calcula el coeficiente de variación.

        Returns:
            float: Coeficiente de variación en porcentaje
        """
        return (self.sd() / self.mean()) * 100

    def gmi(self) -> float:
        """
        Calcula el Glucose Management Index (GMI).

        El GMI es una estimación de la HbA1c basada en los datos de CGM.

        Returns:
            float: GMI (estimación de HbA1c en %)

        Reference:
            DOI: 10.2337/dc18-1581
        """
        return round(3.31 + (0.02392 * self.mean()), 2)

    def distribution_analysis(self) -> Dict[str, Any]:
        """
        Analiza la distribución de los valores de glucosa.

        Returns:
            dict: Diccionario con estadísticas de distribución
        """
        stats = {
            "media": self.mean(),
            "mediana": self.median(),
            "desviacion_estandar": self.sd(),
            "coef_variacion": self.cv(),
            "asimetria": self.data["glucose"].skew(),
            "curtosis": self.data["glucose"].kurtosis(),
            "percentiles": {
                "p5": self.percentile(5),
                "p25": self.percentile(25),
                "p50": self.percentile(50),
                "p75": self.percentile(75),
                "p95": self.percentile(95),
                "IQR": self.percentile(75) - self.percentile(25),
            },
        }
        return stats

    def calculate_all_metrics(self) -> Dict[str, Any]:
        """
        Resumen de estadísticas básicas.

        Returns:
            dict: Resumen completo de métricas básicas
        """
        return {
            "GMI": self.gmi(),
            "Media": self.mean(),
            "Mediana": self.median(),
            "Desviacion_estandar": self.sd(),
            "CV": self.cv(),
            "P5": self.percentile(5),
            "P25": self.percentile(25),
            "P75": self.percentile(75),
            "P95": self.percentile(95),
            "Asimetria": self.data["glucose"].skew(),
            "Curtosis": self.data["glucose"].kurtosis(),
        }

    def __str__(self) -> str:
        """
        Representación en string de las métricas básicas en formato simple y legible.
        """
        stats = self.basic_statistics_summary()

        # Formateo de valores con unidades apropiadas
        gmi_str = f"{stats['GMI']:.1f}%"
        media_str = f"{stats['Media']:.1f} mg/dL"
        mediana_str = f"{stats['Mediana']:.1f} mg/dL"
        sd_str = f"{stats['Desviacion_estandar']:.1f} mg/dL"
        cv_str = f"{stats['CV']:.1f}%"

        # Percentiles
        p5_str = f"{stats['P5']:.1f} mg/dL"
        p25_str = f"{stats['P25']:.1f} mg/dL"
        p75_str = f"{stats['P75']:.1f} mg/dL"
        p95_str = f"{stats['P95']:.1f} mg/dL"

        # Estadísticas de forma
        asimetria_str = f"{stats['Asimetria']:.2f}"
        curtosis_str = f"{stats['Curtosis']:.2f}"

        summary = (
            "MÉTRICAS BÁSICAS DE GLUCOSA\n"
            "\n"
            "ESTADÍSTICAS CENTRALES:\n"
            f"  - GMI (HbA1c estimada):   {gmi_str:>8}\n"
            f"  - Media:                 {media_str:>12}\n"
            f"  - Mediana:               {mediana_str:>12}\n"
            f"  - Desv. Estándar:        {sd_str:>12}\n"
            f"  - Coef. Variación:       {cv_str:>12}\n"
            "\n"
            "PERCENTILES:\n"
            f"  - P5:                    {p5_str:>12}\n"
            f"  - P25:                   {p25_str:>12}\n"
            f"  - P75:                   {p75_str:>12}\n"
            f"  - P95:                   {p95_str:>12}\n"
            "\n"
            "FORMA DE LA DISTRIBUCIÓN:\n"
            f"  - Asimetría:             {asimetria_str:>8}\n"
            f"  - Curtosis:              {curtosis_str:>8}\n"
        )
        return summary
