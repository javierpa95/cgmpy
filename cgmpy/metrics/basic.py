"""
Basic metrics module for glucose data.

This module contains fundamental metrics for glucose data analysis:
- Basic descriptive statistics (mean, median, percentiles)
- Standard deviation and coefficient of variation
- Glucose Management Index (GMI)
- Distribution analysis
"""

from typing import Any, Dict


class BasicMetrics:
    """
    Base class for basic glucose metrics.

    This class should be used as a mixin with GlucoseData.
    """

    # Basic descriptive metrics
    def mean(self) -> float:
        """
        Calculates mean glucose.

        Returns:
            float: Mean glucose in mg/dL
        """
        return self.data["glucose"].mean()

    def median(self) -> float:
        """
        Calculates median glucose.

        Returns:
            float: Median glucose in mg/dL
        """
        return self.data["glucose"].median()

    def percentile(self, percentile: float) -> float:
        """
        Calculates glucose percentile.

        Args:
            percentile: Percentile to calculate (0-100)

        Returns:
            float: Percentile value in mg/dL
        """
        return self.data["glucose"].quantile(percentile / 100)

    def sd(self) -> float:
        """
        Calculates glucose standard deviation.

        Returns:
            float: Standard deviation in mg/dL
        """
        return self.data["glucose"].std()

    def cv(self) -> float:
        """
        Calculates coefficient of variation.

        Returns:
            float: Coefficient of variation in percentage
        """
        return (self.sd() / self.mean()) * 100

    def gmi(self) -> float:
        """
        Calculates Glucose Management Index (GMI).

        GMI is an HbA1c estimation based on CGM data.

        Returns:
            float: GMI (HbA1c estimation in %)

        Reference:
            DOI: 10.2337/dc18-1581
        """
        return round(3.31 + (0.02392 * self.mean()), 2)

    def distribution_analysis(self) -> Dict[str, Any]:
        """
        Analyzes glucose value distribution.

        Returns:
            dict: Dictionary with distribution statistics
        """
        stats = {
            "mean": self.mean(),
            "median": self.median(),
            "std": self.sd(),
            "cv": self.cv(),
            "skewness": self.data["glucose"].skew(),
            "kurtosis": self.data["glucose"].kurtosis(),
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

    def calculate_all_metrics(self) -> Dict[str, float]:
        """
        Summary of basic statistics.

        Returns:
            dict: Complete summary of basic metrics
        """
        return {
            "GMI": self.gmi(),
            "Mean": self.mean(),
            "Median": self.median(),
            "Std": self.sd(),
            "CV": self.cv(),
            "P5": self.percentile(5),
            "P25": self.percentile(25),
            "P75": self.percentile(75),
            "P95": self.percentile(95),
            "Skewness": self.data["glucose"].skew(),
            "Kurtosis": self.data["glucose"].kurtosis(),
        }

    def __str__(self) -> str:
        """
        String representation of basic metrics in a simple and readable format.
        """
        stats = self.calculate_all_metrics()

        # Formatting values with appropriate units
        gmi_str = f"{stats['GMI']:.1f}%"
        media_str = f"{stats['Mean']:.1f} mg/dL"
        mediana_str = f"{stats['Median']:.1f} mg/dL"
        sd_str = f"{stats['Std']:.1f} mg/dL"
        cv_str = f"{stats['CV']:.1f}%"

        # Percentiles
        p5_str = f"{stats['P5']:.1f} mg/dL"
        p25_str = f"{stats['P25']:.1f} mg/dL"
        p75_str = f"{stats['P75']:.1f} mg/dL"
        p95_str = f"{stats['P95']:.1f} mg/dL"

        # Shape statistics
        asimetria_str = f"{stats['Skewness']:.2f}"
        curtosis_str = f"{stats['Kurtosis']:.2f}"

        summary = (
            "BASIC GLUCOSE METRICS\n"
            "\n"
            "CENTRAL STATISTICS:\n"
            f"  - GMI (HbA1c est.):       {gmi_str:>8}\n"
            f"  - Mean:                  {media_str:>12}\n"
            f"  - Median:                {mediana_str:>12}\n"
            f"  - Std Dev:               {sd_str:>12}\n"
            f"  - CV:                    {cv_str:>12}\n"
            "\n"
            "PERCENTILES:\n"
            f"  - P5:                    {p5_str:>12}\n"
            f"  - P25:                   {p25_str:>12}\n"
            f"  - P75:                   {p75_str:>12}\n"
            f"  - P95:                   {p95_str:>12}\n"
            "\n"
            "DISTRIBUTION SHAPE:\n"
            f"  - Skewness:              {asimetria_str:>8}\n"
            f"  - Kurtosis:              {curtosis_str:>8}\n"
        )
        return summary
