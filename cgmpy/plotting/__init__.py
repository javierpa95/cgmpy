"""
Glucose data visualization module.

This module contains the classes and functions for:
- Ambulatory glucose profiles (AGP)
- Daily and overlay plots
- Statistical plots: histograms, boxplots
- Dashboards and combined visualisations
"""

# Available imports
from .agp import AGPPlotter
from .daily_plots import DailyPlotter
from .statistical_plots import StatisticalPlotter


# Combined class that integrates all modular plotters
class ModularGlucosePlot(AGPPlotter, DailyPlotter, StatisticalPlotter):
    """
    Class that combines all modular plotters.

    This class allows using the plots in a modular way while
    maintaining compatibility with the existing interface.
    """

    pass


__all__ = ["AGPPlotter", "DailyPlotter", "ModularGlucosePlot", "StatisticalPlotter"]
