"""Variability metrics package.

This package splits the original ``variability.py`` module into one
file per metric family (SD, MAGE, MODD, CONGA, Lability, Risk).

Individual mixin classes are exported for users who only need
a subset of metrics::

    from cgmpy.metrics.variability import SDMetrics, MAGEMetrics
"""

from ._base import VariabilityBase
from .conga import CONGAMetrics
from .lability import LabilityMetrics
from .mage import MAGEMetrics
from .modd import MODDMetrics
from .risk import RiskMetrics
from .sd import SDMetrics

__all__ = [
    "CONGAMetrics",
    "LabilityMetrics",
    "MAGEMetrics",
    "MODDMetrics",
    "RiskMetrics",
    "SDMetrics",
    "VariabilityBase",
]
