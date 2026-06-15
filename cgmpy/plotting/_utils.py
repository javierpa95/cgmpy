"""Plotting utilities."""

from typing import Any, cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ._constants import (
    GLUCOSE_HIST_MAX,
    HYPO,
    SEVERE_HYPER,
    SEVERE_HYPO,
    TARGET_HIGH,
)


def resolve_axes(ax: Axes | None, **subplots_kwargs: Any) -> tuple[Figure, Axes, bool]:
    """Resolve an optional axis into a concrete ``(figure, axes, created)``.

    If ``ax`` is ``None`` a new figure/axis is created with
    ``subplots_kwargs``; otherwise the caller's axis (and its figure) is
    reused. ``created`` tells the caller whether it owns the figure (and may,
    e.g., call ``tight_layout``).

    Args:
        ax: An existing axis to draw into, or ``None`` to create a new figure.
        **subplots_kwargs: Forwarded to ``plt.subplots`` when creating a figure.

    Returns:
        Tuple of ``(figure, axes, created)``.
    """
    if ax is None:
        fig, ax = plt.subplots(**subplots_kwargs)
        return fig, ax, True
    return cast(Figure, ax.get_figure()), ax, False


def add_glucose_zones(
    ax: plt.Axes,
    low_threshold: float = HYPO,
    high_threshold: float = TARGET_HIGH,
    very_high_threshold: float = SEVERE_HYPER,
    alpha_hypo: float = 0.15,
    alpha_hyper: float = 0.15,
) -> None:
    """Add coloured glucose zone backgrounds to an axis.

    Args:
        ax: Matplotlib axis to draw on.
        low_threshold: Hypoglycemia threshold (default 70 mg/dL).
        high_threshold: Hyperglycemia threshold (default 180 mg/dL).
        very_high_threshold: Very high threshold (default 250 mg/dL).
        alpha_hypo: Transparency for hypoglycemia zones.
        alpha_hyper: Transparency for hyperglycemia zones.
    """
    # Severe hypoglycemia (< 54)
    ax.axvspan(0, SEVERE_HYPO, color="#ff0000", alpha=alpha_hypo, label="Severe hypoglycemia")
    # Hypoglycemia (54 - low_threshold)
    ax.axvspan(SEVERE_HYPO, low_threshold, color="#ffaa00", alpha=alpha_hypo, label="Hypoglycemia")
    # Target range (low_threshold - high_threshold)
    ax.axvspan(low_threshold, high_threshold, color="#00ff00", alpha=0.1, label="Target range")
    # Hyperglycemia (high_threshold - very_high_threshold)
    ax.axvspan(
        high_threshold,
        very_high_threshold,
        color="#ffaa00",
        alpha=alpha_hyper,
        label="Hyperglycemia",
    )
    # Severe hyperglycemia (> very_high_threshold)
    ax.axvspan(
        very_high_threshold,
        GLUCOSE_HIST_MAX,
        color="#ff0000",
        alpha=alpha_hyper,
        label="Severe hyperglycemia",
    )
