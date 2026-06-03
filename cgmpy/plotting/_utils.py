"""Plotting utilities."""

import matplotlib.pyplot as plt


def add_glucose_zones(
    ax: plt.Axes,
    low_threshold: float = 70,
    high_threshold: float = 180,
    very_high_threshold: float = 250,
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
    ax.axvspan(0, 54, color="#ff0000", alpha=alpha_hypo, label="Severe hypoglycemia")
    # Hypoglycemia (54 - low_threshold)
    ax.axvspan(54, low_threshold, color="#ffaa00", alpha=alpha_hypo, label="Hypoglycemia")
    # Target range (low_threshold - high_threshold)
    ax.axvspan(low_threshold, high_threshold, color="#00ff00", alpha=0.1, label="Target range")
    # Hyperglycemia (high_threshold - very_high_threshold)
    ax.axvspan(high_threshold, very_high_threshold, color="#ffaa00", alpha=alpha_hyper, label="Hyperglycemia")
    # Severe hyperglycemia (> very_high_threshold)
    ax.axvspan(very_high_threshold, 500, color="#ff0000", alpha=alpha_hyper, label="Severe hyperglycemia")
