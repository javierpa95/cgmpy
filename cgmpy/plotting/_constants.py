"""Shared display constants for glucose plots.

These are the standard international-consensus reference bands used for
*visualisation* (Battelino et al., Diabetes Care 2019). They describe how the
glucose axis is shaded and where reference lines are drawn, independent of any
individual patient's :class:`~cgmpy.metrics.targets.GlucoseTargets`. Keeping
them here avoids scattering magic numbers across the plotting modules.
"""

# Glucose axis bounds (mg/dL)
GLUCOSE_AXIS_MIN = 0
GLUCOSE_AXIS_MAX = 400
# Histogram spans the full physiological range up to severe hyper.
GLUCOSE_HIST_MAX = 500

# Standard glycemic zone thresholds (mg/dL)
SEVERE_HYPO = 54  # Level-2 hypoglycemia
HYPO = 70  # Level-1 hypoglycemia / lower target bound
TARGET_LOW = 70
TARGET_HIGH = 180  # Upper target bound
SEVERE_HYPER = 250  # Level-2 hyperglycemia
