"""Bridge between CGMPy and the optional ``py_agata`` reference library.

The :mod:`py_agata` package is an **optional** dependency of CGMPy. The
top-level :func:`import cgmpy` call must not require it; only the call
sites in this module do. We import lazily and surface a clean
:class:`~cgmpy.errors.AgataNotInstalledError` when the library is missing.
"""

from __future__ import annotations

from typing import Any

from ..data.core import GlucoseData
from ..errors import AgataNotInstalledError
from .adapter import prepare_data_for_agata

# Lazy import: py_agata is an optional dependency. We allow `import cgmpy`
# to succeed even when py_agata is not installed; the missing-dependency
# error is raised only when the analysis functions are actually called.
try:
    from py_agata.py_agata import Agata
except ImportError:
    Agata = None


def analyze_one_arm(
    data_list: list[GlucoseData], glycemic_target: str = "diabetes", **kwargs
) -> dict:
    """
    Analyzes a group (arm) of glucose data using the py_agata library.

    Args:
        data_list (list[GlucoseData]): List of GlucoseData objects to analyze.
        glycemic_target (str): The glycemic target to use.

    Returns:
        dict: The results dictionary returned by py_agata with group statistics.

    Raises:
        AgataNotInstalledError: If ``py_agata`` is not installed.
        EmptyDataError: If any of the input datasets is empty (propagated
            from :func:`prepare_data_for_agata`).
    """
    if Agata is None:
        raise AgataNotInstalledError()

    # 1. Prepare all DataFrames
    prepared_dfs = [prepare_data_for_agata(d) for d in data_list]

    # 2. Instantiate Agata and run the group analysis
    analyzer = Agata(glycemic_target=glycemic_target)
    return analyzer.analyze_one_arm(prepared_dfs)


def analyze_with_agata(
    glucose_data: GlucoseData, glycemic_target: str = "diabetes", **kwargs
) -> dict:
    """Analyze a single :class:`GlucoseData` using ``py_agata``.

    Raises:
        AgataNotInstalledError: If ``py_agata`` is not installed.
        EmptyDataError: If the input dataset is empty (propagated from
            :func:`prepare_data_for_agata`).
    """
    if Agata is None:
        raise AgataNotInstalledError()

    # 1. Prepare the data with the adapter
    df_for_agata = prepare_data_for_agata(glucose_data)

    # 2. Instantiate Agata with the correct target
    analyzer = Agata(glycemic_target=glycemic_target)

    # 3. Run the analysis
    return analyzer.analyze_glucose_profile(df_for_agata)


def summarize_agata_results(results: dict) -> dict:
    """
    Converts the py_agata nested dictionary into a flat summary (variable: value),
    removing the extensive event arrays and keeping only the averages.
    """
    summary = {}

    for category, metrics in results.items():
        if category == "events":
            # For events, iterate by type (hypo, hyper, etc) and level (l1, l2)
            for event_type, levels in metrics.items():
                if isinstance(levels, dict):
                    for level, data in levels.items():
                        if isinstance(data, dict):
                            # Build clear names: e.g. "hypo_l1_mean_duration"
                            prefix = f"{event_type}_{level}"
                            if "mean_duration" in data:
                                summary[f"{prefix}_mean_duration"] = data["mean_duration"]
                            if "events_per_week" in data:
                                summary[f"{prefix}_per_week"] = data["events_per_week"]
        else:
            # For the rest, just flatten one level
            if isinstance(metrics, dict):
                for name, value in metrics.items():
                    summary[f"{category}_{name}"] = value
            else:
                summary[category] = metrics

    return summary


# --- NEW CLASS ---
class AgataAnalysis(GlucoseData):
    """Object-oriented bridge to the ``py_agata`` analysis pipeline.

    Inherits from :class:`~cgmpy.data.core.GlucoseData` so it can
    be constructed in exactly the same way (file path, DataFrame, etc.),
    and exposes :meth:`run` / :meth:`analyze_one_arm` to drive the
    py_agata analysis.

    Raises:
        AgataNotInstalledError: If ``py_agata`` is not installed.
        EmptyDataError: If :meth:`run` (or :meth:`analyze_one_arm`) is
            called on empty data, raised by the adapter during data
            preparation.
    """

    def __init__(self, *args, glycemic_target: str = "diabetes", **kwargs: Any):
        """
        Initializes the Agata analysis.

        Args:
            *args: Positional arguments for GlucoseData.
            glycemic_target (str): The glycemic target ('diabetes' or 'pregnancy').
            **kwargs: Other arguments for GlucoseData.
        """
        super().__init__(*args, **kwargs)
        self.glycemic_target = glycemic_target

    def run(self, glycemic_target: str = "diabetes", summary: bool = False, **kwargs: Any) -> dict:
        """
        Runs the full py_agata analysis pipeline on the loaded data.

        Args:
            glycemic_target (str): The glycemic target to use ('diabetes', 'pregnancy', etc.).
                                   If None, uses the value defined at class initialization.
            summary (bool): If True, returns a flat dictionary with summary metrics.
                            If False, returns the complete nested dictionary.
            **kwargs: Additional arguments for future use.

        Returns:
            dict: The results dictionary.

        Raises:
            EmptyDataError: If the underlying data is empty.
            AgataNotInstalledError: If ``py_agata`` is not installed.
        """
        # Use the defined target if no specific one is passed in the call
        target = glycemic_target or self.glycemic_target

        # 1. Get raw results
        results = analyze_with_agata(self, glycemic_target=target, **kwargs)

        # 2. Summarize if requested
        if summary:
            return summarize_agata_results(results)

        return results

    @classmethod
    def analyze_one_arm(
        cls,
        data_list: list[GlucoseData],
        glycemic_target: str = "diabetes",
        summary: bool = False,
        **kwargs,
    ) -> dict:
        """
        Analyzes a group (arm) of GlucoseData objects.

        Args:
            data_list (list[GlucoseData]): List of objects to analyze.
            glycemic_target (str): The glycemic target to use.
            summary (bool): If True, returns a flat summary.

        Raises:
            EmptyDataError: If any element of ``data_list`` is empty.
            AgataNotInstalledError: If ``py_agata`` is not installed.
        """
        results = analyze_one_arm(data_list, glycemic_target=glycemic_target, **kwargs)

        if summary:
            # Note: py_agata.analyze_one_arm usually returns a similar format
            return summarize_agata_results(results)

        return results
