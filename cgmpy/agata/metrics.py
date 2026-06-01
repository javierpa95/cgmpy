try:
    from py_agata.py_agata import Agata
except ImportError:
    # We don't fail here, allow the package to be imported.
    # The error will occur if the functionality is used.
    Agata = None

from ..data.core import ModularGlucoseData
from .adapter import prepare_data_for_agata

"""Comments
def time_in_given_range(data, th_l, th_h, include_th_l=False, include_th_h=False)

def time_in_target(data, glycemic_target='diabetes'):
    # By default, not include thresholds. I Change to True
    return time_in_given_range(data=data, th_l=th_l, th_h=th_h, include_th_l=True, include_th_h=True)

"""


def analyze_one_arm(
    data_list: list[ModularGlucoseData], glycemic_target: str = "diabetes", **kwargs
) -> dict:
    """
    Analyzes a group (arm) of glucose data using the py_agata library.

    Args:
        data_list (list[ModularGlucoseData]): List of ModularGlucoseData objects to analyze.
        glycemic_target (str): The glycemic target to use.

    Returns:
        dict: The results dictionary returned by py_agata with group statistics.
    """
    if Agata is None:
        raise ImportError(
            "The 'py_agata' library is required for this functionality. Please install it."
        )

    # 1. Prepare all DataFrames
    prepared_dfs = [prepare_data_for_agata(d) for d in data_list]

    try:
        # 2. Instantiate Agata and run the group analysis
        analyzer = Agata(glycemic_target=glycemic_target)
        results = analyzer.analyze_one_arm(prepared_dfs)

        return results

    except Exception as e:
        raise


def analyze_with_agata(
    glucose_data: ModularGlucoseData, glycemic_target: str = "diabetes", **kwargs
) -> dict:
    """
    Analyzes a cgmpy data object using the py_agata library.
    """
    if Agata is None:
        raise ImportError(
            "The 'py_agata' library is required for this functionality. Please install it."
        )

    # 1. Prepare the data with the adapter
    df_for_agata = prepare_data_for_agata(glucose_data)

    try:
        # 2. Instantiate Agata with the correct target
        analyzer = Agata(glycemic_target=glycemic_target)

        # 3. Run the analysis
        results = analyzer.analyze_glucose_profile(df_for_agata)

        return results

    except Exception as e:
        raise


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
class AgataAnalysis(ModularGlucoseData):
    """
    Container class to analyze glucose data using the py_agata library.

    This class acts as an object-oriented bridge, allowing data loading
    and the py_agata analysis pipeline to be executed in a simple way.
    """

    def __init__(self, *args, glycemic_target: str = "diabetes", **kwargs):
        """
        Initializes the Agata analysis.

        Args:
            *args: Positional arguments for ModularGlucoseData.
            glycemic_target (str): The glycemic target ('diabetes' or 'pregnancy').
            **kwargs: Other arguments for ModularGlucoseData.
        """
        super().__init__(*args, **kwargs)
        self.glycemic_target = glycemic_target

    def run(self, glycemic_target: str = "diabetes", summary: bool = False, **kwargs) -> dict:
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
        data_list: list[ModularGlucoseData],
        glycemic_target: str = "diabetes",
        summary: bool = False,
        **kwargs,
    ) -> dict:
        """
        Analyzes a group (arm) of ModularGlucoseData objects.

        Args:
            data_list (list[ModularGlucoseData]): List of objects to analyze.
            glycemic_target (str): The glycemic target to use.
            summary (bool): If True, returns a flat summary.
        """
        results = analyze_one_arm(data_list, glycemic_target=glycemic_target, **kwargs)

        if summary:
            # Note: py_agata.analyze_one_arm usually returns a similar format
            return summarize_agata_results(results)

        return results
