from dataclasses import dataclass


@dataclass
class GlucoseTargets:
    """Class to define glucose targets for metrics calculation."""

    hypo_level2: float
    hypo_level1: float
    target_low: float
    target_high: float
    hyper_level1: float
    hyper_level2: float
    name: str = "Standard"

    @classmethod
    def standard(cls):
        """
        Standard targets for general diabetes.
        Level 2 hypo < 54, Level 1 hypo 54-70, TIR 70-180, TAR 180-250, TAR > 250.
        """
        return cls(
            hypo_level2=54,
            hypo_level1=70,  # Below 70
            target_low=70,
            target_high=180,
            hyper_level1=180,  # Above 180
            hyper_level2=250,  # Above 250
            name="Diabetes",
        )

    @classmethod
    def pregnancy(cls):
        """
        Specific targets for pregnancy.
        Level 2 hypo < 55, Level 1 hypo 55-63, TIR 63-140, TAR 140-250, TAR > 250.
        """
        return cls(
            hypo_level2=55,
            hypo_level1=63,  # Below 63
            target_low=63,
            target_high=140,
            hyper_level1=140,  # Above 140
            hyper_level2=250,  # Above 250
            name="Pregnancy",
        )


_VALID_TARGET_TYPES = frozenset({"diabetes", "pregnancy"})


def get_targets(target_type: str = "diabetes") -> GlucoseTargets:
    """
    Factory function to get glucose targets.

    Args:
        target_type: Either 'diabetes' (default) or 'pregnancy'.

    Returns:
        GlucoseTargets object.

    Raises:
        ValueError: If `target_type` is not a recognised profile name.
    """
    normalized = target_type.lower()
    if normalized not in _VALID_TARGET_TYPES:
        valid = ", ".join(sorted(_VALID_TARGET_TYPES))
        raise ValueError(f"Unknown target type {target_type!r}. Valid options are: {valid}.")
    if normalized == "pregnancy":
        return GlucoseTargets.pregnancy()
    return GlucoseTargets.standard()
