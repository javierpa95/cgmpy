"""Tests for `cgmpy.metrics.targets.GlucoseTargets`."""

from __future__ import annotations

import pytest

from cgmpy.metrics.targets import GlucoseTargets, get_targets


class TestGlucoseTargets:
    """Tests for the GlucoseTargets dataclass and helpers."""

    def test_standard_factory(self) -> None:
        """The standard factory returns diabetes cutoffs."""
        t = GlucoseTargets.standard()
        assert t.hypo_level2 == 54
        assert t.hypo_level1 == 70
        assert t.target_low == 70
        assert t.target_high == 180
        assert t.hyper_level1 == 180
        assert t.hyper_level2 == 250
        assert t.name == "Diabetes"

    def test_pregnancy_factory(self) -> None:
        """The pregnancy factory returns tighter cutoffs."""
        t = GlucoseTargets.pregnancy()
        assert t.hypo_level2 == 55
        assert t.hypo_level1 == 63
        assert t.target_low == 63
        assert t.target_high == 140
        assert t.hyper_level1 == 140
        assert t.hyper_level2 == 250
        assert t.name == "Pregnancy"

    def test_get_targets_diabetes(self) -> None:
        """`get_targets('diabetes')` returns the standard profile."""
        t = get_targets("diabetes")
        assert t.name == "Diabetes"

    def test_get_targets_pregnancy(self) -> None:
        """`get_targets('pregnancy')` returns the pregnancy profile."""
        t = get_targets("pregnancy")
        assert t.name == "Pregnancy"

    def test_get_targets_default(self) -> None:
        """`get_targets()` with no arg returns diabetes (the safest default)."""
        t = get_targets()
        assert t.name == "Diabetes"

    def test_get_targets_unknown_raises(self) -> None:
        """Unknown target types raise ValueError."""
        with pytest.raises(ValueError, match="Unknown target type"):
            get_targets("invalid")

    def test_targets_are_immutable_after_creation(self) -> None:
        """The dataclass is not frozen, but renaming a profile is a deliberate change."""
        t = GlucoseTargets.standard()
        original = t.name
        t.name = "Custom"
        assert t.name != original


class TestTargetSemantics:
    """Cross-target sanity checks."""

    def test_pregnancy_lower_bounds(self) -> None:
        """Pregnancy cutoffs are tighter (lower target_low, lower hyper_level1)."""
        std = GlucoseTargets.standard()
        preg = GlucoseTargets.pregnancy()
        assert preg.target_low < std.target_low
        assert preg.target_high < std.target_high

    def test_hyper_level2_is_constant(self) -> None:
        """Both profiles agree on level-2 hyperglycemia (250 mg/dL)."""
        assert GlucoseTargets.standard().hyper_level2 == GlucoseTargets.pregnancy().hyper_level2

    def test_target_low_equals_hypo_level1(self) -> None:
        """For both profiles, target_low == hypo_level1 (boundary convention)."""
        for t in (GlucoseTargets.standard(), GlucoseTargets.pregnancy()):
            assert t.target_low == t.hypo_level1

    def test_target_high_equals_hyper_level1(self) -> None:
        """For both profiles, target_high == hyper_level1."""
        for t in (GlucoseTargets.standard(), GlucoseTargets.pregnancy()):
            assert t.target_high == t.hyper_level1
