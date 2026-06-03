"""Tests for glucose unit conversion."""
import pandas as pd
import numpy as np
from cgmpy.metrics.units import (
    GlucoseUnit, to_mg_per_dl, to_mmol_per_l, convert, MGDL_TO_MMOLL,
)
import pytest


class TestGlucoseUnit:
    def test_enum_values(self):
        assert GlucoseUnit.MG_DL.value == "mg/dL"
        assert GlucoseUnit.MMOLL.value == "mmol/L"

    def test_from_string(self):
        assert GlucoseUnit("mg/dL") == GlucoseUnit.MG_DL
        assert GlucoseUnit("mmol/L") == GlucoseUnit.MMOLL


class TestConversion:
    def test_mgdl_to_mmol(self):
        """100 mg/dL ≈ 5.55 mmol/L"""
        result = to_mmol_per_l(100.0)
        assert result == pytest.approx(100 / MGDL_TO_MMOLL)

    def test_mmol_to_mgdl(self):
        """5.55 mmol/L ≈ 100 mg/dL"""
        result = to_mg_per_dl(5.55)
        assert result == pytest.approx(5.55 * MGDL_TO_MMOLL)

    def test_roundtrip(self):
        """Converting mg/dL → mmol/L → mg/dL should give original."""
        original = np.array([70.0, 100.0, 180.0, 250.0])
        mmol = to_mmol_per_l(original)
        back = to_mg_per_dl(mmol)
        assert original == pytest.approx(back)

    def test_series_conversion(self):
        s = pd.Series([70.0, 100.0, 180.0])
        result = to_mmol_per_l(s)
        assert isinstance(result, pd.Series)
        assert len(result) == 3

    def test_convert_same_unit(self):
        assert convert(100.0, GlucoseUnit.MG_DL, GlucoseUnit.MG_DL) == 100.0

    def test_identity(self):
        """convert should match individual functions."""
        assert convert(100.0, GlucoseUnit.MG_DL, GlucoseUnit.MMOLL) == to_mmol_per_l(100.0)
        assert convert(5.55, GlucoseUnit.MMOLL, GlucoseUnit.MG_DL) == to_mg_per_dl(5.55)


class TestGlucoseDataUnit:
    def test_default_unit(self):
        """GlucoseData should default to mg/dL."""
        from cgmpy import GlucoseData
        d = GlucoseData("tests/fixtures/synthetic/sine_24h.csv")
        assert d.unit == GlucoseUnit.MG_DL

    def test_glucose_in_unit(self):
        """glucose_in_unit(mg/dL) should match .glucose."""
        from cgmpy import GlucoseData
        d = GlucoseData("tests/fixtures/synthetic/sine_24h.csv")
        assert d.glucose_in_unit(GlucoseUnit.MG_DL) is d.glucose
