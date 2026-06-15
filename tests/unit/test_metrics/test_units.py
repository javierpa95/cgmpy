"""Tests for glucose unit conversion."""

import numpy as np
import pandas as pd
import pytest

from cgmpy.metrics.units import (
    MGDL_TO_MMOLL,
    GlucoseUnit,
    convert,
    to_mg_per_dl,
    to_mmol_per_l,
)


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
        """glucose_in_unit(mg/dL) should equal .glucose."""
        from cgmpy import GlucoseData

        d = GlucoseData("tests/fixtures/synthetic/sine_24h.csv")
        result = d.glucose_in_unit(GlucoseUnit.MG_DL)
        assert result.equals(d.glucose)


class TestMmolNormalization:
    """mmol/L input must be normalized to mg/dL so every metric is unit-safe."""

    @staticmethod
    def _mgdl_df() -> pd.DataFrame:
        times = pd.date_range("2024-01-01", periods=288, freq="5min")
        glucose = 120 + 40 * np.sin(np.linspace(0, 4 * np.pi, 288))
        return pd.DataFrame({"time": times, "glucose": glucose})

    def test_mmol_input_is_stored_as_mgdl(self):
        from cgmpy import GlucoseData

        mgdl = self._mgdl_df()
        mmol = mgdl.assign(glucose=mgdl["glucose"] / MGDL_TO_MMOLL)

        d = GlucoseData(data_source=mmol, unit="mmol/L")

        # Stored values are mg/dL; the original unit is remembered separately.
        assert d.unit == GlucoseUnit.MG_DL
        assert d.source_unit == GlucoseUnit.MMOLL
        assert d.glucose.to_numpy() == pytest.approx(mgdl["glucose"].to_numpy(), rel=1e-6)

    def test_glucose_in_unit_roundtrips_to_source(self):
        from cgmpy import GlucoseData

        mgdl = self._mgdl_df()
        mmol = mgdl.assign(glucose=mgdl["glucose"] / MGDL_TO_MMOLL)

        d = GlucoseData(data_source=mmol, unit="mmol/L")
        back = d.glucose_in_unit("mmol/L")
        assert back.to_numpy() == pytest.approx(mmol["glucose"].to_numpy(), rel=1e-6)

    def test_metrics_match_between_mgdl_and_mmol_input(self):
        """TIR/mean/GMI must be identical whether the data was given in mg/dL or mmol/L."""
        from cgmpy import GlucoseAnalysis, GlucoseData

        mgdl = self._mgdl_df()
        mmol = mgdl.assign(glucose=mgdl["glucose"] / MGDL_TO_MMOLL)

        a_mgdl = GlucoseAnalysis(GlucoseData(data_source=mgdl, unit="mg/dL"))
        a_mmol = GlucoseAnalysis(GlucoseData(data_source=mmol, unit="mmol/L"))

        assert a_mmol.mean() == pytest.approx(a_mgdl.mean(), rel=1e-6)
        assert a_mmol.TIR() == pytest.approx(a_mgdl.TIR(), rel=1e-6)
        assert a_mmol.gmi() == pytest.approx(a_mgdl.gmi(), rel=1e-6)
        assert a_mmol.MAGE() == pytest.approx(a_mgdl.MAGE(), rel=1e-6)
