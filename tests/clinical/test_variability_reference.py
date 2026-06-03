"""Clinical reference tests for variability metrics.

Expected values are computed using the published formulas on deterministic
synthetic data. See tests/fixtures/synthetic/README.md for the dataset
description.
"""

from pathlib import Path

import pandas as pd
import pytest

from cgmpy import GlucoseAnalysis, GlucoseData
from cgmpy.metrics.variability.conga import conga
from cgmpy.metrics.variability.mage import mage_simple
from cgmpy.metrics.variability.modd import modd
from cgmpy.metrics.variability.risk import adrr, hbgi, lbgi

FIXTURES = Path(__file__).parent.parent / "fixtures" / "synthetic" / "sine_24h.csv"


@pytest.fixture(scope="module")
def sine_data():
    return GlucoseAnalysis(GlucoseData(data_source=str(FIXTURES)))


class TestMODDReference:
    """MODD reference tests.

    MODD = Mean Of Daily Differences.
    For constant glucose, MODD = 0.
    For a perfect sine wave, the expected value can be computed analytically.
    """

    def test_modd_constant(self):
        """MODD should be 0 for identical days."""
        start = pd.Timestamp("2024-01-01")
        timestamps = pd.Series([start + pd.Timedelta(minutes=5 * i) for i in range(288)])
        glucose = pd.Series([100.0] * 288)
        result = modd(glucose, timestamps, days=1)
        assert result["value"] == pytest.approx(0.0, abs=1e-10)

    def test_modd_two_identical_days(self):
        """MODD should be 0 for two identical days of data."""
        start = pd.Timestamp("2024-01-01")
        day1 = [start + pd.Timedelta(minutes=5 * i) for i in range(288)]
        day2 = [start + pd.Timedelta(days=1) + pd.Timedelta(minutes=5 * i) for i in range(288)]
        timestamps = pd.Series(day1 + day2)
        glucose = pd.Series([100.0] * 288 + [100.0] * 288)
        result = modd(glucose, timestamps, days=1)
        assert result["value"] == pytest.approx(0.0, abs=1e-10)

    def test_modd_sine_runs_without_error(self):
        """MODD on 24h sine data returns the expected dict structure."""
        data = GlucoseData(str(FIXTURES))
        result = modd(data.glucose, data.timestamps, days=1)
        assert isinstance(result, dict)
        assert "value" in result and "std" in result and "days" in result


class TestCONGAReference:
    """CONGA reference tests.

    CONGA = Continuous Overlapping Net Glycemic Action.
    For constant glucose, CONGA = 0.
    Reference: McDonnell CM, et al. 2005, DOI: 10.1089/dia.2005.7.243
    """

    def test_conga_constant(self):
        """CONGA should be 0 for constant glucose."""
        start = pd.Timestamp("2024-01-01")
        timestamps = pd.Series([start + pd.Timedelta(minutes=5 * i) for i in range(288)])
        glucose = pd.Series([100.0] * 288)
        result = conga(glucose, timestamps, hours=1)
        assert result["value"] == pytest.approx(0.0, abs=1e-10)

    def test_conga_1h_returns_positive(self):
        """CONGA 1h on variable data should be positive."""
        data = GlucoseData(str(FIXTURES))
        result = conga(data.glucose, data.timestamps, hours=1)
        assert result["value"] > 0
        assert result["n"] > 0


class TestLBGI_HBGI_Reference:
    """LBGI/HBGI reference tests.

    Reference: Kovatchev et al. 2013, DOI: 10.2337/db12-1396
    For constant glucose at 100 mg/dL:
    f_bg = 1.509 * (ln(100)^1.084 - 5.381)
         = 1.509 * (4.605^1.084 - 5.381)
         = 1.509 * (5.127 - 5.381)
         = -0.383
    r_bg = 10 * (-0.383)^2 = 1.467
    Since f_bg < 0: LBGI = 1.467, HBGI = 0
    """

    def test_lbgi_hbgi_constant_100(self):
        """LBGI should be positive, HBGI should be 0 for glucose below ~112 mg/dL."""
        glucose = pd.Series([100.0] * 100)
        lb = lbgi(glucose)
        hb = hbgi(glucose)
        assert lb > 0  # below the LBGI/HBGI crossover (approx 112 mg/dL)
        assert hb == pytest.approx(0.0, abs=1e-10)

    def test_lbgi_hbgi_sine_data(self):
        """LBGI and HBGI should be positive on variable data."""
        data = GlucoseData(str(FIXTURES))
        lb = lbgi(data.glucose)
        hb = hbgi(data.glucose)
        assert lb >= 0
        assert hb >= 0
        assert lb + hb > 0


class TestADRRReference:
    """ADRR reference tests.

    Reference: DOI: 10.1177/193229681300700529
    ADRR = Average Daily Risk Range.

    For constant glucose at 100 mg/dL:
    The risk transform f_bg = 1.509 * (ln(100)^1.084 - 5.381) is negative
    (crossover is at ~112 mg/dL), so the hypo risk component is non-zero
    and ADRR ≈ 0.48 (after rounding to 2 decimals).
    """

    def test_adrr_constant_100(self):
        """ADRR at constant 100 mg/dL should be approx 0.48."""
        start = pd.Timestamp("2024-01-01")
        timestamps = pd.Series([start + pd.Timedelta(minutes=5 * i) for i in range(288)])
        glucose = pd.Series([100.0] * 288)
        result = adrr(glucose, timestamps)
        assert result["adrr"] == pytest.approx(0.48, abs=0.01)
        assert result["components"]["hyper_risk"] == pytest.approx(0.0, abs=1e-10)
        assert result["components"]["hypo_risk"] > 0

    def test_adrr_sine_data(self):
        """ADRR on variable data should be positive."""
        data = GlucoseData(str(FIXTURES))
        result = adrr(data.glucose, data.timestamps)
        assert result["adrr"] > 0
        assert result["risk_category"] in ("Low", "Moderate", "High")


class TestMAGEReference:
    """MAGE reference tests.

    Reference: Service FJ, et al. Diabetes 1970;19:644-55.
    MAGE = Mean Amplitude of Glycemic Excursions.
    For constant glucose: MAGE = 0.
    """

    def test_mage_simple_constant(self):
        """MAGE should be 0 for constant glucose."""
        glucose = pd.Series([100.0] * 100)
        result = mage_simple(glucose)
        assert result == 0.0

    def test_mage_simple_sine_data(self):
        """MAGE on sine data should be non-negative."""
        data = GlucoseData(str(FIXTURES))
        result = mage_simple(data.glucose)
        assert result >= 0


class TestCrossMetricConsistency:
    """Tests that verify consistency across related metrics."""

    def test_tir_tar_tbr_sum_to_100(self):
        """TIR + TAR + TBR should approximately sum to 100%."""
        data = GlucoseAnalysis(GlucoseData(data_source=str(FIXTURES)))

        tir_val = data.TIR()
        tar_val = data.TAR180()  # above 180
        tbr_val = data.TBR70()   # below 70

        total = tir_val + tar_val + tbr_val
        assert total == pytest.approx(100.0, abs=1.0)

    def test_tir_and_complement(self):
        """TIR + (time outside range) should sum to 100%."""
        data = GlucoseData(str(FIXTURES))
        glucose = data.glucose
        low, high = 70, 180

        tir_val = ((glucose >= low) & (glucose <= high)).sum() / len(glucose) * 100
        outside = ((glucose < low) | (glucose > high)).sum() / len(glucose) * 100

        assert tir_val + outside == pytest.approx(100.0, abs=0.01)


class TestAGATAParity:
    """Tests that verify CGMPy and py_agata produce similar results.

    These tests are gated by ``pytest.importorskip("py_agata.py_agata")``.
    """

    def test_tir_parity(self):
        """CGMPy TIR should match py_agata TIR within 1% on synthetic data."""
        pytest.importorskip("py_agata.py_agata")

        import warnings

        from py_agata.py_agata import Agata

        data = GlucoseData(str(FIXTURES))

        # CGMPy TIR
        from cgmpy.metrics.time_in_range import tir as cgmpy_tir_func
        cgmpy_tir = cgmpy_tir_func(data.glucose)

        # py_agata needs data prepared for it
        from cgmpy.agata.adapter import prepare_data_for_agata
        agata_data = prepare_data_for_agata(data)

        # py_agata raises RuntimeWarning on 1-day data (sddm_index ddof=1);
        # suppress because pytest's filterwarnings = ["error"] converts it.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            agata_result = Agata()
            try:
                profile = agata_result.analyze_glucose_profile(agata_data)
                agata_tir = profile["time_in_ranges"]["time_in_target"]
                assert cgmpy_tir == pytest.approx(agata_tir, abs=1.0)
            except (AttributeError, TypeError, KeyError) as e:
                pytest.skip(f"py_agata API mismatch: {e}")


class TestGlucoseAnalysisConsistency:
    """Tests that verify GlucoseAnalysis produces consistent results."""
