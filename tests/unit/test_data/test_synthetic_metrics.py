"""Exact-value metric assertions on the deterministic v0.5.2 fixtures.

These tests use the hand-computed ground truth documented in:

* ``tests/fixtures/synthetic/README.md`` — sine 24 h dataset
* ``tests/fixtures/devices/README.md`` — 4 constant-120 device CSVs

The intent is to lock the metric math to **known** values, byte-for-byte.
If a future refactor shifts the implementation, the regression is detected
here before it reaches the user.

The test module does not require ``py_agata``.

Important: specialized loaders (``Dexcom``, ``Libreview``,
``MedtronicCarelink``, ``TandemDiabetes``) do **not** include the metrics
mixin. We load with the specialized loader to exercise its contract, then
re-wrap the cleaned DataFrame in :class:`GlucoseMetrics` to assert metrics.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cgmpy import (
    Dexcom,
    GlucoseAnalysis,
    GlucoseData,
    Libreview,
    MedtronicCarelink,
    TandemDiabetes,
)
from cgmpy.data.specialized import detect_device_type

# ------------------------------------------------------------------ paths
FIXTURES_DEVICES = Path(__file__).resolve().parents[2] / "fixtures" / "devices"
FIXTURES_SYNTHETIC = Path(__file__).resolve().parents[2] / "fixtures" / "synthetic"

# Reused path handles (all 4 device CSVs are 288 rows of constant 120 mg/dL).
DEXCOM_PATH = FIXTURES_DEVICES / "dexcom_constant_120.csv"
LIBREVIEW_PATH = FIXTURES_DEVICES / "libreview_constant_120.csv"
MEDTRONIC_PATH = FIXTURES_DEVICES / "medtronic_constant_120.csv"
TANDEM_PATH = FIXTURES_DEVICES / "tandem_constant_120.csv"

SINE_PATH = FIXTURES_SYNTHETIC / "sine_24h.csv"


# ----------------------------------------------------------- helpers
def _metrics_for_dataframe(df: pd.DataFrame) -> GlucoseAnalysis:
    """Wrap a cleaned DataFrame (with ``time``/``glucose`` cols) into ``GlucoseAnalysis``."""
    return GlucoseAnalysis(data=GlucoseData(data_source=df))


# =================================================================== sine
class TestSine24hMetrics:
    """Exact-value assertions on ``sine_24h.csv``.

    Signal: ``g(t) = 120 + 30 * sin(2*pi*t / 720)`` over 288 five-minute
    samples covering exactly **2 full periods** of the 12-h sinusoid. See
    ``tests/fixtures/synthetic/README.md`` for the derivation of every
    value below.
    """

    @pytest.fixture
    def sine_metrics(self) -> GlucoseAnalysis:
        """The GlucoseAnalysis object built from the sine-24h CSV."""
        return GlucoseAnalysis(data=str(SINE_PATH))

    def test_n_records(self, sine_metrics: GlucoseAnalysis) -> None:
        """The sine CSV contains exactly 288 rows (24 h x 12 readings/h)."""
        assert len(sine_metrics.data) == 288

    def test_mean_is_120(self, sine_metrics: GlucoseAnalysis) -> None:
        """Mean glucose = 120.0 mg/dL (exact, by symmetry of ``sin`` over 2 full periods)."""
        assert sine_metrics.mean() == pytest.approx(120.0, abs=0.01)

    def test_sd_sample_is_21_2501280996(self, sine_metrics: GlucoseAnalysis) -> None:
        """Sample SD (ddof=1) = 21.2501280996 mg/dL (pandas default).

        The "theoretical" SD of ``30 * sin(2*pi*t/720)`` is ``30/sqrt(2) =
        21.2132``, but the library returns the sample SD with the Bessel
        correction ``sqrt(450 * 288/287)``.
        """
        assert sine_metrics.sd() == pytest.approx(21.250128099634388, abs=0.01)

    def test_cv_is_17_708_percent(self, sine_metrics: GlucoseAnalysis) -> None:
        """CV = sd / mean * 100 ≈ 17.708 %."""
        assert sine_metrics.cv() == pytest.approx(17.7084401, abs=0.01)

    def test_min_is_90(self, sine_metrics: GlucoseAnalysis) -> None:
        """Minimum = 120 - 30 = 90 mg/dL."""
        assert sine_metrics.data["glucose"].min() == pytest.approx(90.0, abs=0.01)

    def test_max_is_150(self, sine_metrics: GlucoseAnalysis) -> None:
        """Maximum = 120 + 30 = 150 mg/dL."""
        assert sine_metrics.data["glucose"].max() == pytest.approx(150.0, abs=0.01)

    def test_tir_is_100_percent(self, sine_metrics: GlucoseAnalysis) -> None:
        """TIR (70-180) = 100 % because every value lies in [90, 150]."""
        assert sine_metrics.TIR() == pytest.approx(100.0, abs=0.01)

    def test_tar_total_is_0(self, sine_metrics: GlucoseAnalysis) -> None:
        """TAR (>180) = 0 % because max is 150 mg/dL."""
        assert sine_metrics.TAR_total() == pytest.approx(0.0, abs=0.01)

    def test_tbr_total_is_0(self, sine_metrics: GlucoseAnalysis) -> None:
        """TBR (<70) = 0 % because min is 90 mg/dL."""
        assert sine_metrics.TBR_total() == pytest.approx(0.0, abs=0.01)

    def test_gmi_is_6_18(self, sine_metrics: GlucoseAnalysis) -> None:
        """GMI = ``round(3.31 + 0.02392 * mean, 2)`` = 6.18 % (Beck 2019).

        Not the older ``eA1c = (mean + 46.7) / 28.7 = 5.81 %`` — CGMPy uses
        the GMI formula on purpose.
        """
        assert sine_metrics.gmi() == pytest.approx(6.18, abs=0.01)

    def test_data_completeness_is_100_percent(self, sine_metrics: GlucoseAnalysis) -> None:
        """288 expected vs. 288 real at 5-min interval → 100 % completeness.

        Note: ``data_completeness`` returns an ``int`` (already rounded),
        not a float, so we assert exactly equality.
        """
        assert sine_metrics.data_completeness() == 100


# ============================================================ device 120
class TestDeviceConstant120Metrics:
    """Exact-value assertions on the 4 constant-120 device CSVs.

    All 4 fixtures are 288 rows of constant 120 mg/dL → every metric has
    a known, trivial expected value (mean=median=120, sd=cv=0, TIR=100,
    TAR=TBR=0, GMI=6.18, completeness=100).

    We load via the specialized loader (to exercise its column-name
    contract) and re-wrap the cleaned DataFrame in ``GlucoseMetrics`` to
    access the metrics mixin.
    """

    @pytest.mark.parametrize(
        ("device_class", "fixture_path", "loader_kwargs"),
        [
            (Dexcom, DEXCOM_PATH, {}),
            # Libreview needs header=2 (2 banner rows above the real header).
            (Libreview, LIBREVIEW_PATH, {"header": 2}),
            (MedtronicCarelink, MEDTRONIC_PATH, {}),
            (TandemDiabetes, TANDEM_PATH, {}),
        ],
    )
    def test_loader_returns_288_rows(
        self,
        device_class: type,
        fixture_path: Path,
        loader_kwargs: dict,
    ) -> None:
        """Each specialized loader returns 288 cleaned rows for its constant-120 fixture."""
        loader = device_class(str(fixture_path), **loader_kwargs)
        cleaned = loader.get_raw_data()
        assert len(cleaned) == 288
        # Cleaned DataFrame uses the standardized column names.
        assert "time" in cleaned.columns
        assert "glucose" in cleaned.columns

    @pytest.mark.parametrize(
        ("device_class", "fixture_path", "loader_kwargs"),
        [
            (Dexcom, DEXCOM_PATH, {}),
            (Libreview, LIBREVIEW_PATH, {"header": 2}),
            (MedtronicCarelink, MEDTRONIC_PATH, {}),
            (TandemDiabetes, TANDEM_PATH, {}),
        ],
    )
    def test_mean_is_120(
        self,
        device_class: type,
        fixture_path: Path,
        loader_kwargs: dict,
    ) -> None:
        """Mean glucose = 120.0 mg/dL on every constant-120 fixture."""
        loader = device_class(str(fixture_path), **loader_kwargs)
        metrics = _metrics_for_dataframe(loader.get_raw_data())
        assert metrics.mean() == pytest.approx(120.0, abs=0.01)

    @pytest.mark.parametrize(
        ("device_class", "fixture_path", "loader_kwargs"),
        [
            (Dexcom, DEXCOM_PATH, {}),
            (Libreview, LIBREVIEW_PATH, {"header": 2}),
            (MedtronicCarelink, MEDTRONIC_PATH, {}),
            (TandemDiabetes, TANDEM_PATH, {}),
        ],
    )
    def test_median_is_120(
        self,
        device_class: type,
        fixture_path: Path,
        loader_kwargs: dict,
    ) -> None:
        """Median glucose = 120.0 mg/dL on every constant-120 fixture."""
        loader = device_class(str(fixture_path), **loader_kwargs)
        metrics = _metrics_for_dataframe(loader.get_raw_data())
        assert metrics.median() == pytest.approx(120.0, abs=0.01)

    @pytest.mark.parametrize(
        ("device_class", "fixture_path", "loader_kwargs"),
        [
            (Dexcom, DEXCOM_PATH, {}),
            (Libreview, LIBREVIEW_PATH, {"header": 2}),
            (MedtronicCarelink, MEDTRONIC_PATH, {}),
            (TandemDiabetes, TANDEM_PATH, {}),
        ],
    )
    def test_sd_is_zero(
        self,
        device_class: type,
        fixture_path: Path,
        loader_kwargs: dict,
    ) -> None:
        """SD = 0.0 mg/dL because the signal is constant."""
        loader = device_class(str(fixture_path), **loader_kwargs)
        metrics = _metrics_for_dataframe(loader.get_raw_data())
        assert metrics.sd() == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize(
        ("device_class", "fixture_path", "loader_kwargs"),
        [
            (Dexcom, DEXCOM_PATH, {}),
            (Libreview, LIBREVIEW_PATH, {"header": 2}),
            (MedtronicCarelink, MEDTRONIC_PATH, {}),
            (TandemDiabetes, TANDEM_PATH, {}),
        ],
    )
    def test_cv_is_zero(
        self,
        device_class: type,
        fixture_path: Path,
        loader_kwargs: dict,
    ) -> None:
        """CV = 0 % because SD = 0."""
        loader = device_class(str(fixture_path), **loader_kwargs)
        metrics = _metrics_for_dataframe(loader.get_raw_data())
        assert metrics.cv() == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize(
        ("device_class", "fixture_path", "loader_kwargs"),
        [
            (Dexcom, DEXCOM_PATH, {}),
            (Libreview, LIBREVIEW_PATH, {"header": 2}),
            (MedtronicCarelink, MEDTRONIC_PATH, {}),
            (TandemDiabetes, TANDEM_PATH, {}),
        ],
    )
    def test_tir_is_100_percent(
        self,
        device_class: type,
        fixture_path: Path,
        loader_kwargs: dict,
    ) -> None:
        """TIR (70-180) = 100 % because 120 ∈ [70, 180]."""
        loader = device_class(str(fixture_path), **loader_kwargs)
        metrics = _metrics_for_dataframe(loader.get_raw_data())
        assert metrics.TIR() == pytest.approx(100.0, abs=0.01)

    @pytest.mark.parametrize(
        ("device_class", "fixture_path", "loader_kwargs"),
        [
            (Dexcom, DEXCOM_PATH, {}),
            (Libreview, LIBREVIEW_PATH, {"header": 2}),
            (MedtronicCarelink, MEDTRONIC_PATH, {}),
            (TandemDiabetes, TANDEM_PATH, {}),
        ],
    )
    def test_tar_total_is_zero(
        self,
        device_class: type,
        fixture_path: Path,
        loader_kwargs: dict,
    ) -> None:
        """TAR_total = 0 % because 120 ≤ 180."""
        loader = device_class(str(fixture_path), **loader_kwargs)
        metrics = _metrics_for_dataframe(loader.get_raw_data())
        assert metrics.TAR_total() == pytest.approx(0.0, abs=0.01)

    @pytest.mark.parametrize(
        ("device_class", "fixture_path", "loader_kwargs"),
        [
            (Dexcom, DEXCOM_PATH, {}),
            (Libreview, LIBREVIEW_PATH, {"header": 2}),
            (MedtronicCarelink, MEDTRONIC_PATH, {}),
            (TandemDiabetes, TANDEM_PATH, {}),
        ],
    )
    def test_tbr_total_is_zero(
        self,
        device_class: type,
        fixture_path: Path,
        loader_kwargs: dict,
    ) -> None:
        """TBR_total = 0 % because 120 ≥ 70."""
        loader = device_class(str(fixture_path), **loader_kwargs)
        metrics = _metrics_for_dataframe(loader.get_raw_data())
        assert metrics.TBR_total() == pytest.approx(0.0, abs=0.01)

    @pytest.mark.parametrize(
        ("device_class", "fixture_path", "loader_kwargs"),
        [
            (Dexcom, DEXCOM_PATH, {}),
            (Libreview, LIBREVIEW_PATH, {"header": 2}),
            (MedtronicCarelink, MEDTRONIC_PATH, {}),
            (TandemDiabetes, TANDEM_PATH, {}),
        ],
    )
    def test_gmi_is_6_18(
        self,
        device_class: type,
        fixture_path: Path,
        loader_kwargs: dict,
    ) -> None:
        """GMI = ``round(3.31 + 0.02392 * 120, 2)`` = 6.18 % on every fixture.

        Not the older eA1c (5.81 %); CGMPy uses Beck 2019 GMI on purpose.
        """
        loader = device_class(str(fixture_path), **loader_kwargs)
        metrics = _metrics_for_dataframe(loader.get_raw_data())
        assert metrics.gmi() == pytest.approx(6.18, abs=0.01)

    @pytest.mark.parametrize(
        ("device_class", "fixture_path", "loader_kwargs"),
        [
            (Dexcom, DEXCOM_PATH, {}),
            (Libreview, LIBREVIEW_PATH, {"header": 2}),
            (MedtronicCarelink, MEDTRONIC_PATH, {}),
            (TandemDiabetes, TANDEM_PATH, {}),
        ],
    )
    def test_data_completeness_is_100(
        self,
        device_class: type,
        fixture_path: Path,
        loader_kwargs: dict,
    ) -> None:
        """288 expected / 288 real at 5-min interval → 100 % completeness.

        ``data_completeness`` returns an ``int``, so we assert exact equality.
        """
        loader = device_class(str(fixture_path), **loader_kwargs)
        metrics = _metrics_for_dataframe(loader.get_raw_data())
        assert metrics.data_completeness() == 100


# ========================================================== device detection
class TestDeviceDetectionOnFixtures:
    """``detect_device_type`` returns the expected string for every fixture
    that has a single-row header.

    The Libreview fixture has 2 banner rows above the real header, so
    ``detect_device_type`` (which uses the default ``header=0``) treats
    the banner as the header and returns ``None`` for that file. The
    *Libreview loader* handles the 2-row banner correctly via its
    ``header=2`` constructor argument; only the auto-detection is
    limited to single-row headers.
    """

    @pytest.mark.parametrize(
        ("fixture_path", "expected_device"),
        [
            (DEXCOM_PATH, "dexcom"),
            (MEDTRONIC_PATH, "medtronic"),
            (TANDEM_PATH, "tandem"),
        ],
    )
    def test_detect_device_type(self, fixture_path: Path, expected_device: str) -> None:
        """Each device fixture with a single-row header is auto-detected."""
        assert detect_device_type(str(fixture_path)) == expected_device

    def test_detect_device_type_libreview_returns_none(self) -> None:
        """The Libreview fixture is *not* auto-detected because it has
        a 2-row banner above the real header. The loader still works
        when called with ``header=2``; this is a documented limitation
        of the auto-detection helper.
        """
        # The first row "Patient export" is taken as the header, so the
        # characteristic column names are not found.
        assert detect_device_type(str(LIBREVIEW_PATH)) is None
