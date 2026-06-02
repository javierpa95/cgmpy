"""End-to-end integration tests: data loading → agata analysis.

This module exercises the full pipeline:

1. Load a device CSV via its specialized loader (``Dexcom``,
   ``Libreview``, ``MedtronicCarelink``, ``TandemDiabetes``).
2. Hand the resulting ``ModularGlucoseData`` to
   :func:`cgmpy.agata.adapter.prepare_data_for_agata` for grid alignment.
3. Call :func:`cgmpy.agata.metrics.analyze_with_agata` to obtain the
   full py_agata analysis.
4. Assert on the expected top-level sections
   (``variability``, ``time_in_ranges``, ``risk``,
   ``glycemic_transformation``, ``data_quality``).

It also covers the negative paths:

* ``AgataNotInstalledError`` when ``py_agata`` is missing or removed at
  runtime via monkey-patch.
* ``EmptyDataError`` for an empty input.
* A 1-row input yielding a 1-row output without crashing.

The whole module is skipped when ``py_agata`` is not installed — this is
intentional, the existing agata test suite follows the same convention.
"""

from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

# Skip the whole module if py_agata is not installed — matches the
# existing agata test convention.
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("py_agata") is None,
    reason="py_agata optional dependency is not installed",
)

import cgmpy.agata.metrics as agata_metrics  # noqa: E402
from cgmpy import (  # noqa: E402
    Dexcom,
    GlucoseMetrics,
    Libreview,
    MedtronicCarelink,
    ModularGlucoseData,
    TandemDiabetes,
)
from cgmpy.agata.adapter import prepare_data_for_agata  # noqa: E402
from cgmpy.agata.metrics import analyze_with_agata  # noqa: E402
from cgmpy.errors import AgataNotInstalledError, EmptyDataError  # noqa: E402

# Top-level keys the py_agata analysis must expose. Mirrors
# ``tests/unit/test_agata/test_metrics.py``.
EXPECTED_TOP_LEVEL_KEYS: set[str] = {
    "variability",
    "time_in_ranges",
    "risk",
    "glycemic_transformation",
    "data_quality",
}

# ------------------------------------------------------------------ paths
FIXTURES_DEVICES = Path(__file__).resolve().parents[1] / "fixtures" / "devices"
FIXTURES_SYNTHETIC = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic"

DEXCOM_PATH = FIXTURES_DEVICES / "dexcom_constant_120.csv"
LIBREVIEW_PATH = FIXTURES_DEVICES / "libreview_constant_120.csv"
MEDTRONIC_PATH = FIXTURES_DEVICES / "medtronic_constant_120.csv"
TANDEM_PATH = FIXTURES_DEVICES / "tandem_constant_120.csv"
SINE_PATH = FIXTURES_SYNTHETIC / "sine_24h.csv"


# =================================================================== pipeline
class TestDeviceToAgataPipeline:
    """Full pipeline tests on the 4 device CSVs (constant 120 mg/dL)."""

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
    def test_prepare_data_for_agata_returns_aligned_grid(
        self,
        device_class: type,
        fixture_path: Path,
        loader_kwargs: dict,
    ) -> None:
        """``prepare_data_for_agata`` produces a 5-min homogeneous grid."""
        loader = device_class(str(fixture_path), **loader_kwargs)
        aligned = prepare_data_for_agata(loader)
        assert isinstance(aligned, pd.DataFrame)
        assert list(aligned.columns) == ["t", "glucose"]
        # 288 rows, 5-min interval.
        deltas = aligned["t"].diff().dropna().dt.total_seconds() / 60
        assert (deltas == 5.0).all()
        assert len(aligned) == 288

    @pytest.mark.parametrize(
        ("device_class", "fixture_path", "loader_kwargs"),
        [
            (Dexcom, DEXCOM_PATH, {}),
            (Libreview, LIBREVIEW_PATH, {"header": 2}),
            (MedtronicCarelink, MEDTRONIC_PATH, {}),
            (TandemDiabetes, TANDEM_PATH, {}),
        ],
    )
    def test_analyze_with_agata_returns_top_level_sections(
        self,
        device_class: type,
        fixture_path: Path,
        loader_kwargs: dict,
    ) -> None:
        """``analyze_with_agata`` returns a non-empty dict with the
        expected top-level sections.

        The full nested result is checked (no ``summary=True``), so the
        top-level keys are preserved.
        """
        loader = device_class(str(fixture_path), **loader_kwargs)
        results = analyze_with_agata(loader)
        assert isinstance(results, dict)
        assert results, "analyze_with_agata returned an empty dict"
        assert EXPECTED_TOP_LEVEL_KEYS.issubset(results.keys())

    @pytest.mark.parametrize(
        ("device_class", "fixture_path", "loader_kwargs"),
        [
            (Dexcom, DEXCOM_PATH, {}),
            (Libreview, LIBREVIEW_PATH, {"header": 2}),
            (MedtronicCarelink, MEDTRONIC_PATH, {}),
            (TandemDiabetes, TANDEM_PATH, {}),
        ],
    )
    def test_analyze_with_agata_summary_returns_flat_dict(
        self,
        device_class: type,
        fixture_path: Path,
        loader_kwargs: dict,
    ) -> None:
        """With ``summary=True`` the result is a non-empty flat dict."""
        loader = device_class(str(fixture_path), **loader_kwargs)
        summary = analyze_with_agata(loader, summary=True)
        assert isinstance(summary, dict)
        assert summary, "analyze_with_agata(summary=True) returned an empty dict"
        # Flat dict → no value is itself a dict.
        assert all(not isinstance(v, dict) for v in summary.values())


# =================================================================== sine
class TestSineToAgataPipeline:
    """Pipeline test on the ``sine_24h.csv`` synthetic fixture.

    The data is in [90, 150] mg/dL, so on the default diabetes target
    (TIR 70-180) the time-in-target should be 100 % — well above the 95 %
    sanity bound we assert on.
    """

    def test_sine_full_round_trip(self) -> None:
        """Load sine_24h.csv via GlucoseMetrics, run py_agata, and check TIR."""
        # GlucoseMetrics already inherits from ModularGlucoseData so it
        # satisfies the type contract of analyze_with_agata.
        data = GlucoseMetrics(data_source=str(SINE_PATH))
        results = analyze_with_agata(data)
        assert EXPECTED_TOP_LEVEL_KEYS.issubset(results.keys())

        # Inspect time_in_ranges → time_in_target → percentage.
        tir_section = results["time_in_ranges"]
        assert "time_in_target" in tir_section
        assert "percentage" in tir_section["time_in_target"]
        pct = tir_section["time_in_target"]["percentage"]
        # Sine is in [90, 150] ⊂ [70, 180] → TIR is 100 %; assert
        # strictly > 95 % to allow for floating-point / unit-conversion
        # edge cases in py_agata.
        assert pct > 95.0

    def test_sine_summary_is_flat(self) -> None:
        """``summary=True`` flattens the sine result into a flat dict."""
        data = GlucoseMetrics(data_source=str(SINE_PATH))
        summary = analyze_with_agata(data, summary=True)
        assert isinstance(summary, dict)
        assert summary
        assert all(not isinstance(v, dict) for v in summary.values())


# ============================================================ error paths
class TestAgataErrorPaths:
    """Negative-path tests: missing py_agata, empty data, single row."""

    def test_agata_not_installed_error_is_raised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``analyze_with_agata`` raises ``AgataNotInstalledError`` when
        ``cgmpy.agata.metrics.Agata`` is forced to ``None``.

        We use monkey-patch (rather than uninstalling py_agata) so the
        test works both with and without the optional dependency
        installed.
        """
        monkeypatch.setattr(agata_metrics, "Agata", None)
        data = GlucoseMetrics(data_source=str(DEXCOM_PATH))
        with pytest.raises(AgataNotInstalledError):
            analyze_with_agata(data)

    def test_agata_not_installed_error_when_module_level_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The error is also raised when the module-level ``Agata``
        attribute is already ``None`` (e.g. py_agata was never installed).
        """
        # Agata may already be None if py_agata is not installed; this
        # test makes the contract explicit regardless.
        monkeypatch.setattr(agata_metrics, "Agata", None)
        data = GlucoseMetrics(data_source=str(DEXCOM_PATH))
        with pytest.raises(AgataNotInstalledError):
            analyze_with_agata(data)

    def test_empty_modular_glucose_data_raises_empty_data_error(self) -> None:
        """An empty ``ModularGlucoseData`` raises ``EmptyDataError`` from
        the adapter — the guard added in v0.5.2.
        """
        empty_df = pd.DataFrame(
            {
                "time": pd.Series(dtype="datetime64[ns]"),
                "glucose": pd.Series(dtype="float64"),
            }
        )
        gd = ModularGlucoseData(data_source=empty_df)
        assert len(gd.data) == 0  # sanity check
        with pytest.raises(EmptyDataError):
            prepare_data_for_agata(gd)

    def test_single_row_adapter_returns_one_row(self) -> None:
        """A 1-row input passes the empty-data guards and yields a
        1-row output with non-NaN glucose.
        """
        one_row_df = pd.DataFrame({"time": [datetime(2024, 1, 1, 12, 0)], "glucose": [120.0]})
        gd = ModularGlucoseData(data_source=one_row_df)
        aligned = prepare_data_for_agata(gd)
        assert isinstance(aligned, pd.DataFrame)
        assert len(aligned) == 1
        assert list(aligned.columns) == ["t", "glucose"]
        assert not aligned["glucose"].isna().any()


# ============================================================ end-to-end glue
class TestRoundTripPreservesConstantSignal:
    """Round-trip on the constant-120 fixtures: the analyzed variability
    must match the known ground truth, even after py_agata has consumed
    the data through its resampling pipeline.
    """

    @pytest.mark.parametrize(
        ("device_class", "fixture_path", "loader_kwargs"),
        [
            (Dexcom, DEXCOM_PATH, {}),
            (Libreview, LIBREVIEW_PATH, {"header": 2}),
            (MedtronicCarelink, MEDTRONIC_PATH, {}),
            (TandemDiabetes, TANDEM_PATH, {}),
        ],
    )
    def test_variability_mean_glucose_is_120(
        self,
        device_class: type,
        fixture_path: Path,
        loader_kwargs: dict,
    ) -> None:
        """``variability.mean_glucose`` is ~120 mg/dL on every constant-120 fixture."""
        loader = device_class(str(fixture_path), **loader_kwargs)
        results = analyze_with_agata(loader)
        var = results["variability"]
        assert "mean_glucose" in var
        # Tolerate small numerical drift in the resampling pipeline.
        assert var["mean_glucose"] == pytest.approx(120.0, abs=0.01)
