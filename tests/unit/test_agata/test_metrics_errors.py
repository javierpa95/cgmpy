"""Unit tests for `cgmpy.agata.metrics` that do not require ``py_agata``.

These tests cover:

* The lightweight ``summarize_agata_results`` flattener.
* The ``AgataAnalysis`` constructor's ``glycemic_target`` parameter.
* The ``AgataNotInstalledError`` raise path when ``py_agata`` is missing.

The module-level ``py_agata`` import is wrapped in a try/except by the
production code, so the test file imports cleanly even when the optional
dependency is missing.
"""

from __future__ import annotations

import importlib.util

import pandas as pd
import pytest

import cgmpy.agata.metrics as agata_metrics
from cgmpy.agata.metrics import (
    AgataAnalysis,
    analyze_with_agata,
    summarize_agata_results,
)
from cgmpy.errors import AgataNotInstalledError

# Skip the WHOLE module if py_agata is not installed — but only for the
# tests that *do* need py_agata (e.g. ``AgataAnalysis.run``). The
# ``summarize_agata_results`` and constructor tests are pure Python and
# do not require the optional dependency. We handle that per-test below.
_PY_AGATA_AVAILABLE = importlib.util.find_spec("py_agata") is not None


# --------------------------------------------------------------------- fixtures
@pytest.fixture
def tiny_glucose_df() -> pd.DataFrame:
    """A minimal non-empty DataFrame for constructing ``AgataAnalysis`` /
    calling ``analyze_with_agata``. 12 rows at 5-min intervals.
    """
    return pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=12, freq="5min"),
            "glucose": [120.0] * 12,
        }
    )


# =========================================================== summarize_agata
class TestSummarizeAgataResults:
    """Unit tests for the pure-Python ``summarize_agata_results`` flattener.

    These tests are 100 % deterministic and do not require ``py_agata``.
    """

    def test_empty_dict_returns_empty_dict(self) -> None:
        """An empty input dict yields an empty output dict."""
        assert summarize_agata_results({}) == {}

    def test_top_level_category_is_flattened(self) -> None:
        """A single non-events category is flattened to ``category_name`` keys."""
        result = summarize_agata_results({"variability": {"var1": 1.0}})
        assert result == {"variability_var1": 1.0}

    def test_multiple_categories_are_flattened(self) -> None:
        """Multiple top-level categories each get their own prefix."""
        result = summarize_agata_results(
            {
                "variability": {"mean_glucose": 120.0, "std_glucose": 30.0},
                "risk": {"lbgi": 0.5},
            }
        )
        assert result == {
            "variability_mean_glucose": 120.0,
            "variability_std_glucose": 30.0,
            "risk_lbgi": 0.5,
        }

    def test_non_dict_top_level_value_is_kept_as_is(self) -> None:
        """A scalar value at the top level is preserved (not flattened)."""
        result = summarize_agata_results({"scalar_value": 42})
        assert result == {"scalar_value": 42}

    def test_events_have_specific_flattening(self) -> None:
        """Events get a special flat shape: ``<event_type>_<level>_<key>``.

        Per the production code, the events path extracts only
        ``mean_duration`` and ``events_per_week`` and renames the latter to
        ``per_week``.
        """
        result = summarize_agata_results(
            {
                "events": {
                    "hypo": {
                        "l1": {"mean_duration": 30, "events_per_week": 0.5},
                    },
                }
            }
        )
        assert result == {
            "hypo_l1_mean_duration": 30,
            "hypo_l1_per_week": 0.5,
        }

    def test_events_missing_optional_fields_are_skipped(self) -> None:
        """Events entries with only one of the two fields yield only that one key."""
        result = summarize_agata_results({"events": {"hyper": {"l2": {"mean_duration": 60}}}})
        assert result == {"hyper_l2_mean_duration": 60}
        assert "hyper_l2_per_week" not in result

    def test_mixed_events_and_categories(self) -> None:
        """Events and non-events categories can be flattened together."""
        result = summarize_agata_results(
            {
                "events": {
                    "hypo": {"l1": {"mean_duration": 30, "events_per_week": 0.5}},
                },
                "variability": {"mean_glucose": 120.0},
            }
        )
        assert result == {
            "hypo_l1_mean_duration": 30,
            "hypo_l1_per_week": 0.5,
            "variability_mean_glucose": 120.0,
        }


# ============================================================ AgataAnalysis ctor
class TestAgataAnalysisConstructor:
    """The ``AgataAnalysis`` constructor stores ``glycemic_target`` correctly.

    These tests do **not** call ``.run()`` or any py_agata-bound method;
    they only verify the constructor's behaviour.
    """

    def test_default_glycemic_target_is_diabetes(self, tiny_glucose_df: pd.DataFrame) -> None:
        """With no ``glycemic_target`` argument, the default is ``"diabetes"``."""
        analyzer = AgataAnalysis(data_source=tiny_glucose_df)
        assert analyzer.glycemic_target == "diabetes"

    def test_explicit_diabetes_glycemic_target(self, tiny_glucose_df: pd.DataFrame) -> None:
        """Passing ``glycemic_target='diabetes'`` stores it verbatim."""
        analyzer = AgataAnalysis(data_source=tiny_glucose_df, glycemic_target="diabetes")
        assert analyzer.glycemic_target == "diabetes"

    def test_explicit_pregnancy_glycemic_target(self, tiny_glucose_df: pd.DataFrame) -> None:
        """Passing ``glycemic_target='pregnancy'`` stores it verbatim."""
        analyzer = AgataAnalysis(data_source=tiny_glucose_df, glycemic_target="pregnancy")
        assert analyzer.glycemic_target == "pregnancy"

    def test_constructor_loads_data(self, tiny_glucose_df: pd.DataFrame) -> None:
        """The constructor still processes the input DataFrame."""
        analyzer = AgataAnalysis(data_source=tiny_glucose_df)
        assert len(analyzer.data) == len(tiny_glucose_df)


# ====================================================== AgataNotInstalledError
class TestAgataNotInstalledError:
    """``analyze_with_agata`` raises ``AgataNotInstalledError`` when py_agata
    is unavailable, regardless of whether it was never installed or was
    monkey-patched out at runtime.
    """

    def test_error_is_raised_when_agata_attribute_is_none(
        self, tiny_glucose_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If ``cgmpy.agata.metrics.Agata`` is forced to ``None``, calling
        ``analyze_with_agata`` raises ``AgataNotInstalledError`` — *before*
        any data preparation runs.
        """
        # Force the "py_agata is missing" condition, whether or not it is
        # actually installed in this environment.
        monkeypatch.setattr(agata_metrics, "Agata", None)

        # `analyze_with_agata` checks `Agata is None` *before* preparing
        # data, so even an empty DataFrame would trigger the error first.
        gd = agata_metrics.GlucoseData  # use the symbol from the module
        # Build a minimal data wrapper (using a fresh DataFrame).
        obj = gd(data_source=tiny_glucose_df)

        with pytest.raises(AgataNotInstalledError):
            analyze_with_agata(obj)

    def test_error_raised_when_py_agata_never_imported(self, tiny_glucose_df: pd.DataFrame) -> None:
        """When py_agata is not installed in this environment, the module
        sets ``Agata = None`` at import time and ``analyze_with_agata``
        raises ``AgataNotInstalledError`` for any input.
        """
        if _PY_AGATA_AVAILABLE:
            pytest.skip("py_agata is installed; this test only runs without it")
        gd = agata_metrics.GlucoseData
        obj = gd(data_source=tiny_glucose_df)
        with pytest.raises(AgataNotInstalledError):
            analyze_with_agata(obj)

    def test_error_raised_after_monkeypatch_even_if_py_agata_present(
        self, tiny_glucose_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If py_agata IS installed, monkey-patching ``Agata`` to ``None``
        should still cause ``analyze_with_agata`` to raise.
        """
        if not _PY_AGATA_AVAILABLE:
            pytest.skip("py_agata is not installed; cannot test the monkey-patch path")
        monkeypatch.setattr(agata_metrics, "Agata", None)
        gd = agata_metrics.GlucoseData
        obj = gd(data_source=tiny_glucose_df)
        with pytest.raises(AgataNotInstalledError):
            analyze_with_agata(obj)
