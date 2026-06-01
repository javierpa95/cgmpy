"""Integration tests for the data pipeline.

These tests cover the full path: load → process → analyze → metrics.
They use the bundled synthetic CSVs in `tests/fixtures/data/`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cgmpy import ModularGlucoseData, ModularGlucoseMetrics

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "data"


class TestDataPipeline:
    """End-to-end tests on the synthetic fixtures."""

    @pytest.mark.parametrize("fixture_name", ["dm.csv", "nodm.csv", "pregnancy.csv"])
    def test_load_process_metrics(self, fixture_name: str) -> None:
        """A full pipeline (load → metrics) works for every fixture."""
        path = FIXTURES / fixture_name
        data = ModularGlucoseData(str(path))
        metrics = ModularGlucoseMetrics(data)

        # Basic metrics are computed and finite.
        result = metrics.basic()
        mean = result.mean()
        assert mean is not None
        assert mean > 0  # mg/dL always positive
        assert mean < 600  # physiological sanity

    def test_gap_detection(self, glucose_df_with_gaps) -> None:
        """A DataFrame with a 2-hour gap is loaded and the gap is detectable."""
        data = ModularGlucoseData(glucose_df_with_gaps)
        info = data.info(include_disconnections=True)
        # The fixture has a 2-hour (120-min) gap; there should be at least one
        # detected disconnection event.
        n_gaps = info.get("n_gaps", info.get("num_gaps", info.get("disconnections", 0)))
        assert n_gaps >= 1

    def test_data_quality_metrics(self) -> None:
        """`get_data_quality_metrics` returns expected keys."""
        path = FIXTURES / "dm.csv"
        data = ModularGlucoseData(str(path))
        quality = data.get_data_quality_metrics()
        assert "total_gaps" in quality or "max_gap_hours" in quality
