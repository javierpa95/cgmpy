"""Integration tests for the data pipeline.

These tests cover the full path: load -> process -> analyze -> metrics.
They use the bundled synthetic CSVs in `tests/fixtures/data/`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cgmpy import GlucoseMetrics, ModularGlucoseData

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "data"


class TestDataPipeline:
    """End-to-end tests on the synthetic fixtures."""

    @pytest.mark.parametrize("fixture_name", ["dm.csv", "nodm.csv", "pregnancy.csv"])
    def test_load_process_metrics(self, fixture_name: str) -> None:
        """A full pipeline (load -> metrics) works for every fixture."""
        path = FIXTURES / fixture_name
        data = ModularGlucoseData(str(path))

        # Compute basic metrics via the user-facing class.
        # DataLoader accepts a path or a DataFrame (not ModularGlucoseData),
        # so we hand it the underlying raw DataFrame.
        metrics = GlucoseMetrics(data_source=data.get_raw_data())
        result = metrics.calculate_all_metrics()
        mean = result["Mean"]

        assert mean is not None
        assert mean > 0  # mg/dL always positive
        assert mean < 600  # physiological sanity

    def test_gap_detection(self, glucose_df_with_gaps) -> None:
        """A DataFrame with a 2-hour gap is loaded and the gap is detectable."""
        data = ModularGlucoseData(glucose_df_with_gaps)
        info = data.info(include_disconnections=True)
        # The fixture has a 2-hour (120-min) gap. `n_disconnections` is a
        # formatted string like '1 disconnections (...)'; the most reliable
        # signal is the disconnection_list length.
        assert isinstance(info.get("disconnection_list"), list)
        assert len(info["disconnection_list"]) >= 1
        # The gap is ~2 hours.
        assert info.get("total_disconnection_time", 0) >= 1.5

    def test_data_quality_metrics(self) -> None:
        """`get_data_quality_metrics` returns expected keys."""
        path = FIXTURES / "dm.csv"
        data = ModularGlucoseData(str(path))
        quality = data.get_data_quality_metrics()
        assert "total_gaps" in quality
        assert "max_gap_hours" in quality
        assert "mean_glucose" in quality
