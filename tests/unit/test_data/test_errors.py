"""Tests for the new exception hierarchy (`cgmpy.errors`) and its raise points.

These tests exercise the v0.5.2 hardening:

* `cgmpy.data.loader.DataLoader` — ``InvalidCSVFormatError`` on malformed files.
* `cgmpy.data.processor.DataProcessor` — ``ColumnNotFoundError`` on missing
  columns and ``GlucoseRangeError`` on out-of-range glucose when the new
  ``strict_glucose_range=True`` flag is set.
* `cgmpy.data.specialized` — ``detect_device_type`` returns ``None`` (not
  the legacy string ``"unknown"``) on unrecognised formats, and
  ``create_specialized_loader`` raises ``DeviceDetectionError`` instead of
  silently falling back to the generic loader.
* `cgmpy.errors` — the public exception types are correctly subclassed
  from ``CGMPyError``, and ``DataError`` is also a ``ValueError`` for
  backward compatibility with pre-v0.5.2 code.

The test module does not require ``py_agata``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from cgmpy import (
    Dexcom,
    GlucoseData,
    Libreview,
    MedtronicCarelink,
    TandemDiabetes,
)
from cgmpy.data.loader import DataLoader
from cgmpy.data.processor import DataProcessor
from cgmpy.data.specialized import create_specialized_loader, detect_device_type
from cgmpy.errors import (
    AgataIntegrationError,
    AgataNotInstalledError,
    CGMPyError,
    ColumnNotFoundError,
    ConfigurationError,
    DataError,
    DeviceDetectionError,
    EmptyDataError,
    GlucoseRangeError,
    InsufficientDataError,
    InvalidCSVFormatError,
    MetricError,
)

# ------------------------------------------------------------------ paths
FIXTURES_DEVICES = Path(__file__).resolve().parents[2] / "fixtures" / "devices"
FIXTURES_EDGE = FIXTURES_DEVICES / "edge_cases"


# ============================================================== hierarchy
class TestExceptionHierarchy:
    """Assert the v0.5.2 exception hierarchy is well-formed."""

    def test_cgmpy_error_inherits_from_exception(self) -> None:
        """``CGMPyError`` is a subclass of the standard ``Exception``."""
        assert issubclass(CGMPyError, Exception)

    def test_data_error_inherits_from_value_error(self) -> None:
        """``DataError`` is a ``ValueError`` for back-compat with pre-v0.5.2 code.

        This is the key back-compat property: code that catches
        ``ValueError`` continues to work after the refactor.
        """
        assert issubclass(DataError, ValueError)
        assert issubclass(DataError, CGMPyError)

    def test_metric_error_does_not_inherit_value_error(self) -> None:
        """``MetricError`` is a *CGMPy-only* error, not a ``ValueError``.

        Unlike ``DataError``, metrics errors do not need to be catchable as
        ``ValueError`` — they are domain-specific failures.
        """
        assert issubclass(MetricError, CGMPyError)
        assert not issubclass(MetricError, ValueError)

    def test_agata_integration_error_does_not_inherit_value_error(self) -> None:
        """``AgataIntegrationError`` is a *CGMPy-only* error, not a ``ValueError``."""
        assert issubclass(AgataIntegrationError, CGMPyError)
        assert not issubclass(AgataIntegrationError, ValueError)

    @pytest.mark.parametrize(
        "exc_cls",
        [
            ColumnNotFoundError,
            InvalidCSVFormatError,
            DeviceDetectionError,
            GlucoseRangeError,
            EmptyDataError,
            InsufficientDataError,
            AgataNotInstalledError,
            ConfigurationError,
        ],
    )
    def test_all_listed_exceptions_subclass_cgmpy_error(self, exc_cls: type[Exception]) -> None:
        """Every documented exception in ``cgmpy.errors`` is a ``CGMPyError``."""
        assert issubclass(exc_cls, CGMPyError)


class TestExceptionContextAttributes:
    """Assert that exceptions expose the context attributes the caller relies on."""

    def test_column_not_found_error_attaches_column_and_available(self) -> None:
        """``ColumnNotFoundError`` carries ``.column`` and ``.available``."""
        err = ColumnNotFoundError("glucose", available=["time", "other"])
        assert err.column == "glucose"
        assert err.available == ["time", "other"]
        # Message is human-readable and mentions the column.
        assert "glucose" in str(err)

    def test_invalid_csv_format_error_attaches_file_path(self) -> None:
        """``InvalidCSVFormatError`` carries ``.file_path`` and ``.reason``."""
        err = InvalidCSVFormatError("/some/file.csv", reason="unexpected token")
        assert err.file_path == "/some/file.csv"
        assert "unexpected token" in str(err)

    def test_device_detection_error_attaches_columns_found(self) -> None:
        """``DeviceDetectionError`` carries ``.file_path`` and ``.columns_found``."""
        err = DeviceDetectionError("/some/file.csv", columns_found=["foo", "bar"])
        assert err.file_path == "/some/file.csv"
        assert err.columns_found == ["foo", "bar"]

    def test_glucose_range_error_attaches_summary(self) -> None:
        """``GlucoseRangeError`` carries the diagnostic attributes for the
        caller (number invalid, total, min, max, plausible bounds)."""
        err = GlucoseRangeError(
            n_invalid=3,
            total=100,
            min_value=20.0,
            max_value=700.0,
            bounds=(39.0, 600.0),
        )
        assert err.n_invalid == 3
        assert err.total == 100
        assert err.min_value == 20.0
        assert err.max_value == 700.0
        assert err.bounds == (39.0, 600.0)

    def test_insufficient_data_error_attaches_metric_required_actual(self) -> None:
        """``InsufficientDataError`` carries ``.metric``, ``.required`` and ``.actual``."""
        err = InsufficientDataError(metric="sd", required=10, actual=3)
        assert err.metric == "sd"
        assert err.required == 10
        assert err.actual == 3


# ================================================================== loader
class TestDataLoaderExceptions:
    """Tests for the new exception raises in ``cgmpy.data.loader.DataLoader``."""

    def test_load_csv_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        """A non-existent CSV path raises the built-in ``FileNotFoundError``.

        This contract is preserved (not replaced by a CGMPy error) because
        ``FileNotFoundError`` is a Python built-in and downstream code may
        already be catching it.
        """
        loader = DataLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_from_source(
                str(tmp_path / "missing.csv"), date_col="time", glucose_col="glucose"
            )

    def test_load_csv_malformed_raises_invalid_csv(self, tmp_path: Path) -> None:
        """A file of random bytes is not a valid CSV → ``InvalidCSVFormatError``.

        The loader tries both ``,`` and ``;`` delimiters internally; both
        fail on binary garbage, so the final exception is
        ``InvalidCSVFormatError``.
        """
        garbage = tmp_path / "garbage.csv"
        garbage.write_bytes(b"\x00\x01\x02\x03\x04")
        loader = DataLoader()
        with pytest.raises(InvalidCSVFormatError) as exc_info:
            loader.load_from_source(str(garbage), date_col="time", glucose_col="glucose")
        # The exception carries the offending file path.
        assert exc_info.value.file_path == str(garbage)

    def test_load_parquet_with_garbage_content_raises_invalid_csv(self, tmp_path: Path) -> None:
        """A non-Parquet file with a ``.parquet`` extension is rejected."""
        bad = tmp_path / "not_really.parquet"
        bad.write_bytes(b"this is not a parquet file at all")
        loader = DataLoader()
        with pytest.raises(InvalidCSVFormatError) as exc_info:
            loader.load_from_source(str(bad), date_col="time", glucose_col="glucose")
        assert exc_info.value.file_path == str(bad)

    def test_load_csv_missing_date_col_raises_invalid_csv(self, tmp_path: Path) -> None:
        """A valid CSV loaded with a non-existent ``date_col`` raises
        ``InvalidCSVFormatError`` at the *loader* layer.

        Note: the *processor* (``DataProcessor.process_data``) is the layer
        that raises ``ColumnNotFoundError`` once the data is in memory.
        The loader is too early in the pipeline to know which columns are
        "expected" vs. "available", so it surfaces a generic
        ``InvalidCSVFormatError`` whose reason explains the mismatch.
        """
        csv = tmp_path / "ok.csv"
        csv.write_text("time,glucose\n2024-01-01T00:00:00,120\n", encoding="utf-8")
        loader = DataLoader()
        with pytest.raises(InvalidCSVFormatError):
            loader.load_from_source(str(csv), date_col="not_a_column", glucose_col="glucose")


# ================================================================ processor
class TestDataProcessorExceptions:
    """Tests for the new exception raises in ``cgmpy.data.processor.DataProcessor``."""

    def _build_processor_df(self) -> pd.DataFrame:
        """A small in-memory DataFrame in the schema the processor expects."""
        return pd.DataFrame(
            {
                "time": pd.date_range("2024-01-01", periods=12, freq="5min"),
                "glucose": [120.0] * 12,
            }
        )

    def test_process_missing_date_col_raises_column_not_found(self) -> None:
        """``process_data`` with a missing ``date_col`` raises
        ``ColumnNotFoundError`` and exposes ``.column`` / ``.available``.
        """
        df = self._build_processor_df()
        processor = DataProcessor()
        with pytest.raises(ColumnNotFoundError) as exc_info:
            processor.process_data(df, date_col="missing_date", glucose_col="glucose")
        assert exc_info.value.column == "missing_date"
        assert "time" in exc_info.value.available
        assert "glucose" in exc_info.value.available

    def test_process_missing_glucose_col_raises_column_not_found(self) -> None:
        """``process_data`` with a missing ``glucose_col`` raises
        ``ColumnNotFoundError`` and exposes ``.column`` / ``.available``.
        """
        df = self._build_processor_df()
        processor = DataProcessor()
        with pytest.raises(ColumnNotFoundError) as exc_info:
            processor.process_data(df, date_col="time", glucose_col="missing_glucose")
        assert exc_info.value.column == "missing_glucose"
        assert "time" in exc_info.value.available
        assert "glucose" in exc_info.value.available

    def test_strict_glucose_range_raises_on_out_of_range_high(self) -> None:
        """``strict_glucose_range=True`` on the out-of-range-high fixture
        raises ``GlucoseRangeError`` with the expected diagnostic summary.

        The fixture (``out_of_range_high.csv``) has 12 rows: 11 at 120 mg/dL
        and one at 700 mg/dL. Only the 700 is above the 600 mg/dL ceiling,
        so ``n_invalid=1`` and ``min=120.0``, ``max=700.0``.
        """
        loader = DataLoader()
        df = loader.load_from_source(
            str(FIXTURES_EDGE / "out_of_range_high.csv"),
            date_col="Marca temporal (AAAA-MM-DDThh:mm:ss)",
            glucose_col="Nivel de glucosa (mg/dL)",
        )
        # Sanity check on the fixture shape: 12 rows, with the spike.
        assert len(df) == 12
        assert float(df["Nivel de glucosa (mg/dL)"].max()) == 700.0

        processor = DataProcessor()
        with pytest.raises(GlucoseRangeError) as exc_info:
            processor.process_data(
                df.rename(
                    columns={
                        "Marca temporal (AAAA-MM-DDThh:mm:ss)": "time",
                        "Nivel de glucosa (mg/dL)": "glucose",
                    }
                ),
                date_col="time",
                glucose_col="glucose",
                strict_glucose_range=True,
            )
        err = exc_info.value
        assert err.n_invalid == 1
        assert err.total == 12
        assert err.min_value == 120.0
        assert err.max_value == 700.0
        assert err.bounds == (39.0, 600.0)

    def test_strict_glucose_range_raises_on_out_of_range_low(self) -> None:
        """``strict_glucose_range=True`` on the out-of-range-low fixture.

        The fixture has 12 rows: 11 at 120 mg/dL and one at 20 mg/dL. Only
        the 20 is below the 39 mg/dL floor, so ``n_invalid=1`` and
        ``min=20.0``, ``max=120.0``.
        """
        loader = DataLoader()
        df = loader.load_from_source(
            str(FIXTURES_EDGE / "out_of_range_low.csv"),
            date_col="Marca temporal (AAAA-MM-DDThh:mm:ss)",
            glucose_col="Nivel de glucosa (mg/dL)",
        )
        assert len(df) == 12
        assert float(df["Nivel de glucosa (mg/dL)"].min()) == 20.0

        processor = DataProcessor()
        with pytest.raises(GlucoseRangeError) as exc_info:
            processor.process_data(
                df.rename(
                    columns={
                        "Marca temporal (AAAA-MM-DDThh:mm:ss)": "time",
                        "Nivel de glucosa (mg/dL)": "glucose",
                    }
                ),
                date_col="time",
                glucose_col="glucose",
                strict_glucose_range=True,
            )
        err = exc_info.value
        assert err.n_invalid == 1
        assert err.total == 12
        assert err.min_value == 20.0
        assert err.max_value == 120.0

    def test_default_strict_glucose_range_does_not_raise(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """``strict_glucose_range=False`` (the default) on the out-of-range
        fixture does **not** raise — it only logs a warning, preserving the
        legacy warn-only behaviour.
        """
        loader = DataLoader()
        df = loader.load_from_source(
            str(FIXTURES_EDGE / "out_of_range_high.csv"),
            date_col="Marca temporal (AAAA-MM-DDThh:mm:ss)",
            glucose_col="Nivel de glucosa (mg/dL)",
        )
        processor = DataProcessor()
        with caplog.at_level(logging.WARNING, logger="cgmpy.metrics.validation"):
            processed, _ = processor.process_data(
                df.rename(
                    columns={
                        "Marca temporal (AAAA-MM-DDThh:mm:ss)": "time",
                        "Nivel de glucosa (mg/dL)": "glucose",
                    }
                ),
                date_col="time",
                glucose_col="glucose",
                # strict_glucose_range=False is the default
            )
        # The 700 mg/dL value is preserved in the processed frame.
        assert len(processed) == 12
        assert float(processed["glucose"].max()) == 700.0
        # And a validation warning was emitted.
        assert any("Glucose validation" in r.message for r in caplog.records)


# ================================================================ specialized
class TestSpecializedExceptions:
    """Tests for the new contracts in ``cgmpy.data.specialized``."""

    def test_detect_device_type_returns_none_for_unknown_format(self, tmp_path: Path) -> None:
        """``detect_device_type`` returns ``None`` (not the legacy
        ``"unknown"`` string) when the file's header matches no known
        device format.
        """
        unknown = tmp_path / "fallback.csv"
        unknown.write_text("time,glucose\n2024-01-01 00:00:00,120\n", encoding="utf-8")
        assert detect_device_type(str(unknown)) is None

    def test_detect_device_type_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        """A missing file yields ``None`` rather than raising."""
        assert detect_device_type(str(tmp_path / "does_not_exist.csv")) is None

    def test_detect_device_type_returns_dexcom_for_all_nan_glucose(
        self,
    ) -> None:
        """Detection is based on the *header*, not the data values.

        The ``all_nan_glucose.csv`` edge fixture has the Dexcom column
        names in the header but NaN glucose values — detection must still
        return ``"dexcom"`` because the header check is independent of the
        data.
        """
        path = str(FIXTURES_EDGE / "all_nan_glucose.csv")
        assert detect_device_type(path) == "dexcom"

    def test_create_specialized_loader_raises_on_unrecognised_csv(self, tmp_path: Path) -> None:
        """``create_specialized_loader`` raises ``DeviceDetectionError``
        (instead of silently falling back to the generic loader) when the
        file's header does not match any known device.
        """
        unknown = tmp_path / "not_a_device.csv"
        unknown.write_text("time,glucose\n2024-01-01 00:00:00,120\n", encoding="utf-8")
        with pytest.raises(DeviceDetectionError) as exc_info:
            create_specialized_loader(str(unknown))
        # The exception carries the columns actually present (capped at 5).
        assert "time" in exc_info.value.columns_found
        assert "glucose" in exc_info.value.columns_found

    def test_create_specialized_loader_with_explicit_unknown_device_raises(
        self, tmp_path: Path
    ) -> None:
        """An explicit but unknown ``device_type`` raises
        ``DeviceDetectionError`` — silent fallback is intentionally
        forbidden.
        """
        known = tmp_path / "dexcom.csv"
        df = pd.DataFrame(
            {
                "Marca temporal (AAAA-MM-DDThh:mm:ss)": pd.date_range(
                    "2024-01-01", periods=4, freq="5min"
                ).strftime("%Y-%m-%dT%H:%M:%S"),
                "Nivel de glucosa (mg/dL)": [120] * 4,
            }
        )
        df.to_csv(known, index=False)
        with pytest.raises(DeviceDetectionError):
            create_specialized_loader(
                str(known),
                device_type="brand_new_made_up_device",
            )


# ============================================================== end-to-end
class TestEndToEndErrorFlow:
    """Smoke tests that the new exception types flow up through the facade."""

    def test_glucose_data_missing_column_raises_invalid_csv(self, tmp_path: Path) -> None:
        """Constructing a ``GlucoseData`` with a bad ``date_col``
        surfaces ``InvalidCSVFormatError`` from the loader.

        The loader rejects unknown ``usecols`` before the processor's
        ``_validate_columns`` step ever runs, so the user sees an
        ``InvalidCSVFormatError`` whose reason explains which columns were
        not found. The processor-level ``ColumnNotFoundError`` only fires
        when the data is supplied as a DataFrame (bypassing the loader).
        """
        csv = tmp_path / "ok.csv"
        csv.write_text("ts,bg\n2024-01-01 00:00:00,120\n", encoding="utf-8")
        with pytest.raises(InvalidCSVFormatError):
            GlucoseData(str(csv), date_col="time", glucose_col="bg")

    def test_glucose_data_dataframe_with_missing_column_raises_column_not_found(
        self,
    ) -> None:
        """When the input is an in-memory DataFrame the loader is bypassed,
        so the processor's ``_validate_columns`` check runs and raises
        ``ColumnNotFoundError``.
        """
        import pandas as pd

        df = pd.DataFrame(
            {
                "ts": pd.date_range("2024-01-01", periods=3, freq="5min"),
                "bg": [120.0, 121.0, 119.0],
            }
        )
        with pytest.raises(ColumnNotFoundError):
            GlucoseData(data_source=df, date_col="time", glucose_col="bg")

    def test_strict_glucose_range_flows_via_processor(self) -> None:
        """End-to-end: the ``strict_glucose_range`` flag is exposed on
        ``DataProcessor.process_data`` (the only place where the strict
        gate is implemented). At the ``GlucoseData`` / ``Dexcom``
        constructor level the legacy warn-only behaviour is preserved.
        """
        loader = DataLoader()
        df = loader.load_from_source(
            str(FIXTURES_EDGE / "out_of_range_high.csv"),
            date_col="Marca temporal (AAAA-MM-DDThh:mm:ss)",
            glucose_col="Nivel de glucosa (mg/dL)",
        )
        df = df.rename(
            columns={
                "Marca temporal (AAAA-MM-DDThh:mm:ss)": "time",
                "Nivel de glucosa (mg/dL)": "glucose",
            }
        )
        # Strict mode raises:
        processor = DataProcessor()
        with pytest.raises(GlucoseRangeError):
            processor.process_data(
                df, date_col="time", glucose_col="glucose", strict_glucose_range=True
            )
        # Default (warn-only) does not raise:
        processed, _ = processor.process_data(df, date_col="time", glucose_col="glucose")
        assert len(processed) == 12
        assert float(processed["glucose"].max()) == 700.0

    def test_dexcom_class_does_not_swallow_detection_error(self, tmp_path: Path) -> None:
        """Constructing a ``Dexcom`` directly with a non-Dexcom file should
        not silently succeed — the loader surfaces ``InvalidCSVFormatError``
        (a ``DataError`` and therefore a ``ValueError``) when the required
        columns are missing from the CSV.
        """
        bad = tmp_path / "not_dexcom.csv"
        bad.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
        with pytest.raises(InvalidCSVFormatError):
            Dexcom(str(bad))

    @pytest.mark.parametrize(
        "device_cls",
        [Dexcom, Libreview, MedtronicCarelink, TandemDiabetes],
    )
    def test_all_device_loaders_load_constant_120_fixture(
        self, device_cls: type[GlucoseData]
    ) -> None:
        """Smoke: every specialized loader opens its 288-row constant-120
        fixture without raising.
        """
        file_name = {
            Dexcom: "dexcom_constant_120.csv",
            Libreview: "libreview_constant_120.csv",
            MedtronicCarelink: "medtronic_constant_120.csv",
            TandemDiabetes: "tandem_constant_120.csv",
        }[device_cls]
        path = FIXTURES_DEVICES / file_name
        # Libreview needs header=2 (2 banner rows above the real header).
        kwargs: dict[str, object] = {}
        if device_cls is Libreview:
            kwargs["header"] = 2
        loader = device_cls(str(path), **kwargs)
        assert len(loader.get_raw_data()) == 288
