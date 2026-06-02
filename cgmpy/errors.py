"""
Exception hierarchy for CGMPy.

All CGMPy-raised exceptions inherit from :class:`CGMPyError`. This lets users
distinguish errors that originate in CGMPy from those raised by their own code
or by third-party libraries such as pandas or numpy.

Quick guide:

    try:
        from cgmpy import GlucoseData
        data = GlucoseData("sensor.csv", date_col="...", glucose_col="...")
    except cgmpy.errors.ColumnNotFoundError as e:
        print(f"Column missing: {e.column}; available: {e.available}")
    except cgmpy.errors.InvalidCSVFormatError as e:
        print(f"Bad CSV {e.file_path}: {e.reason}")
    except cgmpy.errors.DeviceDetectionError as e:
        print(f"Unknown device, columns found: {e.columns_found}")
    except cgmpy.errors.CGMPyError as e:
        print(f"CGMPy error: {e}")

Hierarchy:

    CGMPyError
    +-- DataError(ValueError)               # catches as ValueError for back-compat
    |   +-- ColumnNotFoundError
    |   +-- InvalidCSVFormatError
    |   +-- DeviceDetectionError
    |   +-- GlucoseRangeError
    |   +-- EmptyDataError
    +-- MetricError
    |   +-- InsufficientDataError
    +-- AgataIntegrationError
    |   +-- AgataNotInstalledError
    +-- ConfigurationError

Design notes:

* :class:`DataError` inherits from both :class:`CGMPyError` and
  :class:`ValueError` so that pre-v0.5.2 code that does ``except ValueError``
  still works. :class:`MetricError`, :class:`AgataIntegrationError` and
  :class:`ConfigurationError` inherit only from :class:`CGMPyError` to keep
  their semantics clean.
* Subclasses attach relevant context as instance attributes (e.g.
  ``ColumnNotFoundError.column``) so that callers can react programmatically
  without parsing the message string.
"""

from __future__ import annotations


class CGMPyError(Exception):
    """Base class for every exception raised by CGMPy.

    Catching this will handle any CGMPy-specific error, regardless of
    sub-domain (data, metrics, AGATA, configuration).
    """


class DataError(CGMPyError, ValueError):
    """Base class for errors in the data loading / processing / validation path.

    Inherits from :class:`ValueError` for backward compatibility with code
    written before v0.5.2 that catches generic ``ValueError``.
    """


class ColumnNotFoundError(DataError):
    """A required column is missing from the input data.

    Attributes:
        column: Name of the column that was expected.
        available: Column names that were actually present in the data.
    """

    def __init__(
        self,
        column: str,
        available: list[str] | None = None,
        message: str | None = None,
    ) -> None:
        self.column = column
        self.available = list(available) if available else []
        if message is None:
            msg = f"Required column '{column}' not found in the data."
            if self.available:
                msg += f" Available columns: {self.available}."
            message = msg
        super().__init__(message)


class InvalidCSVFormatError(DataError):
    """A CSV file could not be parsed or did not match the expected schema.

    Attributes:
        file_path: Path to the file that failed to parse.
        reason: Short description of the parse error.
    """

    def __init__(self, file_path: str, reason: str, hint: str | None = None) -> None:
        self.file_path = file_path
        self.reason = reason
        msg = f"Cannot parse CSV file '{file_path}': {reason}"
        if hint:
            msg += f" Hint: {hint}"
        super().__init__(msg)


class DeviceDetectionError(DataError):
    """The CGM device type of a file could not be automatically detected.

    Attributes:
        file_path: Path to the file that could not be classified.
        columns_found: Column names that were present in the file's header.
    """

    def __init__(self, file_path: str, columns_found: list[str] | None = None) -> None:
        self.file_path = file_path
        self.columns_found = list(columns_found) if columns_found else []
        msg = (
            f"Could not detect the device type of '{file_path}'. "
            f"Found columns: {self.columns_found}. "
            f"Supported auto-detected devices: Dexcom Clarity, FreeStyle "
            f"Libreview, Medtronic CareLink, Tandem Diabetes. "
            f"Use ModularGlucoseData(file, date_col=..., glucose_col=...) "
            f"to load a custom format manually."
        )
        super().__init__(msg)


class GlucoseRangeError(DataError):
    """Glucose values are outside the physiologically plausible range.

    The default plausible range is 39-600 mg/dL for living subjects, matching
    the bounds used by :func:`cgmpy.metrics.validation.validate_glucose_range`.
    Use ``strict=False`` in the loader to keep the old warn-only behaviour.

    Attributes:
        n_invalid: Number of out-of-range values found.
        total: Total number of glucose values in the dataset.
        min_value: Minimum glucose value observed in the data.
        max_value: Maximum glucose value observed in the data.
        bounds: ``(low, high)`` plausible range in mg/dL.
    """

    def __init__(
        self,
        n_invalid: int,
        total: int,
        min_value: float,
        max_value: float,
        bounds: tuple[float, float] = (39.0, 600.0),
    ) -> None:
        self.n_invalid = n_invalid
        self.total = total
        self.min_value = min_value
        self.max_value = max_value
        self.bounds = bounds
        super().__init__(
            f"{n_invalid} of {total} glucose values are outside the "
            f"physiologically plausible range {bounds[0]:.0f}-"
            f"{bounds[1]:.0f} mg/dL. "
            f"Observed range: {min_value:.1f} to {max_value:.1f} mg/dL."
        )


class EmptyDataError(DataError):
    """No rows remained after processing.

    Typical causes: the input file is empty, the date-range filter excluded
    every row, or all rows were dropped by validation.

    Attributes:
        context: Short description of the stage at which emptiness was
            detected (e.g. ``"input file"``, ``"date range filter"``).
    """

    def __init__(self, context: str = "data") -> None:
        self.context = context
        super().__init__(
            f"No data available in {context}. Check that your input file "
            f"is not empty and that any date-range filter includes at least "
            f"one row."
        )


class MetricError(CGMPyError):
    """Base class for errors during metric calculation."""


class InsufficientDataError(MetricError):
    """Not enough data points to compute a metric.

    Attributes:
        metric: Name of the metric that could not be computed.
        required: Minimum number of points required.
        actual: Number of points actually available.
    """

    def __init__(self, metric: str, required: int, actual: int) -> None:
        self.metric = metric
        self.required = required
        self.actual = actual
        super().__init__(
            f"Cannot compute '{metric}': requires at least {required} data points, got {actual}."
        )


class AgataIntegrationError(CGMPyError):
    """An error occurred in the py_agata integration bridge."""


class AgataNotInstalledError(AgataIntegrationError):
    """py_agata is not installed but is required for the requested operation.

    Install with ``pip install 'cgmpy[agata]'``.
    """

    def __init__(self) -> None:
        super().__init__(
            "The 'py_agata' library is required for this functionality. "
            "Install it with: pip install 'cgmpy[agata]'"
        )


class ConfigurationError(CGMPyError):
    """Invalid or missing configuration (e.g. CLI, environment, or runtime)."""


__all__ = [
    "AgataIntegrationError",
    "AgataNotInstalledError",
    "CGMPyError",
    "ColumnNotFoundError",
    "ConfigurationError",
    "DataError",
    "DeviceDetectionError",
    "EmptyDataError",
    "GlucoseRangeError",
    "InsufficientDataError",
    "InvalidCSVFormatError",
    "MetricError",
]
