"""
Module with specialized classes for specific glucose devices.

Errors raised by this module are CGMPy-specific subclasses of
:class:`ValueError` (see :mod:`cgmpy.errors`).
"""

import datetime
import logging

import pandas as pd

from ..errors import DeviceDetectionError
from .core import GlucoseData

logger = logging.getLogger(__name__)


def _device_str(cls_name: str, info: dict) -> str:
    """Generate a standardised device info string.

    Args:
        cls_name: The device class name (e.g. 'Dexcom').
        info: The info dictionary from :meth:`~cgmpy.data.analyzer.DataAnalyzer.get_basic_info`.

    Returns:
        Formatted string with device info.
    """
    return (
        f"{cls_name} CGM Device\n"
        f"====================\n"
        f"  Records: {info.get('n_records', 'N/A'):,}\n"
        f"  Start: {info.get('start_date', 'N/A')}\n"
        f"  End: {info.get('end_date', 'N/A')}\n"
        f"  Completeness: {info.get('completeness', 'N/A'):.1f}%\n"
        f"  Typical interval: {info.get('typical_interval', 'N/A')} min\n"
        f"  Mean glucose: {info.get('mean_glucose', 'N/A')} mg/dL\n"
        f"  GMI: {info.get('gmi', 'N/A')}%\n"
    )


#: Tuple of device type strings recognised by :func:`detect_device_type` and
#: :func:`create_specialized_loader`. Used in error messages and for sanity
#: checks.
SUPPORTED_DEVICES: tuple[str, ...] = ("dexcom", "libreview", "medtronic", "tandem")


class Dexcom(GlucoseData):
    """
    Specialized class for Dexcom device data.

    This class inherits from GlucoseData and automatically configures
    the specific column names for files exported from Dexcom Clarity.
    """

    def __init__(
        self,
        file_path: str,
        start_date: str | datetime.datetime | None = None,
        end_date: str | datetime.datetime | None = None,
        log: bool = False,
    ):
        """
        Initializes Dexcom data.

        Args:
            file_path: Path to the exported Clarity CSV file
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)
            log: If True, enables detailed performance logs

        Usage example:
        >>> dexcom = Dexcom("dexcom_data.csv")
        >>> print(dexcom.info())
        """
        super().__init__(
            data_source=file_path,
            date_col="Marca temporal (AAAA-MM-DDThh:mm:ss)",
            glucose_col="Nivel de glucosa (mg/dL)",
            start_date=start_date,
            end_date=end_date,
            log=log,
        )

    def __str__(self) -> str:
        """Custom representation for Dexcom."""
        return _device_str("Dexcom", self.info())


class Libreview(GlucoseData):
    """
    Specialized class for Libreview device data.

    This class inherits from GlucoseData and automatically configures
    the specific column names for files exported from Libreview.
    """

    def __init__(
        self,
        file_path: str,
        header: int = 2,
        start_date: str | datetime.datetime | None = None,
        end_date: str | datetime.datetime | None = None,
        log: bool = False,
    ):
        """
        Initializes Libreview data.

        Args:
            file_path: Path to the exported Libreview CSV file
            header: Header row (usually 2 for Libreview)
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)
            log: If True, enables detailed performance logs

        Usage example:
        >>> libreview = Libreview("libreview_data.csv")
        >>> print(libreview.info())
        """
        super().__init__(
            data_source=file_path,
            date_col="Sello de tiempo del dispositivo",
            glucose_col="Historial de glucosa mg/dL",
            header=header,
            start_date=start_date,
            end_date=end_date,
            log=log,
        )

    def __str__(self) -> str:
        """Custom representation for Libreview."""
        return _device_str("Libreview", self.info())


class MedtronicCarelink(GlucoseData):
    """
    Specialized class for Medtronic CareLink device data.

    This class inherits from GlucoseData and automatically configures
    the specific column names for files exported from CareLink.
    """

    def __init__(
        self,
        file_path: str,
        start_date: str | datetime.datetime | None = None,
        end_date: str | datetime.datetime | None = None,
        log: bool = False,
    ):
        """
        Initializes Medtronic CareLink data.

        Args:
            file_path: Path to the exported CareLink CSV file
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)
            log: If True, enables detailed performance logs

        Usage example:
        >>> carelink = MedtronicCarelink("carelink_data.csv")
        >>> print(carelink.info())
        """
        super().__init__(
            data_source=file_path,
            date_col="Fecha y hora",
            glucose_col="Valor del sensor (mg/dL)",
            start_date=start_date,
            end_date=end_date,
            log=log,
        )

    def __str__(self) -> str:
        """Custom representation for Medtronic CareLink."""
        return _device_str("MedtronicCarelink", self.info())


class TandemDiabetes(GlucoseData):
    """
    Specialized class for Tandem Diabetes device data.

    This class inherits from GlucoseData and automatically configures
    the specific column names for files exported from Tandem.
    """

    def __init__(
        self,
        file_path: str,
        start_date: str | datetime.datetime | None = None,
        end_date: str | datetime.datetime | None = None,
        log: bool = False,
    ):
        """
        Initializes Tandem Diabetes data.

        Args:
            file_path: Path to the exported Tandem CSV file
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)
            log: If True, enables detailed performance logs

        Usage example:
        >>> tandem = TandemDiabetes("tandem_data.csv")
        >>> print(tandem.info())
        """
        super().__init__(
            data_source=file_path,
            date_col="Timestamp",
            glucose_col="CGM Glucose Value (mg/dL)",
            start_date=start_date,
            end_date=end_date,
            log=log,
        )

    def __str__(self) -> str:
        """Custom representation for Tandem Diabetes."""
        return _device_str("TandemDiabetes", self.info())


def detect_device_type(file_path: str) -> str | None:
    """
    Automatically detects the device type based on the file's header.

    Args:
        file_path: Path to the CSV file

    Returns:
        The detected device type (``"dexcom"``, ``"libreview"``,
        ``"medtronic"`` or ``"tandem"``) or ``None`` when the file cannot
        be read or does not match any known format.
    """
    try:
        # Read the first few rows to detect the format
        sample = pd.read_csv(file_path, nrows=5)
        columns = sample.columns.tolist()

        # Detect by characteristic column names
        if "Marca temporal (AAAA-MM-DDThh:mm:ss)" in columns:
            return "dexcom"
        elif "Sello de tiempo del dispositivo" in columns:
            return "libreview"
        elif "Fecha y hora" in columns and "Valor del sensor (mg/dL)" in columns:
            return "medtronic"
        elif "Timestamp" in columns and "CGM Glucose Value (mg/dL)" in columns:
            return "tandem"
        else:
            return None

    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        # Reading the header failed (missing file, unreadable, malformed CSV).
        # Per the documented contract we return None so callers can raise a
        # clear DeviceDetectionError, but we log the cause instead of hiding it.
        logger.warning("Could not read %s for device detection: %s", file_path, exc)
        return None


def create_specialized_loader(file_path: str, device_type: str | None = None, **kwargs):
    """
    Automatically creates the appropriate specialized loader.

    Args:
        file_path: Path to the file
        device_type: Device type (if ``None``, it is detected
            automatically via :func:`detect_device_type`).
        kwargs: Additional arguments forwarded to the loader constructor.

    Returns:
        Instance of the appropriate specialized loader.

    Raises:
        DeviceDetectionError: If ``device_type`` is ``None`` after
            auto-detection (i.e. the file does not match any known format and
            no explicit override was provided). The exception's
            ``columns_found`` attribute lists the first columns of the input
            file to help the caller diagnose the issue.
    """
    if device_type is None:
        device_type = detect_device_type(file_path)

    # Auto-detection returned None → cannot decide which loader to use.
    if device_type is None:
        try:
            columns_found = pd.read_csv(file_path, nrows=0).columns.tolist()
        except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
            columns_found = []
        # Cap to the first 5 columns for readability.
        columns_found = list(columns_found[:5])
        raise DeviceDetectionError(file_path, columns_found=columns_found)

    device_type = device_type.lower()

    if device_type == "dexcom":
        return Dexcom(file_path, **kwargs)
    elif device_type == "libreview":
        return Libreview(file_path, **kwargs)
    elif device_type == "medtronic":
        return MedtronicCarelink(file_path, **kwargs)
    elif device_type == "tandem":
        return TandemDiabetes(file_path, **kwargs)
    else:
        # Explicit device type that is not in the known set. Treat the same
        # way as auto-detection failure: do not silently fall back to the
        # generic loader, raise so the user can make an explicit choice.
        try:
            columns_found = pd.read_csv(file_path, nrows=0).columns.tolist()
        except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
            columns_found = []
        columns_found = list(columns_found[:5])
        raise DeviceDetectionError(file_path, columns_found=columns_found)
