"""
Module with specialized classes for specific glucose devices.

Errors raised by this module are CGMPy-specific subclasses of
:class:`ValueError` (see :mod:`cgmpy.errors`).
"""

import datetime

import pandas as pd

from ..errors import DeviceDetectionError
from .core import ModularGlucoseData

#: Tuple of device type strings recognised by :func:`detect_device_type` and
#: :func:`create_specialized_loader`. Used in error messages and for sanity
#: checks.
SUPPORTED_DEVICES: tuple[str, ...] = ("dexcom", "libreview", "medtronic", "tandem")


class Dexcom(ModularGlucoseData):
    """
    Specialized class for Dexcom device data.

    This class inherits from ModularGlucoseData and automatically configures
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

        :param file_path: Path to the exported Clarity CSV file
        :param start_date: Optional start date filter (YYYY-MM-DD)
        :param end_date: Optional end date filter (YYYY-MM-DD)
        :param log: If True, enables detailed performance logs

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
        info = self.info()
        return (
            f"Dexcom Data: {info['n_records']} readings between "
            f"{info['start_date']} and {info['end_date']}.\n"
            f"Typical interval: {info['typical_interval']:.1f} minutes.\n"
            f"Availability: {info['completeness']:.1f}%\n"
            f"Disconnections detected: {info['n_disconnections']}\n"
            f"Memory usage: {info['memory_usage_mb']:.2f} MB"
        )


class Libreview(ModularGlucoseData):
    """
    Specialized class for Libreview device data.

    This class inherits from ModularGlucoseData and automatically configures
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

        :param file_path: Path to the exported Libreview CSV file
        :param header: Header row (usually 2 for Libreview)
        :param start_date: Optional start date filter (YYYY-MM-DD)
        :param end_date: Optional end date filter (YYYY-MM-DD)
        :param log: If True, enables detailed performance logs

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
        info = self.info()
        return (
            f"Libreview Data: {info['n_records']} readings between "
            f"{info['start_date']} and {info['end_date']}.\n"
            f"Typical interval: {info['typical_interval']:.1f} minutes.\n"
            f"Availability: {info['completeness']:.1f}%\n"
            f"Disconnections detected: {info['n_disconnections']}\n"
            f"Memory usage: {info['memory_usage_mb']:.2f} MB"
        )


class MedtronicCarelink(ModularGlucoseData):
    """
    Specialized class for Medtronic CareLink device data.

    This class inherits from ModularGlucoseData and automatically configures
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

        :param file_path: Path to the exported CareLink CSV file
        :param start_date: Optional start date filter (YYYY-MM-DD)
        :param end_date: Optional end date filter (YYYY-MM-DD)
        :param log: If True, enables detailed performance logs

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
        info = self.info()
        return (
            f"Medtronic CareLink Data: {info['n_records']} readings between "
            f"{info['start_date']} and {info['end_date']}.\n"
            f"Typical interval: {info['typical_interval']:.1f} minutes.\n"
            f"Availability: {info['completeness']:.1f}%\n"
            f"Disconnections detected: {info['n_disconnections']}\n"
            f"Memory usage: {info['memory_usage_mb']:.2f} MB"
        )


class TandemDiabetes(ModularGlucoseData):
    """
    Specialized class for Tandem Diabetes device data.

    This class inherits from ModularGlucoseData and automatically configures
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

        :param file_path: Path to the exported Tandem CSV file
        :param start_date: Optional start date filter (YYYY-MM-DD)
        :param end_date: Optional end date filter (YYYY-MM-DD)
        :param log: If True, enables detailed performance logs

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
        info = self.info()
        return (
            f"Tandem Diabetes Data: {info['n_records']} readings between "
            f"{info['start_date']} and {info['end_date']}.\n"
            f"Typical interval: {info['typical_interval']:.1f} minutes.\n"
            f"Availability: {info['completeness']:.1f}%\n"
            f"Disconnections detected: {info['n_disconnections']}\n"
            f"Memory usage: {info['memory_usage_mb']:.2f} MB"
        )


def detect_device_type(file_path: str) -> str | None:
    """
    Automatically detects the device type based on the file's header.

    :param file_path: Path to the CSV file
    :return: The detected device type (``"dexcom"``, ``"libreview"``,
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

    except Exception:
        return None


def create_specialized_loader(file_path: str, device_type: str | None = None, **kwargs):
    """
    Automatically creates the appropriate specialized loader.

    :param file_path: Path to the file
    :param device_type: Device type (if ``None``, it is detected
        automatically via :func:`detect_device_type`).
    :param kwargs: Additional arguments forwarded to the loader constructor.
    :return: Instance of the appropriate specialized loader.
    :raises DeviceDetectionError: If ``device_type`` is ``None`` after
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
        except Exception:
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
        except Exception:
            columns_found = []
        columns_found = list(columns_found[:5])
        raise DeviceDetectionError(file_path, columns_found=columns_found)
