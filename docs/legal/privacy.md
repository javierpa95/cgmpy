# Privacy

CGMPy is a **research and analysis library**. It does not collect,
transmit, or store any user data. The library operates entirely
locally on the data you provide.

## Data you provide to CGMPy

When you call `ModularGlucoseData("my_file.csv")`, CGMPy reads the file
from your local disk and processes it in memory. **No data ever leaves
your machine** as a result of using the library.

## Data you share with the CGMPy project

CGMPy does not have a backend. Sharing data with the project is
entirely under your control and happens only when you:

- Open an **issue** (GitHub Issues).
- Open a **discussion** (GitHub Discussions).
- Open a **pull request**.

In all of these cases, **you decide what to share**. Please review the
[Security policy](https://github.com/javierpa95/cgmpy/blob/main/SECURITY.md)
before sharing data, especially medical data.

## Cookies and tracking

The CGMPy documentation site is hosted on **GitHub Pages** and does not
set any cookies. No analytics, no tracking, no third-party scripts.

## Telemetry

CGMPy does not phone home, does not collect usage statistics, and does
not embed any analytics SDK.

## What CGMPy does log

CGMPy uses the Python `logging` module. By default, log levels are
`WARNING` and above. The library does not log glucose values, patient
IDs, or any other potentially sensitive data.

If you enable `INFO` or `DEBUG` logging, you may see messages like:

```
INFO: cgmpy.data.loader: Loaded 1728 records from /path/to/file.csv
INFO: cgmpy.data.processor: Dropped 3 duplicate rows
```

These messages **do not** include glucose values or patient identifiers.

## Compliance

CGMPy is released under the **MIT License** and ships without warranty.
The library is not certified under HIPAA, GDPR, or any other medical-
data regulation. **It is the user's responsibility** to ensure that
their use of the library complies with applicable laws in their
jurisdiction.

## See also

- [GDPR notes](gdpr.md).
- [Security policy](https://github.com/javierpa95/cgmpy/blob/main/SECURITY.md).
- [License](https://github.com/javierpa95/cgmpy/blob/main/LICENSE).
