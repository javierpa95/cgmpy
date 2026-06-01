"""
Utilities module for handling glucose data.

This module contains helper functions and classes for:
- Date and time utilities
- Medical data validators
- Centralised configuration
- General helper functions
"""

# Imports that will be available when implemented
# from .date_utils import parse_date
# from .validators import DataValidator
# from .config import GlucoseConfig


# For now, we import from the current location to maintain backward compatibility
# We avoid circular imports by using a lazy import
def parse_date(*args, **kwargs):
    from ..utils import parse_date as _parse_date

    return _parse_date(*args, **kwargs)


__all__ = ["parse_date"]
