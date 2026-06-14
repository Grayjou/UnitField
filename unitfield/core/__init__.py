"""Unit field transformation module for coordinate mappings."""

from .border_config import BorderConfig
from .enums import BorderMode, InterpMethod
from .types import UnitArray, UnitSpaceVector
from .unitfield import (
    MappedUnitField,
    Unit1DMappedEndomorphism,
    Unit2DMappedEndomorphism,
    UnitMappedEndomorphism,
    UnitNdimField,
    remap_tensor,
    remap_tensor_1d,
)

__version__ = "1.0.0"
__all__ = [
    'InterpMethod',
    'BorderMode',
    'BorderConfig',
    'UnitArray',
    'UnitSpaceVector',
    'UnitNdimField',
    'MappedUnitField',
    'UnitMappedEndomorphism',
    'Unit2DMappedEndomorphism',
    'Unit1DMappedEndomorphism',
    'remap_tensor',
    'remap_tensor_1d',
]
