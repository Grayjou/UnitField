"""Unit field transformation module for coordinate mappings."""

from .border_config import BorderConfig, RemapBorderConfig
from .enums import BorderMode, DistanceMode, FeatheringMode, InterpMethod
from .types import UnitArray, UnitSpaceVector
from .unitfield import (
    MappedUnitField,
    Unit1DMappedEndomorphism,
    Unit2DMappedEndomorphism,
    UnitMappedEndomorphism,
    UnitNdimField,
    remap_envelope,
    remap_tensor,
    remap_tensor_1d,
    sample_callable_1d,
)

__version__ = "0.2.10"
__all__ = [
    'InterpMethod',
    'BorderMode',
    'DistanceMode',
    'FeatheringMode',
    'BorderConfig',
    'RemapBorderConfig',
    'UnitArray',
    'UnitSpaceVector',
    'UnitNdimField',
    'MappedUnitField',
    'UnitMappedEndomorphism',
    'Unit2DMappedEndomorphism',
    'Unit1DMappedEndomorphism',
    'remap_tensor',
    'remap_tensor_1d',
    'remap_envelope',
    'sample_callable_1d',
]
