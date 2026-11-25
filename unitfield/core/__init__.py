"""Unit field transformation module for coordinate mappings."""

from .border_config import BorderConfig, RemapBorderConfig
from .enums import BorderMode, DistanceMode, FeatheringMode, InterpMethod
from .types import UnitArray, UnitSpaceVector
from .unitfield import (
    MappedUnitField,
    Unit1DEndoInterpolator,
    Unit1dEndoInterpolator,
    Unit1DMappedEndomorphism,
    Unit2DEndoInterpolator,
    Unit2dEndoInterpolator,
    Unit2DMappedEndomorphism,
    UnitEndomorphismInterpolator,
    UnitMappedEndomorphism,
    UnitNdimField,
    remap_envelope,
    remap_tensor,
    remap_tensor_1d,
    resize_nearest,
    resize_unit_field_nearest,
    sample_callable_1d,
)

__version__ = "0.3.0"
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
    'UnitEndomorphismInterpolator',
    'Unit1DEndoInterpolator',
    'Unit1dEndoInterpolator',
    'Unit2DEndoInterpolator',
    'Unit2dEndoInterpolator',
    'remap_tensor',
    'remap_tensor_1d',
    'resize_nearest',
    'resize_unit_field_nearest',
    'remap_envelope',
    'sample_callable_1d',
]
