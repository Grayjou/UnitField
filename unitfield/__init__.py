from .core.border_config import BorderConfig, RemapBorderConfig
from .core.enums import BorderMode, DistanceMode, FeatheringMode, InterpMethod
from .core.types import UnitArray, UnitSpaceVector
from .core.unitfield import (
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
from .cython._zero_sign import ZERO_SIGN

try:
    from .cython._clip_abs import clip_abs
except ImportError:
    from .cython._clip_abs_fallback import clip_abs
from .utilities import (
    flat_1d_pbm,
    flat_1d_upbm,
    pbm_2d,
    pbm_ndim,
    positional_basematrix2d,
    positional_basematrix_ndim,
    unit_positional_basematrix2d,
    unit_positional_basematrix_ndim,
    upbm_2d,
    upbm_ndim,
)

Unit2DEndo = Unit2DMappedEndomorphism
UnitEndo2D = Unit2DMappedEndomorphism
U2DE = Unit2DMappedEndomorphism
UEndo2D = Unit2DMappedEndomorphism
Unit1DEndo = Unit1DMappedEndomorphism
UnitEndo1D = Unit1DMappedEndomorphism
U1DE = Unit1DMappedEndomorphism
UEndo1D = Unit1DMappedEndomorphism
U1DEInterpolator = Unit1DEndoInterpolator
UEndo1DInterpolator = Unit1DEndoInterpolator
U2DEInterpolator = Unit2DEndoInterpolator
UEndo2DInterpolator = Unit2DEndoInterpolator
UNDField = UnitNdimField
MUnitField = MappedUnitField


__version__ = "0.2.12"
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
    'clip_abs',
    'ZERO_SIGN',
    'positional_basematrix2d',
    'unit_positional_basematrix2d',
    'positional_basematrix_ndim',
    'unit_positional_basematrix_ndim',
    'pbm_2d',
    'upbm_2d',
    'pbm_ndim',
    'upbm_ndim',
    'flat_1d_upbm',
    'flat_1d_pbm',
]
