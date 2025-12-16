from .core import *

Unit2DEndo = Unit2DMappedEndomorphism
UnitEndo2D = Unit2DMappedEndomorphism
U2DE = Unit2DMappedEndomorphism
UEndo2D = Unit2DMappedEndomorphism
UNDField = UnitNdimField
MUnitField = MappedUnitField


__version__ = "1.0.0"
__all__ = [
    'InterpMethod',
    'UnitArray',
    'UnitSpaceVector',
    'UnitNdimField',
    'MappedUnitField',
    'UnitMappedEndomorphism',
    'Unit2DMappedEndomorphism',
    'remap_tensor_cv2'
]