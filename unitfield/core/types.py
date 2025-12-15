#UnitFied\unitfield\core\types.py
"""
Type definitions for unit field transformations.
"""

from typing import Annotated, Tuple, List, Union, TypeAlias, Final
from typing_extensions import TypeAlias as TypeAliasExt
import numpy as np
from numpy.typing import NDArray
from boundednumbers import UnitFloat


# Unit array type (values in [0, 1])
UnitArray: TypeAlias = Annotated[NDArray[np.floating], "values in [0, 1]"]

# Unit space vector types
UnitSpaceVector: TypeAlias = Union[
    UnitArray,
    Tuple[UnitFloat, ...],
    List[UnitFloat],
]

# Type aliases for clarity
Coordinate: TypeAlias = Tuple[UnitFloat, ...]
PixelCoordinate: TypeAlias = Tuple[float, ...]
ImageShape: TypeAlias = Tuple[int, int]

# Constants
MAX_CACHE_SIZE: Final[int] = 1024
DEFAULT_CACHE_SIZE: Final[int] = 128
DEFAULT_DTYPE: Final[type] = np.float32