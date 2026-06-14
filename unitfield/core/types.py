#UnitFied\unitfield\core\types.py
"""
Type definitions for unit field transformations.
"""

from typing import Annotated, Final, TypeAlias

import numpy as np
from boundednumbers import UnitFloat  # type: ignore[attr-defined]
from numpy.typing import NDArray

# Unit array type (values in [0, 1])
UnitArray: TypeAlias = Annotated[NDArray[np.floating], "values in [0, 1]"]

# Unit space vector types
UnitSpaceVector: TypeAlias = UnitArray | tuple[UnitFloat, ...] | list[UnitFloat]

# Type aliases for clarity
Coordinate: TypeAlias = tuple[UnitFloat, ...]
PixelCoordinate: TypeAlias = tuple[float, ...]
ImageShape: TypeAlias = tuple[int, int]

# Constants
MAX_CACHE_SIZE: Final[int] = 1024
DEFAULT_CACHE_SIZE: Final[int] = 128
DEFAULT_DTYPE: Final[type] = np.float32
