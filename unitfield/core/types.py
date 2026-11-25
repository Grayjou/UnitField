#UnitFied\unitfield\core\types.py
"""
Type definitions for unit field transformations.
"""

from typing import Annotated, Final, TypeAlias

import numpy as np
from boundednumbers import UnitFloat
from numpy.typing import NDArray

# Sampled coordinate/vector array. Values may leave [0, 1] when a caller wants
# remapping border behavior; this alias is descriptive rather than validating.
UnitArray: TypeAlias = Annotated[NDArray[np.floating], "sampled coordinate/vector values"]

# Unit space vector types
UnitSpaceVector: TypeAlias = UnitArray | tuple[UnitFloat, ...] | list[UnitFloat]

# Type aliases for clarity
Coordinate: TypeAlias = tuple[UnitFloat, ...]
PixelCoordinate: TypeAlias = tuple[float, ...]
# Query coordinates for sampled fields follow NumPy spatial-axis order, e.g.
# (y, x), while coordinate-field values conventionally use geometric (x, y).
AxisCoordinate: TypeAlias = Coordinate
XYCoordinate: TypeAlias = tuple[UnitFloat, UnitFloat]
ImageShape: TypeAlias = tuple[int, int]

# Constants
MAX_CACHE_SIZE: Final[int] = 1024
DEFAULT_CACHE_SIZE: Final[int] = 128
DEFAULT_DTYPE: Final[type] = np.float32
