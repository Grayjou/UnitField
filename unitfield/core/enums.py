#UnitFied\unitfield\core\enums.py
"""
Interpolation method and border mode enumerations for unit field transformations.
"""

from enum import Enum, IntEnum


class InterpMethod(Enum):
    """Interpolation methods for unit field sampling."""

    NEAREST_MANHATTAN = "nearest"
    LINEAR = "linear"
    NEAREST_EUCLIDEAN = "euclidean"
    LANCZOS4 = "lanczos4"
    CUBIC = "cubic"

    # Backward compatibility alias
    EUCLIDEAN = "euclidean"

    @classmethod
    def get_cv2_methods(cls) -> set['InterpMethod']:
        """Get interpolation methods supported by OpenCV backend."""
        return {
            cls.NEAREST_MANHATTAN,
            cls.LINEAR,
            cls.NEAREST_EUCLIDEAN,
            cls.CUBIC,
            cls.LANCZOS4
        }

    @classmethod
    def get_numpy_methods(cls) -> dict['InterpMethod', str]:
        """Get mapping of interpolation methods to numpy backend names."""
        return {
            cls.NEAREST_MANHATTAN: "nearest_manhattan",
            cls.LINEAR: "linear",
            cls.NEAREST_EUCLIDEAN: "nearest_euclidean",
            cls.CUBIC: "cubic",
            cls.LANCZOS4: "lanczos4"
        }


class BorderMode(IntEnum):
    """Border handling modes for coordinate remapping.

    Mirrors OpenCV border modes but with cleaner names. Used by BorderConfig
    and the Cython remap kernel.
    """
    CLAMP       = 0  # Clamp to edge (cv2.BORDER_REPLICATE / BORDER_CLAMP)
    CONSTANT    = 1  # Fill with constant value (cv2.BORDER_CONSTANT)
    REFLECT     = 2  # Mirror at edge (cv2.BORDER_REFLECT)
    WRAP        = 3  # Tile / repeat (cv2.BORDER_WRAP)
    REFLECT_101 = 4  # Mirror with edge pixel repeated (cv2.BORDER_REFLECT_101)
    ARRAY       = 5  # Per-pixel border from a background array
