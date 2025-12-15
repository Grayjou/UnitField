#UnitFied\unitfield\core\enums.py
"""
Interpolation method enumerations for unit field transformations.
"""

from enum import Enum, unique
from typing import Dict, Set



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
    def get_cv2_methods(cls) -> Set['InterpMethod']:
        """Get interpolation methods supported by OpenCV backend."""
        return {
            cls.NEAREST_MANHATTAN,
            cls.LINEAR,
            cls.NEAREST_EUCLIDEAN,
            cls.CUBIC,
            cls.LANCZOS4
        }
    
    @classmethod
    def get_numpy_methods(cls) -> Dict['InterpMethod', str]:
        """Get mapping of interpolation methods to numpy backend names."""
        return {
            cls.NEAREST_MANHATTAN: "nearest_manhattan",
            cls.LINEAR: "linear",
            cls.NEAREST_EUCLIDEAN: "nearest_euclidean",
            cls.CUBIC: "cubic",
            cls.LANCZOS4: "lanczos4"
        }