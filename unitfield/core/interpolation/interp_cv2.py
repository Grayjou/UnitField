#UnitFied\unitfield\core\interpolation\interp_cv2.py
"""
OpenCV-based interpolation backend for 2D unit fields.
"""

from functools import partial, lru_cache
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from ..enums import InterpMethod


# OpenCV interpolation method mapping
_CV2_INTERP_MAP: Dict[InterpMethod, int] = {
    InterpMethod.NEAREST_MANHATTAN: cv2.INTER_NEAREST,
    InterpMethod.LINEAR: cv2.INTER_LINEAR,
    InterpMethod.NEAREST_EUCLIDEAN: cv2.INTER_NEAREST,
    InterpMethod.CUBIC: cv2.INTER_CUBIC,
    InterpMethod.LANCZOS4: cv2.INTER_LANCZOS4,
}

# Default border settings
_DEFAULT_BORDER_MODE: int = cv2.BORDER_REPLICATE
_DEFAULT_BORDER_VALUE: float = 0.0


@lru_cache(maxsize=10)
def _create_cv2_remap_function(
    interpolation: int,
    border_mode: int = _DEFAULT_BORDER_MODE,
    border_value: float = _DEFAULT_BORDER_VALUE
) -> callable:
    """Create a cached cv2.remap function with specified parameters."""
    return partial(
        cv2.remap,
        interpolation=interpolation,
        borderMode=border_mode,
        borderValue=border_value
    )


def _scale_to_pixel_space(
    coords_array: np.ndarray,
    height: int,
    width: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scale unit-space coordinates [0, 1] to pixel-space coordinates.
    
    Args:
        coords_array: Array of shape (..., 2) with unit-space coordinates
        height: Image height in pixels
        width: Image width in pixels
    
    Returns:
        Tuple of (map_x, map_y) arrays for cv2.remap
    """
    if width > 1:
        scale_x = width - 1
    else:
        scale_x = 1.0
    
    if height > 1:
        scale_y = height - 1
    else:
        scale_y = 1.0
    
    map_x = (coords_array[..., 0] * scale_x).astype(np.float32)
    map_y = (coords_array[..., 1] * scale_y).astype(np.float32)
    
    return map_x, map_y


def cv2_unit_field_sample(
    data: np.ndarray,
    coords_array: np.ndarray,
    interp_method: InterpMethod,
    border_mode: Optional[int] = None,
    border_value: Optional[float] = None
) -> np.ndarray:
    """
    Sample a 2D unit-space field using OpenCV remap.
    
    Args:
        data: Array of shape (H, W, C) or (H, W)
        coords_array: Array of shape (..., 2), unit-space coordinates in [0, 1]
        interp_method: Interpolation method enum
        border_mode: cv2 border mode (default: BORDER_REPLICATE)
        border_value: Border value for constant border mode
    
    Returns:
        Interpolated values at coords_array
    
    Raises:
        ValueError: If coords_array is not 2D or method not supported
        RuntimeError: If OpenCV remap fails
    """
    # Validation
    if coords_array.shape[-1] != 2:
        raise ValueError(
            f"OpenCV backend only supports 2D fields. "
            f"Got coords_array.shape[-1] = {coords_array.shape[-1]}"
        )
    
    if interp_method not in InterpMethod.get_cv2_methods():
        raise ValueError(
            f"Unsupported interpolation method for OpenCV backend: {interp_method}"
        )
    
    if border_mode is None:
        border_mode = _DEFAULT_BORDER_MODE
    if border_value is None:
        border_value = _DEFAULT_BORDER_VALUE
    
    # Get image dimensions
    if data.ndim == 2:
        height, width = data.shape
    else:
        height, width = data.shape[:2]
    
    # Scale to pixel space
    map_x, map_y = _scale_to_pixel_space(coords_array, height, width)
    
    # Get cv2 interpolation constant
    cv2_interp = _CV2_INTERP_MAP[interp_method]
    
    # Get remap function
    remap_fn = _create_cv2_remap_function(cv2_interp, border_mode, border_value)
    
    try:
        return remap_fn(data, map_x, map_y)
    except cv2.error as e:
        raise RuntimeError(f"OpenCV remap failed: {str(e)}") from e


# Create cv2 interpolation dictionary (backward compatibility)
cv2_interp_dict = {
    method: _create_cv2_remap_function(cv2_constant)
    for method, cv2_constant in _CV2_INTERP_MAP.items()
}