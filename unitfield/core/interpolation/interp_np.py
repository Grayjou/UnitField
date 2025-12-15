#UnitFied\unitfield\core\interpolation\interp_np.py
"""
NumPy-based interpolation backend for N-dimensional unit fields.
"""

from functools import lru_cache
from typing import Dict, Callable, Tuple
from itertools import product

import numpy as np

from ..enums import InterpMethod
from ..types import DEFAULT_DTYPE


def _scale_coords(coords: np.ndarray, spatial_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Scale unit coordinates [0, 1] to array indices.
    
    Args:
        coords: Unit coordinates array of shape (..., D)
        spatial_shape: Spatial dimensions of the field
    
    Returns:
        Scaled coordinates in array index space
    """
    coords = np.asarray(coords)
    D = coords.shape[-1]  # number of spatial dimensions

    spatial_shape_array = np.asarray(spatial_shape[:D], dtype=DEFAULT_DTYPE)
    return coords * (spatial_shape_array - 1)

def _ensure_float_array(arr: np.ndarray) -> np.ndarray:
    # Convert integer data to float for interpolation
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float64)
    return arr

def _safe_indexing(
    data: np.ndarray,
    indices: np.ndarray,
    spatial_shape: Tuple[int, ...]
) -> np.ndarray:
    """
    Safe multi-dimensional array indexing with clipping.
    Only spatial dimensions are indexed; remaining dims are values.
    """
    indices = np.asarray(indices)
    D = indices.shape[-1]  # number of spatial dimensions
    
    # Only use first D dimensions for spatial clipping
    spatial_shape_spatial = np.asarray(spatial_shape[:D], dtype=int)
    
    spatial_min = np.zeros(D, dtype=int)
    spatial_max = spatial_shape_spatial - 1
    
    clipped_indices = np.clip(indices, spatial_min, spatial_max)
    
    # Create index tuple for spatial dimensions
    index_tuple = tuple(clipped_indices[..., i] for i in range(D))
    
    # Add slice(None) for all remaining (value) dimensions
    if data.ndim > D:
        index_tuple += (slice(None),) * (data.ndim - D)
    
    return data[index_tuple]

@lru_cache(maxsize=5)
def _get_kernel_weights(
    kernel_type: str,
    radius: int = 2
) -> Dict[Tuple[int, ...], float]:
    """
    Cache pre-computed kernel weights for common interpolation methods.
    """
    # Implementation for cubic, lanczos, etc.
    # This is a placeholder - implement based on your specific needs
    return {}


def nearest_manhattan_vectorized_arr(
    coords: np.ndarray,
    spatial_shape: Tuple[int, ...],
    data: np.ndarray
) -> np.ndarray:
    """
    Nearest neighbor interpolation using Manhattan distance.
    
    Args:
        coords: Unit coordinates array of shape (..., D)
        spatial_shape: Spatial dimensions of the field
        data: Source data array
    
    Returns:
        Interpolated values
    """
    # Scale and round to nearest integer indices
    scaled = _scale_coords(coords, spatial_shape)
    indices = np.floor(scaled + 0.5).astype(int)
    
    return _safe_indexing(data, indices, spatial_shape)


def linear_vectorized_arr(
    coords: np.ndarray,
    spatial_shape: Tuple[int, ...],
    data: np.ndarray
) -> np.ndarray:
    """
    Linear (bilinear/trilinear) interpolation.
    
    Args:
        coords: Unit coordinates array of shape (..., D)
        spatial_shape: Spatial dimensions of the field
        data: Source data array
    
    Returns:
        Interpolated values
    """
    # Extract spatial dimensions (D) from coords
    D = coords.shape[-1]
    
    data = _ensure_float_array(data)

    # spatial_shape should only contain spatial dimensions
    spatial_shape_array = np.asarray(spatial_shape[:D])
    
    p = _scale_coords(coords, spatial_shape)
    base = np.floor(p).astype(int)
    
    # Only clip spatial dimensions
    base = np.clip(base, 0, spatial_shape_array - 2)
    
    frac = p - base
    
    # Check if there are value dimensions (like channels)
    has_value_dims = data.ndim > D
    
    if has_value_dims:
        # Get the shape for result - preserve batch and value dimensions
        result_shape = coords.shape[:-1] + data.shape[D:]
        result = np.zeros(result_shape, dtype=data.dtype)
    else:
        # Scalar-valued data
        result_shape = coords.shape[:-1]
        result = np.zeros(result_shape, dtype=data.dtype)
    
    # Iterate over all 2^D corners
    for corner in product([0, 1], repeat=D):
        corner_array = np.array(corner)
        idx = base + corner_array
        
        # Calculate weight for spatial interpolation
        weight = np.prod(
            np.where(corner_array, frac, 1 - frac),
            axis=-1
        )
        
        # Get the interpolated values at this corner
        corner_values = _safe_indexing(data, idx, spatial_shape)
        
        if has_value_dims:
            # Add extra dimension for broadcasting with value dimensions
            weight = np.expand_dims(weight, axis=-1)
            result += weight * corner_values
        else:
            # Scalar multiplication
            result += weight * corner_values
    
    return result


def cubic_vectorized_arr(coords, spatial_shape, data):
    """
    Cubic convolution interpolation (a=-0.5, Catmull-Rom spline).
    
    Args:
        coords: Unit coordinates array of shape (..., D)
        spatial_shape: Spatial dimensions of the field
        data: Source data array
    
    Returns:
        Interpolated values
    """
    def cubic_kernel(t, a=-0.5):
        at = np.abs(t)
        return np.where(
            at <= 1,
            (a + 2) * at**3 - (a + 3) * at**2 + 1,
            np.where(
                at < 2,
                a * at**3 - 5 * a * at**2 + 8 * a * at - 4 * a,
                0.0
            )
        )
    
    p = _scale_coords(coords, spatial_shape)
    base = np.floor(p).astype(int)
    frac = p - base
    
    D = coords.shape[-1]
    
    data = _ensure_float_array(data)
    # Check if there are value dimensions
    has_value_dims = data.ndim > D
    
    if has_value_dims:
        result_shape = coords.shape[:-1] + data.shape[D:]
        result = np.zeros(result_shape, dtype=data.dtype)
    else:
        result_shape = coords.shape[:-1]
        result = np.zeros(result_shape, dtype=data.dtype)
    
    # Only use spatial dimensions for clipping
    spatial_shape_array = np.asarray(spatial_shape[:D])
    
    for offset in product(range(-1, 3), repeat=D):
        offset_array = np.array(offset)
        idx = np.clip(base + offset_array, 0, spatial_shape_array - 1)
        
        # Calculate weight per coordinate
        weights = np.prod(
            cubic_kernel(frac - offset_array),
            axis=-1
        )
        
        corner_values = _safe_indexing(data, idx, spatial_shape)
        
        if has_value_dims:
            # Add extra dimension for broadcasting with value dimensions
            weights_expanded = np.expand_dims(weights, axis=-1)
            result += weights_expanded * corner_values
        else:
            result += weights * corner_values
    
    return result

def lanczos4_vectorized_arr(
    coords: np.ndarray,
    spatial_shape: Tuple[int, ...],
    data: np.ndarray
) -> np.ndarray:
    """
    Lanczos-4 interpolation.
    
    Args:
        coords: Unit coordinates array of shape (..., D)
        spatial_shape: Spatial dimensions of the field
        data: Source data array
    
    Returns:
        Interpolated values
    """
    def lanczos_kernel(t: np.ndarray, a: int = 4) -> np.ndarray:
        """Lanczos kernel."""
        at = np.abs(t)
        result = np.zeros_like(t)
        mask = at < a
        t_masked = t[mask]
        result[mask] = np.sinc(t_masked) * np.sinc(t_masked / a)
        return result
    
    p = _scale_coords(coords, spatial_shape)
    base = np.floor(p).astype(int)
    frac = p - base
    
    D = coords.shape[-1]

    data = _ensure_float_array(data)
    
    # Check if there are value dimensions
    has_value_dims = data.ndim > D
    
    if has_value_dims:
        result_shape = coords.shape[:-1] + data.shape[D:]
        result = np.zeros(result_shape, dtype=data.dtype)
    else:
        result_shape = coords.shape[:-1]
        result = np.zeros(result_shape, dtype=data.dtype)
    
    # Only use spatial dimensions for clipping
    spatial_shape_array = np.asarray(spatial_shape[:D])
    
    # Iterate over 8^D neighborhood (radius 4)
    for offset in product(range(-3, 5), repeat=D):
        offset_array = np.array(offset)
        idx = np.clip(base + offset_array, 0, spatial_shape_array - 1)
        
        weights = np.prod(
            lanczos_kernel(frac - offset_array, a=4),
            axis=-1
        )
        
        corner_values = _safe_indexing(data, idx, spatial_shape)
        
        if has_value_dims:
            # Add extra dimension for broadcasting with value dimensions
            weights_expanded = np.expand_dims(weights, axis=-1)
            result += weights_expanded * corner_values
        else:
            result += weights * corner_values
    
    return result

# Mapping of interpolation methods to their implementations
np_interp_dict: Dict[InterpMethod, Callable] = {
    InterpMethod.NEAREST_MANHATTAN: nearest_manhattan_vectorized_arr,
    InterpMethod.LINEAR: linear_vectorized_arr,
    InterpMethod.NEAREST_EUCLIDEAN: nearest_manhattan_vectorized_arr,
    InterpMethod.CUBIC: cubic_vectorized_arr,
    InterpMethod.LANCZOS4: lanczos4_vectorized_arr,
}


def get_numpy_interpolator(method: InterpMethod) -> Callable:
    """
    Get numpy-based interpolator function for a given method.
    
    Args:
        method: Interpolation method
    
    Returns:
        Interpolator function
    
    Raises:
        ValueError: If method is not supported by numpy backend
    """
    if method not in np_interp_dict:
        raise ValueError(
            f"Interpolation method {method} not supported by numpy backend. "
            f"Supported methods: {list(np_interp_dict.keys())}"
        )
    return np_interp_dict[method]