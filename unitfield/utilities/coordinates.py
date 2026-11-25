"""Explicit conversions between array indices and endpoint-aligned unit coordinates."""

from __future__ import annotations

import numpy as np

from ..core.enums import InterpMethod


def _shape_scale(spatial_shape: tuple[int, ...]) -> np.ndarray:
    if not spatial_shape or any(size <= 0 for size in spatial_shape):
        raise ValueError(f"spatial_shape must contain positive dimensions, got {spatial_shape}")
    return np.maximum(np.asarray(spatial_shape, dtype=np.float64) - 1.0, 1.0)


def pixel_to_unit(coords: np.ndarray, spatial_shape: tuple[int, ...]) -> np.ndarray:
    """Convert array-axis pixel coordinates to endpoint-aligned unit coordinates.

    Coordinate vectors use NumPy axis order. A singleton axis maps its sole
    pixel to unit coordinate zero; inputs on that axis therefore map to zero.
    Empty dimensions are rejected. This function neither clips nor validates
    finite values, so out-of-domain coordinates round-trip predictably.
    """
    values = np.asarray(coords)
    sizes = np.asarray(spatial_shape, dtype=np.intp)
    scale = _shape_scale(tuple(spatial_shape))
    if values.shape[-1] != scale.size:
        raise ValueError(f"coordinate dimension {values.shape[-1]} does not match {scale.size}")
    result = np.asarray(values, dtype=np.result_type(values, np.float64)) / scale
    return np.where(sizes == 1, 0.0, result)


def unit_to_pixel(coords: np.ndarray, spatial_shape: tuple[int, ...]) -> np.ndarray:
    """Convert endpoint-aligned unit coordinates to array-axis pixel coordinates.

    Singleton axes always convert to zero. Inputs are not clipped, which keeps
    this suitable for converting border-handled out-of-domain coordinates.
    """
    values = np.asarray(coords)
    sizes = np.asarray(spatial_shape, dtype=np.intp)
    scale = _shape_scale(tuple(spatial_shape))
    if values.shape[-1] != scale.size:
        raise ValueError(f"coordinate dimension {values.shape[-1]} does not match {scale.size}")
    result = np.asarray(values, dtype=np.result_type(values, np.float64)) * scale
    return np.where(sizes == 1, 0.0, result)


def xy_to_yx(coords: np.ndarray) -> np.ndarray:
    """Return 2-D coordinate vectors reordered from geometric ``(x, y)`` to ``(y, x)``."""
    values = np.asarray(coords)
    if values.shape[-1] != 2:
        raise ValueError(f"expected coordinate vectors with length 2, got {values.shape[-1]}")
    return values[..., ::-1]


def yx_to_xy(coords: np.ndarray) -> np.ndarray:
    """Return 2-D coordinate vectors reordered from array ``(y, x)`` to ``(x, y)``."""
    return xy_to_yx(coords)


def identity_coordinate_field(width: int, height: int, *, dtype=np.float32) -> np.ndarray:
    """Create ``field[y, x] = (x_unit, y_unit)`` at the requested resolution."""
    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive, got {(width, height)}")
    ys, xs = np.meshgrid(
        np.linspace(0.0, 1.0, height, dtype=dtype),
        np.linspace(0.0, 1.0, width, dtype=dtype),
        indexing="ij",
    )
    return np.stack((xs, ys), axis=-1)


def absolute_to_displacement(field: np.ndarray) -> np.ndarray:
    """Convert an absolute 2-D ``(source_x, source_y)`` field to displacement."""
    values = np.asarray(field)
    if values.ndim != 3 or values.shape[-1] != 2:
        raise ValueError(f"field must have shape (height, width, 2), got {values.shape}")
    identity = identity_coordinate_field(values.shape[1], values.shape[0], dtype=values.dtype)
    return values - identity


def displacement_to_absolute(displacement: np.ndarray) -> np.ndarray:
    """Convert a 2-D displacement field to absolute ``(source_x, source_y)`` coordinates."""
    values = np.asarray(displacement)
    if values.ndim != 3 or values.shape[-1] != 2:
        raise ValueError(f"displacement must have shape (height, width, 2), got {values.shape}")
    identity = identity_coordinate_field(values.shape[1], values.shape[0], dtype=values.dtype)
    return values + identity


def resample_coordinate_field(
    field: np.ndarray,
    width: int,
    height: int,
    *,
    interpolation: InterpMethod = InterpMethod.LINEAR,
    dtype: np.dtype | type | None = None,
) -> np.ndarray:
    """Resample an absolute 2-D coordinate field onto an endpoint-aligned grid.

    This interpolates field values, not a source tensor. ``field`` follows the
    conventional storage contract ``field[y, x] = (source_x, source_y)``.
    """
    from ..core.unitfield import Unit2DMappedEndomorphism

    values = np.asarray(field)
    if values.ndim != 3 or values.shape[-1] != 2:
        raise ValueError(f"field must have shape (height, width, 2), got {values.shape}")
    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive, got {(width, height)}")
    result_dtype = values.dtype if dtype is None else dtype
    endomorphism = Unit2DMappedEndomorphism(values, interp_method=interpolation, cache_size=0)
    ys, xs = np.meshgrid(
        np.linspace(0.0, 1.0, height), np.linspace(0.0, 1.0, width), indexing="ij"
    )
    return endomorphism.get_values(np.stack((ys, xs), axis=-1)).astype(result_dtype, copy=False)
