"""Explicit conversions between array indices and endpoint-aligned unit coordinates."""

from __future__ import annotations

import numpy as np

from ..core.enums import InterpMethod


def _validate_shape(spatial_shape: tuple[int, ...]) -> np.ndarray:
    try:
        shape = tuple(spatial_shape)
    except TypeError as exc:
        raise TypeError("spatial_shape must be an iterable of positive integers") from exc
    if not shape:
        raise ValueError("spatial_shape must contain at least one dimension")
    if any(isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer)) for size in shape):
        raise TypeError(f"spatial_shape must contain integers, got {shape}")
    if any(size <= 0 for size in shape):
        raise ValueError(f"spatial_shape must contain positive dimensions, got {shape}")
    return np.asarray(shape, dtype=np.intp)


def _validate_real_finite(values: np.ndarray, *, name: str, vector_size: int | None = None) -> np.ndarray:
    if values.ndim == 0:
        raise ValueError(f"{name} must contain coordinate vectors, not a scalar")
    if vector_size is not None and values.shape[-1] != vector_size:
        raise ValueError(f"{name} dimension {values.shape[-1]} does not match {vector_size}")
    if values.dtype.kind not in "iuf":
        raise TypeError(f"{name} must have a real numeric dtype, got {values.dtype}")
    result = values.astype(np.float64, copy=False)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _validate_float_dtype(dtype: np.dtype | type, *, name: str) -> np.dtype:
    result = np.dtype(dtype)
    if result.kind != "f":
        raise TypeError(f"{name} must be a real floating-point dtype, got {result}")
    return result


def _validate_resolution(width: int, height: int) -> tuple[int, int]:
    if isinstance(width, (bool, np.bool_)) or not isinstance(width, (int, np.integer)):
        raise TypeError(f"width must be an integer, got {type(width)}")
    if isinstance(height, (bool, np.bool_)) or not isinstance(height, (int, np.integer)):
        raise TypeError(f"height must be an integer, got {type(height)}")
    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive, got {(width, height)}")
    return int(width), int(height)


def pixel_to_unit(coords: np.ndarray, spatial_shape: tuple[int, ...]) -> np.ndarray:
    """Convert array-axis pixel coordinates to endpoint-aligned unit coordinates.

    Coordinate vectors use NumPy axis order. A singleton axis maps its sole
    pixel to unit coordinate zero; inputs on that axis therefore map to zero.
    Empty dimensions and nonfinite coordinates are rejected. Finite
    out-of-domain coordinates are not clipped, so border-aware values can
    round-trip predictably. Integer and unsigned inputs are accepted and
    converted to float64.
    """
    sizes = _validate_shape(spatial_shape)
    values = _validate_real_finite(np.asarray(coords), name="coords", vector_size=sizes.size)
    scale = np.maximum(sizes.astype(np.float64) - 1.0, 1.0)
    result = values / scale
    return np.where(sizes == 1, 0.0, result)


def unit_to_pixel(coords: np.ndarray, spatial_shape: tuple[int, ...]) -> np.ndarray:
    """Convert endpoint-aligned unit coordinates to array-axis pixel coordinates.

    Singleton axes always convert to zero. Inputs are not clipped, which keeps
    this suitable for converting border-handled out-of-domain coordinates.
    """
    sizes = _validate_shape(spatial_shape)
    values = _validate_real_finite(np.asarray(coords), name="coords", vector_size=sizes.size)
    scale = np.maximum(sizes.astype(np.float64) - 1.0, 1.0)
    result = values * scale
    return np.where(sizes == 1, 0.0, result)


def xy_to_yx(coords: np.ndarray) -> np.ndarray:
    """Return 2-D coordinate vectors reordered from geometric ``(x, y)`` to ``(y, x)``."""
    values = _validate_real_finite(np.asarray(coords), name="coords", vector_size=2)
    return values[..., ::-1]


def yx_to_xy(coords: np.ndarray) -> np.ndarray:
    """Return 2-D coordinate vectors reordered from array ``(y, x)`` to ``(x, y)``."""
    return xy_to_yx(coords)


def identity_coordinate_field(width: int, height: int, *, dtype=np.float32) -> np.ndarray:
    """Create a finite floating ``field[y, x] = (x_unit, y_unit)`` identity map."""
    width, height = _validate_resolution(width, height)
    dtype = _validate_float_dtype(dtype, name="dtype")
    ys, xs = np.meshgrid(
        np.linspace(0.0, 1.0, height, dtype=dtype),
        np.linspace(0.0, 1.0, width, dtype=dtype),
        indexing="ij",
    )
    return np.stack((xs, ys), axis=-1)


def absolute_to_displacement(field: np.ndarray) -> np.ndarray:
    """Convert an absolute 2-D ``(source_x, source_y)`` field to displacement."""
    values = _validate_real_finite(np.asarray(field), name="field", vector_size=2)
    if values.ndim != 3 or values.shape[-1] != 2:
        raise ValueError(f"field must have shape (height, width, 2), got {values.shape}")
    identity = identity_coordinate_field(values.shape[1], values.shape[0], dtype=np.float64)
    return values - identity


def displacement_to_absolute(displacement: np.ndarray) -> np.ndarray:
    """Convert a 2-D displacement field to absolute ``(source_x, source_y)`` coordinates."""
    values = _validate_real_finite(np.asarray(displacement), name="displacement", vector_size=2)
    if values.ndim != 3 or values.shape[-1] != 2:
        raise ValueError(f"displacement must have shape (height, width, 2), got {values.shape}")
    identity = identity_coordinate_field(values.shape[1], values.shape[0], dtype=np.float64)
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

    values = _validate_real_finite(np.asarray(field), name="field", vector_size=2)
    if values.ndim != 3 or values.shape[-1] != 2:
        raise ValueError(f"field must have shape (height, width, 2), got {values.shape}")
    width, height = _validate_resolution(width, height)
    result_dtype = np.dtype(values.dtype if dtype is None else dtype)
    _validate_float_dtype(result_dtype, name="dtype")
    endomorphism = Unit2DMappedEndomorphism(values, interp_method=interpolation, cache_size=0)
    ys, xs = np.meshgrid(
        np.linspace(0.0, 1.0, height), np.linspace(0.0, 1.0, width), indexing="ij"
    )
    return endomorphism.get_values(np.stack((ys, xs), axis=-1)).astype(result_dtype, copy=False)
