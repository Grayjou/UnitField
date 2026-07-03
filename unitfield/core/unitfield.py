# UnitField\unitfield\core\unitfield.py
"""
Main unit field transformation classes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import cast

import numpy as np

from ..interpolation import (
    np_interp_dict,
)
from .border_config import BorderConfig
from .enums import BorderMode, InterpMethod
from .types import (
    DEFAULT_CACHE_SIZE,
    DEFAULT_DTYPE,
    Coordinate,
    UnitArray,
    UnitSpaceVector,
)


def remap_tensor(
    src: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    *,
    interpolation: int | InterpMethod = 1,
    border_config: BorderConfig | None = None,
    num_threads: int = -1,
) -> np.ndarray:
    """Remap an image using unit-space coordinates via the Cython kernel.

    Args:
        src: Input array, shape (H, W) or (H, W, C).
        map_x: X-coordinate map, shape (H, W), values in [0, 1].
        map_y: Y-coordinate map, shape (H, W), values in [0, 1].
        interpolation: Interpolation mode.
            int: 0=nearest, 1=bilinear, 2=cubic, 3=lanczos3, 4=lanczos4.
            InterpMethod enum also accepted.
        border_config: BorderConfig instance (or None for clamp default).
        num_threads: Thread count (-1 = auto).

    Returns:
        Remapped array, same shape and dtype as src.
    """
    from ..cython._remap_kernel import remap_tensor as _remap_core

    # Cython wrapper expects int; convert InterpMethod enum if needed
    if isinstance(interpolation, InterpMethod):
        _ID_MAP = {
            InterpMethod.NEAREST_MANHATTAN: 0,
            InterpMethod.NEAREST_EUCLIDEAN: 0,
            InterpMethod.LINEAR: 1,
            InterpMethod.CUBIC: 2,
            InterpMethod.LANCZOS4: 4,
        }
        interpolation = _ID_MAP.get(interpolation, 1)

    # Normalize border_config in Python (Cython isinstance can be unreliable)
    if border_config is not None and isinstance(border_config, BorderConfig):
        bc = border_config
    elif border_config is not None:
        bc = BorderConfig.from_dict(border_config)  # type: ignore[unreachable]
    else:
        bc = None

    # Handle array-valued constant_value: convert to ARRAY mode
    if bc is not None and bc.mode == BorderMode.CONSTANT and isinstance(bc.constant_value, np.ndarray):
        bc = _convert_constant_value_to_array(bc, src)

    return cast(np.ndarray, _remap_core(
        src, map_x, map_y,
        interpolation=interpolation,
        border_config=bc,
        num_threads=num_threads,
    ))


def remap_tensor_1d(
    src: np.ndarray,
    map_x: np.ndarray,
    *,
    interpolation: int | InterpMethod = 1,
    border_config: BorderConfig | None = None,
    num_threads: int = -1,
) -> np.ndarray:
    """Remap a 1-D signal using unit-space coordinates via the Cython kernel.

    Args:
        src: Input array, shape (N,) or (N, C).
        map_x: Coordinate map, shape (N,), values in [0, 1].
        interpolation: Interpolation mode.
            int: 0=nearest, 1=linear, 2=cubic, 3=lanczos3, 4=lanczos4.
            InterpMethod enum also accepted.
        border_config: BorderConfig instance (or None for clamp default).
        num_threads: Thread count (-1 = auto).

    Returns:
        Remapped array, same shape and dtype as src.
    """
    from ..cython._remap_kernel_1d import remap_tensor_1d as _remap_1d_core

    if isinstance(interpolation, InterpMethod):
        _ID_MAP = {
            InterpMethod.NEAREST_MANHATTAN: 0,
            InterpMethod.NEAREST_EUCLIDEAN: 0,
            InterpMethod.LINEAR: 1,
            InterpMethod.CUBIC: 2,
            InterpMethod.LANCZOS4: 4,
        }
        interpolation = _ID_MAP.get(interpolation, 1)

    if border_config is not None and isinstance(border_config, BorderConfig):
        bc = border_config
    elif border_config is not None:
        bc = BorderConfig.from_dict(border_config)  # type: ignore[unreachable]
    else:
        bc = None

    if bc is not None and bc.mode == BorderMode.CONSTANT and isinstance(bc.constant_value, np.ndarray):
        bc = _convert_constant_value_to_array_1d(bc, src)

    return cast(np.ndarray, _remap_1d_core(
        src, map_x,
        interpolation=interpolation,
        border_config=bc,
        num_threads=num_threads,
    ))


def _nearest_resize_indices(src_size: int, dst_size: int) -> np.ndarray:
    if src_size <= 0:
        raise ValueError(f"source dimensions must be positive, got {src_size}")
    if dst_size <= 0:
        raise ValueError(f"target dimensions must be positive, got {dst_size}")
    if src_size == 1 or dst_size == 1:
        return np.zeros(dst_size, dtype=np.intp)
    positions = np.linspace(0.0, src_size - 1, dst_size, dtype=np.float64)
    return np.clip(np.floor(positions + 0.5), 0, src_size - 1).astype(np.intp)


def _resize_nearest_numpy(image: np.ndarray, width: int, height: int) -> np.ndarray:
    y_idx = _nearest_resize_indices(image.shape[0], height)
    x_idx = _nearest_resize_indices(image.shape[1], width)
    if image.ndim == 2:
        return np.ascontiguousarray(image[y_idx[:, None], x_idx])
    return np.ascontiguousarray(image[y_idx[:, None], x_idx, :])


def resize_nearest(
    image: np.ndarray,
    width: int,
    height: int,
    *,
    preserve_dtype: bool = True,
) -> np.ndarray:
    """Resize a 2-D or 3-D array with nearest-neighbor sampling.

    This helper is intended primarily for resizing sampled unit fields/maps for
    preview remap workflows. Coordinates are sampled with endpoint alignment,
    matching UnitField's unit-space convention.
    """
    arr = np.asarray(image)
    if arr.ndim not in (2, 3):
        raise ValueError(f"image must be 2-D or 3-D, got {arr.ndim}-D")
    if arr.shape[0] <= 0 or arr.shape[1] <= 0:
        raise ValueError(f"image spatial dimensions must be positive, got {arr.shape[:2]}")
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")
    if height <= 0:
        raise ValueError(f"height must be positive, got {height}")

    result: np.ndarray | None = None
    if arr.dtype.kind in "fiu" and (arr.dtype.kind != "u" or arr.dtype.itemsize <= 4):
        try:
            from ..cython._resize_nearest import resize_nearest as _resize_core
        except ImportError:
            _resize_core = None
        if _resize_core is not None:
            result = cast(np.ndarray, _resize_core(arr, width, height))

    if result is None:
        result = _resize_nearest_numpy(arr, width, height)

    if not preserve_dtype:
        result = result.astype(DEFAULT_DTYPE, copy=False)
    return np.ascontiguousarray(result)


def resize_unit_field_nearest(
    field: np.ndarray,
    width: int,
    height: int,
    *,
    preserve_dtype: bool = True,
) -> np.ndarray:
    """Resize a sampled unit field/map with nearest-neighbor sampling.

    Args:
        field: Array of shape ``(H, W, C)``, for example ``(H, W, 2)`` maps.
        width: Target width.
        height: Target height.
        preserve_dtype: Preserve input dtype by default. If false, output uses
            UnitField's default dtype.
    """
    arr = np.asarray(field)
    if arr.ndim != 3:
        raise ValueError(f"field must have shape (height, width, channels), got {arr.ndim}-D")
    if arr.shape[2] <= 0:
        raise ValueError("field must have at least one channel")
    return resize_nearest(arr, width, height, preserve_dtype=preserve_dtype)


def sample_callable_1d(
    fn,
    count: int,
    *,
    endpoint: bool = True,
    dtype: type = DEFAULT_DTYPE,
) -> np.ndarray:
    """Sample a callable over unit interval coordinates into an array."""
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    xs: np.ndarray = np.linspace(0.0, 1.0, count, endpoint=endpoint, dtype=dtype)
    values: np.ndarray = np.asarray(fn(xs), dtype=dtype)
    if values.shape == () or values.shape[0] != count:
        values = np.asarray([fn(float(x)) for x in xs], dtype=dtype)
    return values


def remap_envelope(
    values: np.ndarray,
    map_x: np.ndarray,
    *,
    interpolation: int | InterpMethod = 1,
    border_config: BorderConfig | None = None,
    num_threads: int = -1,
) -> np.ndarray:
    """Remap a 1-D sampled envelope using unit-space coordinates."""
    return remap_tensor_1d(
        np.asarray(values),
        np.asarray(map_x, dtype=np.float64),
        interpolation=interpolation,
        border_config=border_config,
        num_threads=num_threads,
    )


def _convert_constant_value_to_array(
    bc: BorderConfig, src: np.ndarray,
) -> BorderConfig:
    """Convert array-valued constant_value to ARRAY mode border_config."""
    arr = bc.constant_value
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"Expected array-valued constant_value, got {type(arr)}")
    H, W = src.shape[0], src.shape[1]
    C = 1 if src.ndim == 2 else src.shape[2]

    if arr.ndim == 0:
        return bc.with_(constant_value=float(arr))
    if arr.ndim == 1:
        if arr.shape[0] != C:
            raise ValueError(
                f"constant_value 1-D length {arr.shape[0]} doesn't match "
                f"src channels {C}"
            )
        border_arr = np.broadcast_to(
            arr.reshape(1, 1, C), (H, W, C)
        ).copy(order='C')
        return BorderConfig(
            mode=BorderMode.ARRAY, array=border_arr,
            feathering_width=bc.feathering_width,
            feathering_x_undershoot_multiplier=bc.feathering_x_undershoot_multiplier,
            feathering_x_overshoot_multiplier=bc.feathering_x_overshoot_multiplier,
            feathering_y_undershoot_multiplier=bc.feathering_y_undershoot_multiplier,
            feathering_y_overshoot_multiplier=bc.feathering_y_overshoot_multiplier,
            feathering_mode=bc.feathering_mode,
            distance_mode=bc.distance_mode,
            preserve_non_feather_dims=bc.preserve_non_feather_dims,
            feather_dims=bc.feather_dims,
        )
    if arr.ndim == 2:
        if arr.shape != (H, W):
            raise ValueError(
                f"constant_value 2-D shape {arr.shape} doesn't match "
                f"src shape ({H}, {W})"
            )
        border_arr = np.broadcast_to(
            arr.reshape(H, W, 1), (H, W, C)
        ).copy(order='C')
        return BorderConfig(
            mode=BorderMode.ARRAY, array=border_arr,
            feathering_width=bc.feathering_width,
            feathering_x_undershoot_multiplier=bc.feathering_x_undershoot_multiplier,
            feathering_x_overshoot_multiplier=bc.feathering_x_overshoot_multiplier,
            feathering_y_undershoot_multiplier=bc.feathering_y_undershoot_multiplier,
            feathering_y_overshoot_multiplier=bc.feathering_y_overshoot_multiplier,
            feathering_mode=bc.feathering_mode,
            distance_mode=bc.distance_mode,
            preserve_non_feather_dims=bc.preserve_non_feather_dims,
            feather_dims=bc.feather_dims,
        )
    if arr.ndim == 3:
        if arr.shape != (H, W, C):
            raise ValueError(
                f"constant_value 3-D shape {arr.shape} doesn't match "
                f"src shape ({H}, {W}, {C})"
            )
        if arr.dtype != np.float64:
            arr = arr.astype(np.float64)
        border_arr = np.ascontiguousarray(arr)
        return BorderConfig(
            mode=BorderMode.ARRAY, array=border_arr,
            feathering_width=bc.feathering_width,
            feathering_x_undershoot_multiplier=bc.feathering_x_undershoot_multiplier,
            feathering_x_overshoot_multiplier=bc.feathering_x_overshoot_multiplier,
            feathering_y_undershoot_multiplier=bc.feathering_y_undershoot_multiplier,
            feathering_y_overshoot_multiplier=bc.feathering_y_overshoot_multiplier,
            feathering_mode=bc.feathering_mode,
            distance_mode=bc.distance_mode,
            preserve_non_feather_dims=bc.preserve_non_feather_dims,
            feather_dims=bc.feather_dims,
        )
    raise ValueError(
        f"constant_value must be 0-D, 1-D, 2-D, or 3-D, got {arr.ndim}-D"
    )


def _convert_constant_value_to_array_1d(
    bc: BorderConfig, src: np.ndarray,
) -> BorderConfig:
    """Convert array-valued constant_value to ARRAY mode (1-D variant)."""
    arr = bc.constant_value
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"Expected array-valued constant_value, got {type(arr)}")
    N = src.shape[0]
    C = 1 if src.ndim == 1 else src.shape[1]

    if arr.ndim == 0:
        return bc.with_(constant_value=float(arr))
    if arr.ndim == 1:
        if arr.shape[0] != C:
            raise ValueError(
                f"constant_value 1-D length {arr.shape[0]} doesn't match "
                f"src channels {C}"
            )
        border_arr = np.broadcast_to(
            arr.reshape(1, C), (N, C)
        ).copy(order='C')
        return BorderConfig(
            mode=BorderMode.ARRAY, array=border_arr,
            feathering_width=bc.feathering_width,
            feathering_x_undershoot_multiplier=bc.feathering_x_undershoot_multiplier,
            feathering_x_overshoot_multiplier=bc.feathering_x_overshoot_multiplier,
            feathering_y_undershoot_multiplier=bc.feathering_y_undershoot_multiplier,
            feathering_y_overshoot_multiplier=bc.feathering_y_overshoot_multiplier,
            feathering_mode=bc.feathering_mode,
            distance_mode=bc.distance_mode,
            preserve_non_feather_dims=bc.preserve_non_feather_dims,
            feather_dims=bc.feather_dims,
        )
    if arr.ndim == 2:
        if arr.shape != (N, C):
            raise ValueError(
                f"constant_value 2-D shape {arr.shape} doesn't match "
                f"src shape ({N}, {C})"
            )
        if arr.dtype != np.float64:
            arr = arr.astype(np.float64)
        border_arr = np.ascontiguousarray(arr)
        return BorderConfig(
            mode=BorderMode.ARRAY, array=border_arr,
            feathering_width=bc.feathering_width,
            feathering_x_undershoot_multiplier=bc.feathering_x_undershoot_multiplier,
            feathering_x_overshoot_multiplier=bc.feathering_x_overshoot_multiplier,
            feathering_y_undershoot_multiplier=bc.feathering_y_undershoot_multiplier,
            feathering_y_overshoot_multiplier=bc.feathering_y_overshoot_multiplier,
            feathering_mode=bc.feathering_mode,
            distance_mode=bc.distance_mode,
            preserve_non_feather_dims=bc.preserve_non_feather_dims,
            feather_dims=bc.feather_dims,
        )
    raise ValueError(
        f"constant_value must be 0-D, 1-D, or 2-D, got {arr.ndim}-D"
    )


class UnitNdimField(ABC):
    """
    Abstract base class for N-dimensional unit fields.

    Unit fields map coordinates in unit space [0, 1]^N to vectors in unit space.
    Concrete implementations provide interpolation and efficient querying.

    Note:
        All coordinate inputs are expected to be in the range [0, 1]. Coordinates
        containing inf or NaN are NOT supported and will produce undefined behavior.
        This design choice avoids unnecessary overhead for the common case of
        normalized coordinate spaces.
    """

    @abstractmethod
    def get_value(self, coords: Coordinate) -> UnitSpaceVector:
        """
        Map unit-space coordinates to unit-space vectors.

        Args:
            coords: Unit-space coordinates tuple

        Returns:
            Unit-space vector at the given coordinates
        """
        pass

    @abstractmethod
    def get_values(self, coords_array: np.ndarray) -> np.ndarray:
        """
        Map multiple unit-space coordinates to unit-space vectors.

        Args:
            coords_array: Array of shape (..., N) where N is field dimension

        Returns:
            Array of unit-space vectors
        """
        pass

    @property
    @abstractmethod
    def ndim(self) -> int:
        """Get the dimensionality of the field."""
        pass

    @property
    @abstractmethod
    def spatial_shape(self) -> tuple[int, ...]:
        """Get the spatial shape of the underlying data."""
        pass

class MappedUnitField(UnitNdimField):
    """
    N-dimensional unit field with interpolation.

    Parameters
    ----------
    data : UnitArray
        N+1 dimensional array of shape (*spatial_dims, N),
        where N is the dimensionality of the coordinate space.
    interp_method : InterpMethod, optional
        Interpolation strategy used to sample the field.
        Default: InterpMethod.NEAREST_MANHATTAN
    cache_size : int, optional
        Size of LRU cache for single coordinate queries.
        Set to 0 or None to disable caching.
        Default: 128
    """

    def __init__(
        self,
        data: UnitArray,
        interp_method: InterpMethod = InterpMethod.NEAREST_MANHATTAN,
        cache_size: int | None = DEFAULT_CACHE_SIZE
    ):
        # Input validation
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Data must be numpy array, got {type(data)}")

        if data.size == 0:
            raise ValueError("Data array must not be empty")

        if data.ndim < 1:
            raise ValueError("Data must have at least 1 dimension")

        if not isinstance(interp_method, InterpMethod):
            raise TypeError(
                f"interp_method must be InterpMethod, got {type(interp_method)}"
            )

        if cache_size is not None and cache_size < 0:
            raise ValueError("cache_size must be non-negative")

        self._data = data
        self._interp_method = interp_method
        self._ndim: int = data.shape[-1]
        self._spatial_shape = data.shape[:-1]
        self._cache_size = cache_size

        # Validate interpolation method
        if interp_method not in np_interp_dict:
            raise ValueError(
                f"Unsupported interpolation method: {interp_method}. "
                f"Supported methods: {list(np_interp_dict.keys())}"
            )

    @property
    def data(self) -> UnitArray:
        """Get the underlying data array."""
        return self._data

    @property
    def interp_method(self) -> InterpMethod:
        """Get the interpolation method."""
        return self._interp_method

    @property
    def ndim(self) -> int:
        """Get the dimensionality of the field."""
        return self._ndim

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        """Get the spatial shape of the underlying data."""
        return self._spatial_shape

    @property
    def cache_size(self) -> int | None:
        """Get the cache size for single coordinate queries."""
        return self._cache_size

    def get_value(self, coords: Coordinate) -> tuple[float, ...]:
        """
        Map unit-space coordinates to unit-space vectors.

        Args:
            coords: Unit-space coordinates tuple in range [0, 1]

        Returns:
            Unit-space vector at the given coordinates

        Note:
            Coordinates containing inf or NaN are NOT supported and will produce
            undefined behavior. Coordinates outside [0, 1] are clipped to the
            valid range. This design avoids overhead for the common case of
            normalized coordinates.
        """
        # Validate coordinate dimensions
        expected_dim = len(self.spatial_shape)
        if len(coords) != expected_dim:
            raise ValueError(
                f"Expected {expected_dim} coordinates, got {len(coords)}."
            )

        # Convert to numpy array
        coords_array: np.ndarray = np.array(coords, dtype=DEFAULT_DTYPE).reshape(1, -1)

        # Get interpolator
        interp_func = np_interp_dict[self._interp_method]

        # Interpolate
        result = interp_func(coords_array, self._spatial_shape, self._data)

        # Return as tuple
        return tuple(result[0])

    def get_values(self, coords_array: np.ndarray) -> np.ndarray:
        """
        Map multiple unit-space coordinates to unit-space vectors.

        Args:
            coords_array: Array of shape (..., N) where N is field dimension.
                         Coordinates should be in range [0, 1].

        Returns:
            Array of unit-space vectors

        Raises:
            ValueError: If coordinate dimensions don't match field dimension

        Note:
            Coordinates containing inf or NaN are NOT supported and will produce
            undefined behavior. Coordinates outside [0, 1] are clipped to the
            valid range. This design avoids overhead for the common case of
            normalized coordinates.
        """
        if coords_array.shape[-1] != self.ndim:
            raise ValueError(
                f"Expected coordinates with {self.ndim} dimensions, "
                f"got {coords_array.shape[-1]}"
            )

        interp_func = np_interp_dict[self._interp_method]
        return cast(np.ndarray, interp_func(coords_array, self._spatial_shape, self._data))

    def with_interp_method(self, method: InterpMethod) -> MappedUnitField:
        """
        Create a copy with different interpolation method.

        Args:
            method: New interpolation method

        Returns:
            New MappedUnitField instance
        """
        return MappedUnitField(
            data=self._data,
            interp_method=method,
            cache_size=self._cache_size
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"shape={self._data.shape}, "
            f"interp={self._interp_method.name}, "
            f"cache={self._cache_size})"
        )


class UnitMappedEndomorphism(MappedUnitField):
    """
    Unit field representing an endomorphism (same input/output dimension).

    Parameters
    ----------
    data : UnitArray
        N+1 dimensional array of shape (*spatial_dims, N),
        where N is the dimensionality of the coordinate space.
    interp_method : InterpMethod, optional
        Interpolation strategy used to sample the field.
        Default: InterpMethod.NEAREST_MANHATTAN
    cache_size : int, optional
        Size of LRU cache for single coordinate queries.
        Default: 128
    """

    def __init__(
        self,
        data: UnitArray,
        interp_method: InterpMethod = InterpMethod.NEAREST_MANHATTAN,
        cache_size: int | None = DEFAULT_CACHE_SIZE
    ):
        super().__init__(data, interp_method, cache_size)

        # Validate endomorphism property
        if self.ndim != len(self.spatial_shape):
            raise ValueError(
                f"Data shape {data.shape} does not represent an endomorphism. "
                f"Expected last dimension size {len(self.spatial_shape)}, "
                f"got {self.ndim}."
            )


# UnitField\unitfield\core\unitfield.py

class Unit2DMappedEndomorphism(UnitMappedEndomorphism):
    """
    2D unit field endomorphism with optimized OpenCV backend.

    Parameters
    ----------
    data : UnitArray
        3-dimensional array of shape (H, W, 2)
    interp_method : InterpMethod, optional
        Interpolation strategy used to sample the field.
        Default: InterpMethod.NEAREST_MANHATTAN
    cache_size : int, optional
        Size of LRU cache for single coordinate queries.
        Default: 128
    """

    def __init__(
        self,
        data: UnitArray,
        interp_method: InterpMethod = InterpMethod.NEAREST_MANHATTAN,
        cache_size: int | None = DEFAULT_CACHE_SIZE
    ):
        # First validate 2D specific properties
        if len(data.shape) != 3 or data.shape[-1] != 2:
            raise ValueError(
                f"Data shape {data.shape} does not represent a 2D endomorphism. "
                f"Expected shape (H, W, 2), got {data.shape}"
            )

        super().__init__(data, interp_method, cache_size)

        # Double-check after parent initialization
        if self.ndim != 2 or len(self.spatial_shape) != 2:
            raise ValueError(
                f"Data shape {data.shape} does not represent a 2D endomorphism. "
                f"Expected shape (H, W, 2), got {data.shape}"
            )

        # Set up caching — wrap the parent's get_value
        self._cache_size = cache_size
        if cache_size and cache_size > 0:
            self.get_value = self._create_cached_get_value()  # type: ignore[method-assign]

    def _create_cached_get_value(self):
        """Create a cached version of get_value using the numpy backend."""
        parent_get_value = super().get_value

        @lru_cache(maxsize=self._cache_size)
        def cached_get_value(coords: tuple[float, float]) -> tuple[float, float]:
            return cast(tuple[float, float], parent_get_value(coords))

        return cached_get_value

    def rasterize_mapping(
        self,
        width: int,
        height: int,
        *,
        dtype: type = DEFAULT_DTYPE
    ) -> np.ndarray:
        """
        Rasterize the unit endomorphism into a pixel-space mapping.

        Args:
            width: Output width in pixels
            height: Output height in pixels
            dtype: Data type for output mapping

        Returns:
            Pixel-space mapping of shape (H, W, 2)

        Raises:
            ValueError: If width or height are invalid
        """
        if width <= 0:
            raise ValueError(f"Width must be positive, got {width}")
        if height <= 0:
            raise ValueError(f"Height must be positive, got {height}")

        # Build unit grid efficiently using broadcasting
        xs: np.ndarray
        ys: np.ndarray
        xs, ys = np.meshgrid(
            np.linspace(0, 1, width, dtype=dtype),
            np.linspace(0, 1, height, dtype=dtype),
            indexing='xy'
        )
        unit_grid = np.stack([xs, ys], axis=-1)

        # Evaluate the field using vectorized operation
        unit_mapping = self.get_values(unit_grid)

        # Convert unit → pixel
        mapping = unit_mapping.copy()
        if width > 1:
            mapping[..., 0] *= (width - 1)
        if height > 1:
            mapping[..., 1] *= (height - 1)

        return cast(np.ndarray, mapping.astype(np.float32))

    def remap(
        self,
        data: np.ndarray,
        *,
        interpolation: int | InterpMethod = 1,
        border_config: BorderConfig | None = None,
    ) -> np.ndarray:
        """
        Remap any (H, W, J) array using this unit endomorphism.

        Args:
            data: Input array of shape (H, W, ...)
            interpolation: Interpolation mode (0=nearest, 1=bilinear, ...)
            border_config: BorderConfig instance (or None for clamp default)

        Returns:
            Remapped array
        """
        height, width = data.shape[:2]
        # Generate unit-space mapping grid
        # Generate unit-space mapping grid in (y, x) order
        # (the numpy backend expects spatial dims first)
        ys, xs = np.meshgrid(
            np.linspace(0, 1, height), np.linspace(0, 1, width), indexing='ij'
        )
        unit_grid = np.stack([ys, xs], axis=-1)
        # Evaluate the field to get the warped coordinate maps
        # get_values returns (x, y) since data stores (x, y)
        warped = self.get_values(unit_grid)
        map_x = warped[..., 0]
        map_y = warped[..., 1]

        return remap_tensor(
            data, map_x, map_y,
            interpolation=interpolation,
            border_config=border_config,
        )

    def compose(
        self,
        other: Unit2DMappedEndomorphism,
        interp_method: InterpMethod | None = None
    ) -> Unit2DMappedEndomorphism:
        """
        Compose this endomorphism with another.

        Args:
            other: Another 2D endomorphism to compose with
            interp_method: Interpolation method for composition

        Returns:
            New composed endomorphism
        """
        if not isinstance(other, Unit2DMappedEndomorphism):
            raise TypeError(
                f"Can only compose with Unit2DMappedEndomorphism, got {type(other)}"
            )

        if interp_method is None:
            interp_method = self.interp_method

        # Sample other field at this field's coordinates in (y, x) order
        height, width = self.spatial_shape
        ys: np.ndarray
        xs: np.ndarray
        ys, xs = np.meshgrid(
            np.linspace(0, 1, height, dtype=DEFAULT_DTYPE),
            np.linspace(0, 1, width, dtype=DEFAULT_DTYPE),
            indexing='ij'
        )
        coords = np.stack([ys, xs], axis=-1)

        # Apply self then other
        mapped_by_self = self.get_values(coords)  # returns (x, y)
        # Swap to (y, x) order for the next get_values call
        mapped_by_both = other.get_values(mapped_by_self[..., ::-1])

        return Unit2DMappedEndomorphism(
            data=mapped_by_both,
            interp_method=interp_method,
            cache_size=self.cache_size
        )


class Unit1DMappedEndomorphism(UnitMappedEndomorphism):
    """1D unit field endomorphism mapping [0,1] → [0,1].

    Parameters
    ----------
    data : UnitArray
        2-dimensional array of shape (N, 1)
    interp_method : InterpMethod, optional
        Interpolation strategy used to sample the field.
        Default: InterpMethod.LINEAR
    cache_size : int, optional
        Size of LRU cache for single coordinate queries.
        Default: 128
    """

    def __init__(
        self,
        data: UnitArray,
        interp_method: InterpMethod = InterpMethod.LINEAR,
        cache_size: int | None = DEFAULT_CACHE_SIZE
    ):
        if len(data.shape) != 2 or data.shape[-1] != 1:
            raise ValueError(
                f"Data shape {data.shape} does not represent a 1D endomorphism. "
                f"Expected shape (N, 1), got {data.shape}"
            )

        super().__init__(data, interp_method, cache_size)

        if self.ndim != 1 or len(self.spatial_shape) != 1:
            raise ValueError(
                f"Data shape {data.shape} does not represent a 1D endomorphism. "
                f"Expected shape (N, 1), got {data.shape}"
            )

        self._cache_size = cache_size
        if cache_size and cache_size > 0:
            self.get_value = self._create_cached_get_value()  # type: ignore[method-assign]

    @classmethod
    def from_envelope(
        cls,
        values: np.ndarray,
        *,
        interp_method: InterpMethod = InterpMethod.LINEAR,
        cache_size: int | None = DEFAULT_CACHE_SIZE,
    ) -> Unit1DMappedEndomorphism:
        """Build a 1-D endomorphism from sampled ``u -> u'`` values."""
        data: np.ndarray = np.asarray(values, dtype=DEFAULT_DTYPE)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return cls(data, interp_method=interp_method, cache_size=cache_size)

    def as_callable(self):
        """Return a callable that evaluates this 1-D endomorphism."""
        def evaluate(u):
            arr: np.ndarray = np.asarray(u, dtype=DEFAULT_DTYPE)
            if arr.ndim == 0:
                return self.get_value((float(arr),))[0]
            values = self.get_values(arr.reshape(-1, 1))[:, 0]
            return values.reshape(arr.shape)

        return evaluate

    def _create_cached_get_value(self):
        parent_get_value = super().get_value

        @lru_cache(maxsize=self._cache_size)
        def cached_get_value(coords: tuple[float, ...]) -> tuple[float, ...]:
            return parent_get_value(coords)

        return cached_get_value

    def remap(
        self,
        data: np.ndarray,
        *,
        interpolation: int | InterpMethod = 1,
        border_config: BorderConfig | None = None,
    ) -> np.ndarray:
        """Remap any (N,) or (N, C) array using this unit endomorphism.

        Args:
            data: Input array of shape (N,) or (N, ...)
            interpolation: Interpolation mode (0=nearest, 1=linear, ...)
            border_config: BorderConfig instance (or None for clamp default)

        Returns:
            Remapped array
        """
        n = data.shape[0]
        xs = np.linspace(0, 1, n, dtype=np.float64)
        warped = self.get_values(xs[:, np.newaxis])
        map_x = warped[:, 0]

        return remap_tensor_1d(
            data, map_x,
            interpolation=interpolation,
            border_config=border_config,
        )


def _coerce_unit_endomorphism(
    value,
    *,
    interp_method: InterpMethod,
    cache_size: int | None,
) -> UnitMappedEndomorphism:
    if isinstance(value, UnitMappedEndomorphism):
        return value

    data: np.ndarray = np.asarray(value, dtype=DEFAULT_DTYPE)
    if data.ndim == 1:
        return Unit1DMappedEndomorphism.from_envelope(
            data,
            interp_method=interp_method,
            cache_size=cache_size,
        )

    if data.ndim >= 2 and data.shape[-1] == data.ndim - 1:
        if data.shape[-1] == 1:
            return Unit1DMappedEndomorphism(
                data,
                interp_method=interp_method,
                cache_size=cache_size,
            )
        if data.shape[-1] == 2 and data.ndim == 3:
            return Unit2DMappedEndomorphism(
                data,
                interp_method=interp_method,
                cache_size=cache_size,
            )
        return UnitMappedEndomorphism(
            data,
            interp_method=interp_method,
            cache_size=cache_size,
        )

    raise ValueError(
        f"Data shape {data.shape} does not represent a unit endomorphism. "
        "Use a 1-D envelope or an array shaped (*spatial_dims, ndim)."
    )


def _unit_coordinate_grid(shape: tuple[int, ...], dtype: type = DEFAULT_DTYPE) -> np.ndarray:
    axes: list[np.ndarray] = [np.linspace(0.0, 1.0, size, dtype=dtype) for size in shape]
    if len(shape) == 1:
        return axes[0].reshape(-1, 1)
    return np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)


def _endomorphism_from_data(
    data: np.ndarray,
    *,
    interp_method: InterpMethod,
    cache_size: int | None,
) -> UnitMappedEndomorphism:
    if data.shape[-1] == 1:
        return Unit1DMappedEndomorphism(
            data,
            interp_method=interp_method,
            cache_size=cache_size,
        )
    if data.shape[-1] == 2 and data.ndim == 3:
        return Unit2DMappedEndomorphism(
            data,
            interp_method=interp_method,
            cache_size=cache_size,
        )
    return UnitMappedEndomorphism(
        data,
        interp_method=interp_method,
        cache_size=cache_size,
    )


class UnitEndomorphismInterpolator:
    """Interpolate between two mapped unit endomorphisms."""

    def __init__(
        self,
        start,
        end,
        *,
        output_shape: tuple[int, ...] | None = None,
        interp_method: InterpMethod = InterpMethod.LINEAR,
        cache_size: int | None = DEFAULT_CACHE_SIZE,
    ):
        self._start = _coerce_unit_endomorphism(
            start,
            interp_method=interp_method,
            cache_size=cache_size,
        )
        self._end = _coerce_unit_endomorphism(
            end,
            interp_method=interp_method,
            cache_size=cache_size,
        )

        if self._start.ndim != self._end.ndim:
            raise ValueError(
                f"Cannot interpolate {self._start.ndim}D and {self._end.ndim}D "
                "endomorphisms."
            )

        if output_shape is None:
            output_shape = self._start.spatial_shape
        self._output_shape = self._validate_output_shape(output_shape)
        self._interp_method = interp_method
        self._cache_size = cache_size

    @property
    def start(self) -> UnitMappedEndomorphism:
        return self._start

    @property
    def end(self) -> UnitMappedEndomorphism:
        return self._end

    @property
    def ndim(self) -> int:
        return self._start.ndim

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self._output_shape

    def interpolate(
        self,
        t: float,
        *,
        output_shape: tuple[int, ...] | None = None,
        interp_method: InterpMethod | None = None,
        cache_size: int | None = None,
        dtype: type = DEFAULT_DTYPE,
    ) -> UnitMappedEndomorphism:
        """Return the mapped endomorphism at interpolation parameter ``t``."""
        alpha = float(t)
        if not np.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
            raise ValueError(f"t must be finite and in [0, 1], got {t}")

        shape = self._output_shape if output_shape is None else self._validate_output_shape(output_shape)
        result_interp_method = self._interp_method if interp_method is None else interp_method
        result_cache_size = self._cache_size if cache_size is None else cache_size

        coords = _unit_coordinate_grid(shape, dtype=dtype)
        start_values = self._start.get_values(coords)
        end_values = self._end.get_values(coords)
        data: np.ndarray = ((1.0 - alpha) * start_values + alpha * end_values).astype(
            dtype,
            copy=False,
        )

        return _endomorphism_from_data(
            data,
            interp_method=result_interp_method,
            cache_size=result_cache_size,
        )

    def _validate_output_shape(self, output_shape: tuple[int, ...]) -> tuple[int, ...]:
        shape = tuple(int(size) for size in output_shape)
        if len(shape) != self.ndim:
            raise ValueError(
                f"Expected output shape with {self.ndim} dimensions, got {len(shape)}."
            )
        if any(size <= 0 for size in shape):
            raise ValueError(f"Output shape values must be positive, got {shape}.")
        return shape


class Unit1DEndoInterpolator(UnitEndomorphismInterpolator):
    """Interpolate between two 1-D unit endomorphisms."""

    def __init__(self, start, end, **kwargs):
        super().__init__(start, end, **kwargs)
        if self.ndim != 1:
            raise ValueError(f"Expected 1-D endomorphisms, got {self.ndim}D.")


class Unit2DEndoInterpolator(UnitEndomorphismInterpolator):
    """Interpolate between two 2-D unit endomorphisms."""

    def __init__(self, start, end, **kwargs):
        super().__init__(start, end, **kwargs)
        if self.ndim != 2:
            raise ValueError(f"Expected 2-D endomorphisms, got {self.ndim}D.")


Unit1dEndoInterpolator = Unit1DEndoInterpolator
Unit2dEndoInterpolator = Unit2DEndoInterpolator
