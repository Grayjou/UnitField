# UnitField\unitfield\core\unitfield.py
"""
Main unit field transformation classes.
"""

from __future__ import annotations
from functools import lru_cache
from typing import Tuple, List, Union, Optional, Dict, Any, Callable
from abc import ABC, abstractmethod

import numpy as np

from .enums import InterpMethod, BorderMode
from .border_config import BorderConfig
from .types import (
    UnitArray, UnitSpaceVector, Coordinate, 
    ImageShape, DEFAULT_CACHE_SIZE, DEFAULT_DTYPE
)
from ..interpolation import (
    np_interp_dict,
)


def remap_tensor(
    src: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    *,
    interpolation: Union[int, InterpMethod] = 1,
    border_config: Optional[BorderConfig] = None,
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

    return _remap_core(
        src, map_x, map_y,
        interpolation=interpolation,
        border_config=border_config,
        num_threads=num_threads,
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
    def spatial_shape(self) -> Tuple[int, ...]:
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
        cache_size: Optional[int] = DEFAULT_CACHE_SIZE
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
        self._ndim = data.shape[-1]
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
    def spatial_shape(self) -> Tuple[int, ...]:
        """Get the spatial shape of the underlying data."""
        return self._spatial_shape
    
    @property
    def cache_size(self) -> Optional[int]:
        """Get the cache size for single coordinate queries."""
        return self._cache_size
    
    def get_value(self, coords: Coordinate) -> Tuple[float, ...]:
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
        coords_array = np.array(coords, dtype=DEFAULT_DTYPE).reshape(1, -1)
        
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
        return interp_func(coords_array, self._spatial_shape, self._data)
    
    def with_interp_method(self, method: InterpMethod) -> 'MappedUnitField':
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
        cache_size: Optional[int] = DEFAULT_CACHE_SIZE
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
        cache_size: Optional[int] = DEFAULT_CACHE_SIZE
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
            self.get_value = self._create_cached_get_value()
    
    def _create_cached_get_value(self):
        """Create a cached version of get_value using the numpy backend."""
        from functools import lru_cache
        parent_get_value = super(Unit2DMappedEndomorphism, self).get_value
        
        @lru_cache(maxsize=self._cache_size)
        def cached_get_value(coords: Tuple[float, float]) -> Tuple[float, float]:
            return parent_get_value(coords)
        
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
        
        return mapping.astype(np.float32)
    
    def remap(
        self,
        data: np.ndarray,
        *,
        interpolation: Union[int, InterpMethod] = 1,
        border_config: Optional[BorderConfig] = None,
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
        ys, xs = np.meshgrid(
            np.linspace(0, 1, height), np.linspace(0, 1, width), indexing='ij'
        )
        unit_grid = np.stack([xs, ys], axis=-1)
        # Evaluate the field to get the warped coordinate maps
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
        other: 'Unit2DMappedEndomorphism',
        interp_method: Optional[InterpMethod] = None
    ) -> 'Unit2DMappedEndomorphism':
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
        
        # Sample other field at this field's coordinates
        height, width = self.spatial_shape
        xs, ys = np.meshgrid(
            np.linspace(0, 1, width, dtype=DEFAULT_DTYPE),
            np.linspace(0, 1, height, dtype=DEFAULT_DTYPE),
            indexing='xy'
        )
        coords = np.stack([xs, ys], axis=-1)
        
        # Apply self then other
        mapped_by_self = self.get_values(coords)
        mapped_by_both = other.get_values(mapped_by_self)
        
        return Unit2DMappedEndomorphism(
            data=mapped_by_both,
            interp_method=interp_method,
            cache_size=self.cache_size
        )