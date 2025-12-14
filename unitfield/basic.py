# basic.py
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Literal, Tuple, Dict, Optional
from boundednumbers import UnitFloat
from .distances import nearest_manhattan, n_linear_vectorized as n_linear, nearest_euclidean, VectorReturnType
from .types import UnitArray

class InterpMethod(Enum):
    NEAREST_MANHATTAN = auto()
    LINEAR = auto()
    NEAREST_EUCLIDEAN = auto()
    EUCLIDEAN = NEAREST_EUCLIDEAN  # Alias for backward compatibility

interp_method_dict: Dict[InterpMethod, callable] = {
    InterpMethod.NEAREST_MANHATTAN: nearest_manhattan,
    InterpMethod.LINEAR: n_linear,
    InterpMethod.NEAREST_EUCLIDEAN: nearest_euclidean,
}

class UnitNdimField(ABC):
    @abstractmethod
    def get_value(
        self,
    ) -> VectorReturnType:
        """Map unit-space coordinates to unit-space vectors."""
        ...


class MappedNdimField(UnitNdimField):
    def __init__(
        self,
        data: UnitArray,
        interp_method: InterpMethod = InterpMethod.NEAREST_MANHATTAN,
        cache_size: Optional[int] = 128
    ):
        """
        Parameters
        ----------
        data :
            N+1 dimensional array of shape (*spatial_dims, N),
            where N is the dimensionality of the coordinate space.
        interp_method :
            Interpolation strategy used to sample the field.
        """
        self.data = data
        self.interp_method = interp_method

        self.ndim = data.shape[-1]
        self.spatial_shape = data.shape[:-1]
        # ... existing initialization ...
        self.cache_size = cache_size
        if cache_size:
            from functools import lru_cache
            self._cached_get_value = lru_cache(maxsize=cache_size)(self._get_value_uncached)
        else:
            self._cached_get_value = self._get_value_uncached


    def get_value(
        self,
        coords: Tuple[UnitFloat, ...],
        return_type: Literal["array", "tuple", "list"] = "tuple"
    ) -> VectorReturnType:
        return self._cached_get_value(coords, return_type=return_type)
    
    def _get_value_uncached(self, coords: Tuple[UnitFloat, ...], return_type: Literal["array", "tuple", "list"] = "tuple") -> VectorReturnType:
        # Move the existing get_value logic here
        if len(coords) != len(self.spatial_shape):
            raise ValueError(
                f"Expected {len(self.spatial_shape)} coordinates, got {len(coords)}."
            )
        
        if self.interp_method not in interp_method_dict:
            raise ValueError(
                f"Unsupported interpolation method: {self.interp_method}."
            )
        
        interp_func = interp_method_dict[self.interp_method]
        return interp_func(coords, self.spatial_shape, self.data, return_type=return_type)


class UnitMappedEndomorphism(MappedNdimField):
    def __init__(
        self,
        data: UnitArray,
        interp_method: InterpMethod = InterpMethod.NEAREST_MANHATTAN,
        cache_size: Optional[int] = 128
    ):
        """
        Maps unit-space coordinates to unit-space vectors of the same dimensionality.
        Parameters
        ----------
        data :
            N+1 dimensional array of shape (*spatial_dims, N),
            where N is the dimensionality of the coordinate space.
        interp_method :
            Interpolation strategy used to sample the field.
        """
        super().__init__(data, interp_method, cache_size)
        if self.ndim != len(self.spatial_shape):
            raise ValueError(
                f"Data shape {data.shape} does not represent an endomorphism. "
                f"Expected last dimension size {len(self.spatial_shape)}, got {self.ndim}."
            )
        
