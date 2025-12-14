from .basic import UnitNdimField, InterpMethod, MappedNdimField
import numpy as np
from typing import Tuple, Optional, Iterable, Literal
from abc import ABC, abstractmethod
from .distances import VectorReturnType
from boundednumbers import UnitFloat, RealNumber



class Field(ABC):
    maxima: Iterable
    minima: Iterable
    field: UnitNdimField
    interp_method: InterpMethod
    return_type: Literal["array", "tuple", "list"]
    @abstractmethod
    def get_value(
            self,
            coords: Tuple[UnitFloat, ...],
            interp_method: Optional[InterpMethod] = None,
            return_type: Optional[Literal["array", "tuple", "list"]] = None
        ) -> VectorReturnType:
        """Get the modulated value based on input unit-space coordinates."""
        interp_method = interp_method or self.interp_method
        return_type = return_type or self.return_type
        unit_values = self.field.get_value(coords, return_type=return_type)
        scaled_values = tuple(
            self.minima[i] + unit_values[i] * (self.maxima[i] - self.minima[i])
            for i in range(len(unit_values))
        )
        return scaled_values

class MappedField(Field):
    def __init__(
            self,
            maxima: np.ndarray,
            minima: np.ndarray,
            field: MappedNdimField,
            interp_method: InterpMethod = InterpMethod.NEAREST_MANHATTAN,
            return_type: Literal["array", "tuple", "list"] = "tuple"
    ):
        if field.ndim != len(maxima) or field.ndim != len(minima):
            raise ValueError(
                "Dimensionality of field does not match length of maxima/minima."
            )
        self.maxima = np.array(maxima)
        self.minima = np.array(minima)
        self.field = field
        self.interp_method = interp_method
        self.return_type = return_type
        

    