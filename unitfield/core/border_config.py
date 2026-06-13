"""Border configuration dataclass for UnitField coordinate remapping."""

from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from typing import Optional, Union, Any

import numpy as np

from .enums import BorderMode


@dataclass(frozen=True)
class BorderConfig:
    """Border-handling configuration for UnitField remap.

    Frozen so a single config can be shared safely between calls.
    Use :meth:`with_` or :func:`dataclasses.replace` to derive modified copies.

    Attributes:
        mode: How out-of-bounds coordinates are handled.
        constant_value: Value for CONSTANT mode. Scalar or per-channel array.
            ``None`` means 0.0 downstream.
        array: Optional 2-D / 3-D background array for ARRAY mode.
        feathering_width: Smooth-blend band width at borders (0.0 = hard edge).
        feathering_x_multiplier: Per-axis asymmetric feather multiplier (x).
        feathering_y_multiplier: Per-axis asymmetric feather multiplier (y).
    """

    mode: BorderMode = BorderMode.CLAMP
    constant_value: Optional[Union[float, np.ndarray]] = None
    array: Optional[np.ndarray] = None
    feathering_width: float = 0.0
    feathering_x_multiplier: float = 1.0
    feathering_y_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.feathering_width < 0.0:
            raise ValueError("feathering_width must be non-negative")
        if self.feathering_x_multiplier < 0.0 or self.feathering_y_multiplier < 0.0:
            raise ValueError("feathering multipliers must be non-negative")
        if self.mode == BorderMode.ARRAY and self.array is None:
            raise ValueError("ARRAY mode requires a non-None array")

    @classmethod
    def clamp(cls) -> BorderConfig:
        """CLAMP mode, no feathering (the common default)."""
        return cls(mode=BorderMode.CLAMP, feathering_width=0.0)

    @classmethod
    def constant(
        cls, value: Union[float, np.ndarray], *,
        feathering_width: float = 0.0,
        feathering_x_multiplier: float = 1.0,
        feathering_y_multiplier: float = 1.0,
    ) -> BorderConfig:
        """CONSTANT mode filled with ``value`` (scalar or per-channel)."""
        return cls(
            mode=BorderMode.CONSTANT,
            constant_value=value,
            feathering_width=feathering_width,
            feathering_x_multiplier=feathering_x_multiplier,
            feathering_y_multiplier=feathering_y_multiplier,
        )

    @classmethod
    def from_array(
        cls, array: np.ndarray, *,
        feathering_width: float = 0.0,
        feathering_x_multiplier: float = 1.0,
        feathering_y_multiplier: float = 1.0,
    ) -> BorderConfig:
        """ARRAY border (composite-onto) path using ``array`` as background."""
        return cls(
            mode=BorderMode.ARRAY,
            array=array,
            feathering_width=feathering_width,
            feathering_x_multiplier=feathering_x_multiplier,
            feathering_y_multiplier=feathering_y_multiplier,
        )

    def with_(self, **changes: Any) -> BorderConfig:
        """Derive a copy with the given field changes."""
        return replace(self, **changes)

    def to_dict(self) -> dict:
        """Serialise to a plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(
        cls, data: Union[BorderConfig, dict, None]
    ) -> BorderConfig:
        """Deserialise from a dict, BorderConfig, or None (returns default)."""
        if data is None:
            return cls()
        if isinstance(data, cls):
            return data
        return cls(**data)
