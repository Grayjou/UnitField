"""Border configuration dataclass for UnitField coordinate remapping."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from .enums import BorderMode


@dataclass(frozen=True)
class BorderConfig:
    """Border-handling configuration for UnitField remap.

    Frozen so a single config can be shared safely between calls.
    Use :meth:`with_` or :func:`dataclasses.replace` to derive modified copies.

    Attributes:
        mode: How out-of-bounds coordinates are handled.
        constant_value: Value for CONSTANT mode.
            Scalar float — same value for all pixels/channels.
            1-D array ``(C,)`` — one value per channel, broadcast spatially.
            2-D array ``(H, W)`` — one value per pixel (single-channel src).
            3-D array ``(H, W, C)`` — full border image.
            ``None`` means 0.0 downstream.
        array: Optional 2-D / 3-D background array for ARRAY mode.
        feathering_width: Smooth-blend band width at borders (0.0 = hard edge).
        feathering_x_multiplier: Per-axis asymmetric feather multiplier (x).
        feathering_y_multiplier: Per-axis asymmetric feather multiplier (y).
    """

    mode: BorderMode = BorderMode.CLAMP
    constant_value: float | np.ndarray | None = None
    array: np.ndarray | None = None
    feathering_width: float = 0.0
    feathering_x_multiplier: float = 1.0
    feathering_y_multiplier: float = 1.0
    feather_dims: list | np.ndarray | None = None
    """Per-channel bool mask of length ``C``.
    ``True`` = channel participates in feather blend; ``False`` = hard
    edge (full border value at OOB).  ``None`` = all channels feather.
    The canonical use case is RGBA with ``[False, False, False, True]``
    for alpha-only feathering."""

    def __post_init__(self) -> None:
        if self.feathering_width < 0.0:
            raise ValueError("feathering_width must be non-negative")
        if self.feathering_x_multiplier < 0.0 or self.feathering_y_multiplier < 0.0:
            raise ValueError("feathering multipliers must be non-negative")
        if self.mode == BorderMode.ARRAY and self.array is None:
            raise ValueError("ARRAY mode requires a non-None array")
        if self.feather_dims is not None:
            fd = np.asarray(self.feather_dims)
            if fd.ndim != 1:
                raise ValueError(
                    f"feather_dims must be 1-D (length C), got shape {fd.shape}"
                )
            object.__setattr__(self, "feather_dims", fd.astype(bool))

    @classmethod
    def clamp(cls) -> BorderConfig:
        """CLAMP mode, no feathering (the common default)."""
        return cls(mode=BorderMode.CLAMP, feathering_width=0.0)

    @classmethod
    def constant(
        cls, value: float | np.ndarray, *,
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
        cls, data: BorderConfig | dict | None
    ) -> BorderConfig:
        """Deserialise from a dict, BorderConfig, or None (returns default)."""
        if data is None:
            return cls()
        if isinstance(data, cls):
            return data
        return cls(**data)


def compute_feather_d(cfg: BorderConfig, C: int) -> np.ndarray:
    """Convert ``feather_dims`` mask to per-dim feathering widths.

    When the mask is ``None`` or ``C <= 1``, returns an empty array
    (kernel falls back to scalar ``feathering_width``).  Otherwise returns
    a float64 array of length ``C`` where a ``True`` dim gets
    ``feathering_width`` and a ``False`` dim gets ``0.0`` (hard edge —
    full border constant at OOB).

    Args:
        cfg: Border configuration.
        C: Number of channels in the source image.

    Returns:
        Float64 array of length ``C``, or empty array for scalar fallback.
    """
    if cfg.feather_dims is None or C <= 1:
        return np.empty((0,), dtype=np.float64)
    mask = np.asarray(cfg.feather_dims, dtype=bool)
    if mask.shape[0] != C:
        raise ValueError(
            f"feather_dims length {mask.shape[0]} does not match "
            f"channel count {C}"
        )
    return np.where(mask, cfg.feathering_width, 0.0).astype(np.float64)
