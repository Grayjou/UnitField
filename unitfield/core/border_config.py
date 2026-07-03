"""Border configuration dataclass for UnitField coordinate remapping."""

from __future__ import annotations

from dataclasses import InitVar, dataclass
from typing import Any

import numpy as np

from .enums import BorderMode, DistanceMode, FeatheringMode


def _coerce_distance_mode(value: DistanceMode | int | str) -> DistanceMode:
    if isinstance(value, DistanceMode):
        return value
    if isinstance(value, str):
        mapping = {
            "max_norm": DistanceMode.MAX_NORM,
            "max": DistanceMode.MAX_NORM,
            "manhattan": DistanceMode.MANHATTAN,
            "l1": DistanceMode.MANHATTAN,
            "scaled_manhattan": DistanceMode.SCALED_MANHATTAN,
            "alpha_max": DistanceMode.ALPHA_MAX,
            "alpha_max_simple": DistanceMode.ALPHA_MAX_SIMPLE,
            "taylor": DistanceMode.TAYLOR,
            "euclidean": DistanceMode.EUCLIDEAN,
            "l2": DistanceMode.EUCLIDEAN,
            "true_distance": DistanceMode.EUCLIDEAN,
            "weighted_minmax": DistanceMode.WEIGHTED_MINMAX,
        }
        try:
            return mapping[value.lower()]
        except KeyError as exc:
            raise ValueError(f"Unknown distance mode: {value!r}") from exc
    return DistanceMode(int(value))


def _normalize_feather_mask(value: list | np.ndarray | None, name: str) -> np.ndarray | None:
    if value is None:
        return None
    mask = np.asarray(value)
    if mask.ndim != 1:
        raise ValueError(f"{name} must be 1-D (length C), got shape {mask.shape}")
    return mask.astype(bool)


def _resolve_feather_mask(
    feather_dims: list | np.ndarray | None,
    feather_channels: list | np.ndarray | None,
) -> np.ndarray | None:
    dims = _normalize_feather_mask(feather_dims, "feather_dims")
    channels = _normalize_feather_mask(feather_channels, "feather_channels")
    if dims is None:
        return channels
    if channels is None:
        return dims
    if dims.shape != channels.shape or not np.array_equal(dims, channels):
        raise ValueError("feather_dims and feather_channels must match when both are provided")
    return dims


@dataclass(frozen=True)
class BorderConfig:
    """Border-handling configuration for UnitField remap.

    Frozen so a single config can be shared safely between calls.
    Use :meth:`with_` to derive modified copies.

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
        feathering_x_undershoot_multiplier: Left edge feather multiplier (u_x < 0.0).
            Negative values cancel feathering (pure border, hard edge).
        feathering_x_overshoot_multiplier: Right edge feather multiplier (u_x > 1.0).
            Negative values cancel feathering (pure border, hard edge).
        feathering_y_undershoot_multiplier: Top edge feather multiplier (u_y < 0.0).
            Negative values cancel feathering (pure border, hard edge).
        feathering_y_overshoot_multiplier: Bottom edge feather multiplier (u_y > 1.0).
            Negative values cancel feathering (pure border, hard edge).
        feathering_mode: Where the feathering blend region lives.
            ``BEYOND_BORDER`` (default) — feathering activates outside [0,1].
            ``TO_BORDER`` — feathering activates inside, toward the nearest edge.
            See :class:`FeatheringMode` for details.
        distance_mode: Distance metric for combining 2-D feather distances.
        preserve_non_feather_dims: When ``True``, channels disabled by
            ``feather_dims`` keep the clamped source value inside a 2-D
            BEYOND_BORDER feather ramp instead of taking the border value.
    """

    mode: BorderMode = BorderMode.CLAMP
    constant_value: float | np.ndarray | None = None
    array: np.ndarray | None = None
    feathering_width: float = 0.0
    feathering_x_undershoot_multiplier: float = 1.0
    feathering_x_overshoot_multiplier: float = 1.0
    feathering_y_undershoot_multiplier: float = 1.0
    feathering_y_overshoot_multiplier: float = 1.0
    feathering_mode: FeatheringMode = FeatheringMode.BEYOND_BORDER
    distance_mode: DistanceMode = DistanceMode.EUCLIDEAN
    preserve_non_feather_dims: bool = False
    feather_dims: list | np.ndarray | None = None
    """Per-channel bool mask of length ``C``.
    ``True`` = channel participates in feather blend; ``False`` = hard
    edge (full border value at OOB).  ``None`` = all channels feather.
    The canonical use case is RGBA with ``[False, False, False, True]``
    for alpha-only feathering."""
    feather_channels: InitVar[list | np.ndarray | None] = None

    def __post_init__(self, feather_channels: list | np.ndarray | None) -> None:
        if self.feathering_width < 0.0:
            raise ValueError("feathering_width must be non-negative")
        # Negative multipliers are valid — they cancel feathering
        # (pure border) for that edge.  No validation needed.
        if self.mode == BorderMode.ARRAY and self.array is None:
            raise ValueError("ARRAY mode requires a non-None array")
        object.__setattr__(self, "distance_mode", _coerce_distance_mode(self.distance_mode))
        object.__setattr__(
            self,
            "feather_dims",
            _resolve_feather_mask(self.feather_dims, feather_channels),
        )

    @classmethod
    def clamp(cls) -> BorderConfig:
        """CLAMP mode, no feathering (the common default)."""
        return cls(mode=BorderMode.CLAMP, feathering_width=0.0)

    @classmethod
    def constant(
        cls, value: float | np.ndarray, *,
        feathering_width: float = 0.0,
        feathering_x_undershoot_multiplier: float = 1.0,
        feathering_x_overshoot_multiplier: float = 1.0,
        feathering_y_undershoot_multiplier: float = 1.0,
        feathering_y_overshoot_multiplier: float = 1.0,
        feathering_mode: FeatheringMode = FeatheringMode.BEYOND_BORDER,
        distance_mode: DistanceMode | int | str = DistanceMode.EUCLIDEAN,
        preserve_non_feather_dims: bool = False,
        feather_dims: list | np.ndarray | None = None,
        feather_channels: list | np.ndarray | None = None,
    ) -> BorderConfig:
        """CONSTANT mode filled with ``value`` (scalar or per-channel)."""
        return cls(
            mode=BorderMode.CONSTANT,
            constant_value=value,
            feathering_width=feathering_width,
            feathering_x_undershoot_multiplier=feathering_x_undershoot_multiplier,
            feathering_x_overshoot_multiplier=feathering_x_overshoot_multiplier,
            feathering_y_undershoot_multiplier=feathering_y_undershoot_multiplier,
            feathering_y_overshoot_multiplier=feathering_y_overshoot_multiplier,
            feathering_mode=feathering_mode,
            distance_mode=_coerce_distance_mode(distance_mode),
            preserve_non_feather_dims=preserve_non_feather_dims,
            feather_dims=feather_dims,
            feather_channels=feather_channels,
        )

    @classmethod
    def from_array(
        cls, array: np.ndarray, *,
        feathering_width: float = 0.0,
        feathering_x_undershoot_multiplier: float = 1.0,
        feathering_x_overshoot_multiplier: float = 1.0,
        feathering_y_undershoot_multiplier: float = 1.0,
        feathering_y_overshoot_multiplier: float = 1.0,
        feathering_mode: FeatheringMode = FeatheringMode.BEYOND_BORDER,
        distance_mode: DistanceMode | int | str = DistanceMode.EUCLIDEAN,
        preserve_non_feather_dims: bool = False,
        feather_dims: list | np.ndarray | None = None,
        feather_channels: list | np.ndarray | None = None,
    ) -> BorderConfig:
        """ARRAY border (composite-onto) path using ``array`` as background."""
        return cls(
            mode=BorderMode.ARRAY,
            array=array,
            feathering_width=feathering_width,
            feathering_x_undershoot_multiplier=feathering_x_undershoot_multiplier,
            feathering_x_overshoot_multiplier=feathering_x_overshoot_multiplier,
            feathering_y_undershoot_multiplier=feathering_y_undershoot_multiplier,
            feathering_y_overshoot_multiplier=feathering_y_overshoot_multiplier,
            feathering_mode=feathering_mode,
            distance_mode=_coerce_distance_mode(distance_mode),
            preserve_non_feather_dims=preserve_non_feather_dims,
            feather_dims=feather_dims,
            feather_channels=feather_channels,
        )

    def with_(self, **changes: Any) -> BorderConfig:
        """Derive a copy with the given field changes."""
        data = self.to_dict()
        if "feather_channels" in changes and "feather_dims" not in changes:
            data["feather_dims"] = None
        data.update(changes)
        return type(self).from_dict(data)

    def to_dict(self) -> dict:
        """Serialise to a plain dict."""
        return {
            "mode": self.mode,
            "constant_value": self.constant_value,
            "array": self.array,
            "feathering_width": self.feathering_width,
            "feathering_x_undershoot_multiplier": self.feathering_x_undershoot_multiplier,
            "feathering_x_overshoot_multiplier": self.feathering_x_overshoot_multiplier,
            "feathering_y_undershoot_multiplier": self.feathering_y_undershoot_multiplier,
            "feathering_y_overshoot_multiplier": self.feathering_y_overshoot_multiplier,
            "feathering_mode": self.feathering_mode,
            "distance_mode": self.distance_mode,
            "preserve_non_feather_dims": self.preserve_non_feather_dims,
            "feather_dims": self.feather_dims,
        }

    def to_json_dict(self) -> dict:
        """Serialise to a dict containing only JSON-native values."""
        data = self.to_dict()
        data["mode"] = self.mode.name
        data["feathering_mode"] = self.feathering_mode.name
        data["distance_mode"] = self.distance_mode.name
        data["constant_value"] = _to_json_value(self.constant_value)
        data["array"] = _to_json_value(self.array)
        data["feather_dims"] = _to_json_value(self.feather_dims)
        return data

    @classmethod
    def from_dict(
        cls, data: BorderConfig | dict | None
    ) -> BorderConfig:
        """Deserialise from a dict, BorderConfig, or None (returns default)."""
        if data is None:
            return cls()
        if isinstance(data, cls):
            return data
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict, BorderConfig, or None, got {type(data)}")
        return cls(**data)

    @classmethod
    def from_json_dict(cls, data: dict | None) -> BorderConfig:
        """Deserialise a JSON-safe dict produced by :meth:`to_json_dict`."""
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict or None, got {type(data)}")
        normalized = dict(data)
        if isinstance(normalized.get("mode"), str):
            normalized["mode"] = BorderMode[normalized["mode"]]
        if isinstance(normalized.get("feathering_mode"), str):
            normalized["feathering_mode"] = FeatheringMode[normalized["feathering_mode"]]
        if isinstance(normalized.get("distance_mode"), str):
            value = normalized["distance_mode"]
            normalized["distance_mode"] = (
                DistanceMode[value]
                if value in DistanceMode.__members__
                else _coerce_distance_mode(value)
            )
        if isinstance(normalized.get("constant_value"), list):
            normalized["constant_value"] = np.asarray(normalized["constant_value"])
        if isinstance(normalized.get("array"), list):
            normalized["array"] = np.asarray(normalized["array"])
        return cls(**normalized)


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


def _to_json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_to_json_value(item) for item in value]
    if isinstance(value, list):
        return [_to_json_value(item) for item in value]
    return value


# Alias to avoid collision

RemapBorderConfig = BorderConfig
