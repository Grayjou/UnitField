"""Pure-Python fallback for clip_abs (used when Cython extension is not compiled)."""

import numpy as np

from ._zero_sign import ZERO_SIGN


def clip_abs(
    arr: np.ndarray,
    threshold: float,
    zero_sign: ZERO_SIGN = ZERO_SIGN.POSITIVE,
    *,
    out: np.ndarray | None = None,
    in_place: bool = False,
) -> np.ndarray:
    """Clip the absolute value of array elements, preserving sign.

    Pure-Python fallback — delegates to NumPy vectorised ops.
    """
    if threshold < 0.0:
        raise ValueError(f"threshold must be >= 0, got {threshold}")
    if not isinstance(zero_sign, ZERO_SIGN):
        raise TypeError(
            f"zero_sign must be a ZERO_SIGN enum value, "
            f"got {type(zero_sign).__name__}"
        )
    if out is not None and in_place:
        raise ValueError("out and in_place are mutually exclusive")
    if arr.dtype not in (np.float32, np.float64):
        raise TypeError(
            f"Unsupported dtype {arr.dtype}; expected float32 or float64"
        )

    abs_x = np.abs(arr)
    below = abs_x < threshold
    zero_val = threshold if zero_sign is ZERO_SIGN.POSITIVE else -threshold

    result = np.where(below, np.sign(arr) * threshold, arr)
    # Handle zero specially — np.sign(0) = 0, we need zero_val
    zero_mask = arr == 0.0
    if np.any(zero_mask & below):
        result = np.where(zero_mask & below, zero_val, result)

    if in_place:
        arr[:] = result
        return arr

    if out is not None:
        out[:] = result
        return out

    return result if out is None else out
