# cython: boundscheck=False, wraparound=False, nonecheck=False
# cython: cdivision=True, initializedcheck=False
"""
Cython remap kernel — replaces cv2.remap as UnitField's primary sampler.

Interpolation modes:
  0=nearest, 1=bilinear, 2=cubic (Catmull-Rom),
  3=lanczos3, 4=lanczos4
  Also accepts InterpMethod enum (NEAREST_*, LINEAR, CUBIC, LANCZOS4)
Border modes: 0=CLAMP, 1=CONSTANT, 2=REFLECT, 3=WRAP, 4=REFLECT_101, 5=ARRAY
"""

import numpy as np
cimport numpy as cnp
from cython.parallel cimport prange
from libc.math cimport floor, sin, fabs

from .border_handling cimport _apply_border

cnp.import_array()


# ---------------------------------------------------------------------------
# Nearest-neighbour interpolation (1 channel)
# ---------------------------------------------------------------------------
cdef inline double _sample_nearest_1ch(
    const double[:, :, ::1] src,
    double u_x, double u_y,
    Py_ssize_t ch, Py_ssize_t H, Py_ssize_t W,
) noexcept nogil:
    cdef Py_ssize_t ix = <Py_ssize_t>(u_x * <double>(W - 1) + 0.5)
    cdef Py_ssize_t iy = <Py_ssize_t>(u_y * <double>(H - 1) + 0.5)
    if ix < 0:
        ix = 0
    elif ix >= W:
        ix = W - 1
    if iy < 0:
        iy = 0
    elif iy >= H:
        iy = H - 1
    return src[iy, ix, ch]


# ---------------------------------------------------------------------------
# Bilinear interpolation (1 channel)
# ---------------------------------------------------------------------------
cdef inline double _sample_bilinear_1ch(
    const double[:, :, ::1] src,
    double u_x, double u_y,
    Py_ssize_t ch, Py_ssize_t H, Py_ssize_t W,
) noexcept nogil:
    cdef double xf = u_x * <double>(W - 1)
    cdef double yf = u_y * <double>(H - 1)
    cdef Py_ssize_t x0 = <Py_ssize_t>xf
    cdef Py_ssize_t y0 = <Py_ssize_t>yf
    cdef double dx = xf - <double>x0
    cdef double dy = yf - <double>y0
    cdef Py_ssize_t x1 = x0 + 1 if x0 + 1 < W else x0
    cdef Py_ssize_t y1 = y0 + 1 if y0 + 1 < H else y0

    cdef double tl = src[y0, x0, ch]
    cdef double tr = src[y0, x1, ch]
    cdef double bl = src[y1, x0, ch]
    cdef double br = src[y1, x1, ch]

    cdef double top = tl + (tr - tl) * dx
    cdef double bottom = bl + (br - bl) * dx
    return top + (bottom - top) * dy


# ---------------------------------------------------------------------------
# Bicubic (Catmull-Rom) helpers
# ---------------------------------------------------------------------------
cdef inline double _cubic_kernel(double t) noexcept nogil:
    """Catmull-Rom cubic kernel (a = -0.5)."""
    cdef double at = fabs(t)
    cdef double at2 = at * at
    cdef double at3 = at2 * at
    if at < 1.0:
        return 1.5 * at3 - 2.5 * at2 + 1.0
    elif at < 2.0:
        return -0.5 * at3 + 2.5 * at2 - 4.0 * at + 2.0
    return 0.0


cdef inline double _sample_bicubic_1ch(
    const double[:, :, ::1] src,
    double u_x, double u_y,
    Py_ssize_t ch, Py_ssize_t H, Py_ssize_t W,
) noexcept nogil:
    cdef double xf = u_x * <double>(W - 1)
    cdef double yf = u_y * <double>(H - 1)
    cdef Py_ssize_t x0 = <Py_ssize_t>xf - 1
    cdef Py_ssize_t y0 = <Py_ssize_t>yf - 1
    cdef double dy, wx, wy, total, norm
    cdef Py_ssize_t ix, iy, sx, sy

    total = 0.0
    norm = 0.0
    for iy in range(4):
        dy = yf - <double>(y0 + iy)
        wy = _cubic_kernel(dy)
        for ix in range(4):
            wx = _cubic_kernel(xf - <double>(x0 + ix))
            sx = x0 + ix
            sy = y0 + iy
            if sx < 0:
                sx = 0
            elif sx >= W:
                sx = W - 1
            if sy < 0:
                sy = 0
            elif sy >= H:
                sy = H - 1
            total += src[sy, sx, ch] * wx * wy
            norm += wx * wy

    if norm == 0.0:
        return 0.0
    return total / norm


# ---------------------------------------------------------------------------
# Lanczos helpers
# ---------------------------------------------------------------------------
cdef inline double _lanczos_kernel(double t, int a) noexcept nogil:
    """Sinc-windowed sinc kernel of radius a."""
    cdef double at = fabs(t)
    cdef double pix
    if at < 1.0e-15:
        return 1.0
    if at < <double>a:
        pix = 3.141592653589793 * t
        return <double>a * sin(pix) * sin(pix / <double>a) / (pix * pix)
    return 0.0


cdef inline double _sample_lanczos_1ch(
    const double[:, :, ::1] src,
    double u_x, double u_y,
    Py_ssize_t ch, Py_ssize_t H, Py_ssize_t W,
    int a,
) noexcept nogil:
    cdef double xf = u_x * <double>(W - 1)
    cdef double yf = u_y * <double>(H - 1)
    cdef Py_ssize_t x0 = <Py_ssize_t>xf - (a - 1)
    cdef Py_ssize_t y0 = <Py_ssize_t>yf - (a - 1)
    cdef Py_ssize_t ix, iy, sx, sy
    cdef double wx, wy, total, norm

    total = 0.0
    norm = 0.0
    for iy in range(2 * a):
        wy = _lanczos_kernel(yf - <double>(y0 + iy), a)
        for ix in range(2 * a):
            wx = _lanczos_kernel(xf - <double>(x0 + ix), a)
            sx = x0 + ix
            sy = y0 + iy
            if sx < 0:
                sx = 0
            elif sx >= W:
                sx = W - 1
            if sy < 0:
                sy = 0
            elif sy >= H:
                sy = H - 1
            total += src[sy, sx, ch] * wx * wy
            norm += wx * wy

    if norm == 0.0:
        return 0.0
    return total / norm


# ---------------------------------------------------------------------------
# Interpolation dispatch
# ---------------------------------------------------------------------------
cdef inline double _sample_1ch(
    const double[:, :, ::1] src,
    double u_x, double u_y,
    Py_ssize_t ch, Py_ssize_t H, Py_ssize_t W,
    int interp_mode,
) noexcept nogil:
    if interp_mode == 0:
        return _sample_nearest_1ch(src, u_x, u_y, ch, H, W)
    elif interp_mode == 1:
        return _sample_bilinear_1ch(src, u_x, u_y, ch, H, W)
    elif interp_mode == 2:
        return _sample_bicubic_1ch(src, u_x, u_y, ch, H, W)
    elif interp_mode == 3:
        return _sample_lanczos_1ch(src, u_x, u_y, ch, H, W, 3)
    elif interp_mode == 4:
        return _sample_lanczos_1ch(src, u_x, u_y, ch, H, W, 4)
    else:
        return _sample_bilinear_1ch(src, u_x, u_y, ch, H, W)


# ---------------------------------------------------------------------------
# Main kernel loop
# ---------------------------------------------------------------------------
cdef void _remap_kernel_impl(
    const double[:, :, ::1] src,
    const double[:, ::1] map_y,
    const double[:, ::1] map_x,
    double[:, :, ::1] dst,
    int border_mode,
    double border_const,
    const double[:, :, ::1] border_array,
    double feather_w,
    double fx, double fy,
    const double[::1] feather_d,
    int interp_mode,
    Py_ssize_t H, Py_ssize_t W, Py_ssize_t C,
) noexcept nogil:
    """Main 2-D remap kernel loop.

    border_mode: 0=CLAMP, 1=CONSTANT, 2=REFLECT, 3=WRAP, 4=REFLECT_101, 5=ARRAY
    feather_d: per-channel feather width array (empty = scalar fallback).
    """
    cdef Py_ssize_t h, w, ch
    cdef double u_x, u_y, extra, val, blend, border_val, fw
    cdef bint oob
    cdef bint use_feather_d = feather_d.shape[0] > 0

    for h in prange(H, nogil=True, schedule='static'):
        for w in range(W):
            u_x = map_x[h, w]
            u_y = map_y[h, w]

            # OOB feather distance; explicit extra=0.0 for OpenMP privatisation
            extra = 0.0
            oob = _apply_border(&u_x, &u_y, &extra, border_mode, fx, fy)

            if oob and (border_mode == 1 or border_mode == 5):
                for ch in range(C):
                    val = _sample_1ch(src, u_x, u_y, ch, H, W, interp_mode)
                    fw = feather_d[ch] if use_feather_d else feather_w
                    if fw <= 0.0 or extra >= fw:
                        if border_mode == 1:
                            dst[h, w, ch] = border_const
                        else:
                            dst[h, w, ch] = border_array[h, w, ch]
                    else:
                        blend = extra / fw
                        if border_mode == 1:
                            border_val = border_const
                        else:
                            border_val = border_array[h, w, ch]
                        dst[h, w, ch] = val + blend * (border_val - val)
            else:
                for ch in range(C):
                    dst[h, w, ch] = _sample_1ch(src, u_x, u_y, ch, H, W, interp_mode)


# ---------------------------------------------------------------------------
# Thread count detection
# ---------------------------------------------------------------------------
import os


def _detect_cpu_count() -> int:
    """Detect available CPU cores."""
    try:
        return os.cpu_count() or 1
    except Exception:
        return 1


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------
def remap_tensor(
    src: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    *,
    interpolation: int = 1,
    border_config=None,
    num_threads: int = -1,
) -> np.ndarray:
    """Remap an image using unit-space coordinates.

    Args:
        src: Input array, shape (H, W) or (H, W, C). float64 preferred.
        map_x: X-coordinate map, shape (H, W), values in [0, 1].
        map_y: Y-coordinate map, shape (H, W), values in [0, 1].
        interpolation: Interpolation mode (0=nearest, 1=bilinear).
        border_config: BorderConfig instance (or None for clamp default).
        num_threads: Thread count (-1 = auto).

    Returns:
        Remapped array, same shape and dtype as src.
    """
    if src.ndim < 2:
        raise ValueError(f"src must be at least 2-dimensional, got ndim={src.ndim}")
    if map_x.ndim != 2 or map_y.ndim != 2:
        raise ValueError("map_x and map_y must be 2-dimensional")
    if map_x.shape != map_y.shape:
        raise ValueError("map_x and map_y must have the same shape")
    if src.shape[0] != map_x.shape[0] or src.shape[1] != map_x.shape[1]:
        raise ValueError(
            f"src shape {src.shape[:2]} and map shape {map_x.shape} must match"
        )
    if src.shape[0] == 0 or src.shape[1] == 0:
        raise ValueError("src must have non-zero spatial dimensions")

    from ._zero_sign import ZERO_SIGN as _ZS  # noqa: avoid unused import issue
    from ..core.border_config import BorderConfig, compute_feather_d
    from ..core.enums import BorderMode

    bc: BorderConfig
    if border_config is None:
        bc = BorderConfig.clamp()
    else:
        bc = border_config

    # Validate ARRAY mode shape
    if bc.mode == BorderMode.ARRAY:
        if bc.array is None:
            raise ValueError("ARRAY mode requires a border_array")
        if bc.array.shape[:2] != src.shape[:2]:
            raise ValueError(
                f"border_array shape {bc.array.shape[:2]} must match "
                f"src shape {src.shape[:2]}"
            )
        C_src = 1 if src.ndim == 2 else src.shape[2]
        arr_ndim = bc.array.ndim
        if arr_ndim not in (2, 3):
            raise ValueError(
                f"border_array must be 2-D or 3-D, got {arr_ndim}-D"
            )
        if arr_ndim == 3 and bc.array.shape[2] != C_src:
            raise ValueError(
                f"border_array channels {bc.array.shape[2]} must match "
                f"src channels {C_src}"
            )

    src_contig = np.ascontiguousarray(src, dtype=np.float64)
    if src_contig.ndim == 2:
        src_contig = src_contig[:, :, np.newaxis]
    mx = np.ascontiguousarray(map_x, dtype=np.float64)
    my = np.ascontiguousarray(map_y, dtype=np.float64)

    H, W = src.shape[0], src.shape[1]
    C = 1 if src.ndim == 2 else src.shape[2]

    dst = np.empty((H, W, C), dtype=np.float64)

    border_array = np.empty((0, 0, 0), dtype=np.float64)
    if bc.mode == BorderMode.ARRAY and bc.array is not None:
        border_array = np.ascontiguousarray(bc.array, dtype=np.float64)

    feather_d = compute_feather_d(bc, C)

    nthreads = num_threads if num_threads > 0 else _detect_cpu_count()

    border_const = bc.constant_value
    if isinstance(border_const, np.ndarray):
        border_const = float(border_const) if border_const.ndim == 0 else 0.0
    elif border_const is None:
        border_const = 0.0
    _remap_kernel_impl(
        src_contig, my, mx, dst,
        bc.mode.value,
        border_const,
        border_array,
        bc.feathering_width,
        bc.feathering_x_multiplier,
        bc.feathering_y_multiplier,
        feather_d,
        interpolation,
        H, W, C,
    )

    if src.ndim == 2:
        return dst.reshape(H, W).astype(src.dtype, copy=False)
    return dst.astype(src.dtype, copy=False)
