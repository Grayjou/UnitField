# cython: boundscheck=False, wraparound=False, nonecheck=False
# cython: cdivision=True, initializedcheck=False
"""
1D remap kernel — remap 1-D signals by a unit-space coordinate map.

Interpolation modes: 0=nearest, 1=linear, 2=cubic, 3=lanczos3, 4=lanczos4
Border modes: 0=CLAMP, 1=CONSTANT, 2=REFLECT, 3=WRAP, 4=REFLECT_101, 5=ARRAY
"""

import numpy as np
cimport numpy as cnp
from cython.parallel cimport prange
from libc.math cimport floor, sin, fabs, fmod

cnp.import_array()

# ---------------------------------------------------------------------------
# 1-D border handling
# ---------------------------------------------------------------------------
cdef inline double _tri2(double t) noexcept nogil:
    cdef double m = fmod(t, 2.0)
    if m < 0.0: m += 2.0
    return 1.0 - fabs(m - 1.0)

cdef inline double _wrap_1d(double t) noexcept nogil:
    cdef double r = fmod(t, 1.0)
    if r < 0.0: r += 1.0
    return r

cdef inline double _ref101(double t) noexcept nogil:
    cdef double m = fmod(t, 2.0)
    if m < 0.0: m += 2.0
    return m if m <= 1.0 else 2.0 - m

cdef inline double _extra_1d(double u) noexcept nogil:
    if u < 0.0:  return -u
    if u > 1.0:  return u - 1.0
    return 0.0

cdef inline double _feather_dist_1d(
    double u, double under_x, double over_x,
) noexcept nogil:
    """1-D per-edge feather distance with multipliers."""
    if u < 0.0:
        return -u * under_x
    elif u > 1.0:
        return (u - 1.0) * over_x
    return 0.0


cdef inline bint _apply_border_1d(
    double* u, double* extra,
    int border_mode,
    double under_x, double over_x,
) noexcept nogil:
    """1-D border handler.

    Transform coordinate in-place and compute OOB feather distance.
    Returns True if the original coordinate was out-of-bounds.
    Sets ``extra[0]`` for CONSTANT/ARRAY modes (1-D overshoot distance).
    """
    cdef bint oob = (u[0] < 0.0 or u[0] > 1.0)

    if not oob:
        extra[0] = 0.0
        return False

    if border_mode == 0:  # CLAMP
        if u[0] < 0.0:   u[0] = 0.0
        elif u[0] > 1.0: u[0] = 1.0
        extra[0] = 0.0

    elif border_mode == 1:  # CONSTANT
        extra[0] = _feather_dist_1d(u[0], under_x, over_x)
        if u[0] < 0.0:   u[0] = 0.0
        elif u[0] > 1.0: u[0] = 1.0

    elif border_mode == 2:   # REFLECT
        u[0] = _tri2(u[0])
        extra[0] = 0.0
    elif border_mode == 3:   # WRAP
        u[0] = _wrap_1d(u[0])
        extra[0] = 0.0
    elif border_mode == 4:   # REFLECT_101
        u[0] = _ref101(u[0])
        extra[0] = 0.0
    elif border_mode == 5:   # ARRAY
        extra[0] = _feather_dist_1d(u[0], under_x, over_x)
        if u[0] < 0.0:   u[0] = 0.0
        elif u[0] > 1.0: u[0] = 1.0

    else:
        extra[0] = 0.0

    return oob


# ---------------------------------------------------------------------------
# 1-D interpolation
# ---------------------------------------------------------------------------
cdef inline double _sample_1d_nearest(
    const double[:] src,
    double u,
    Py_ssize_t N,
) noexcept nogil:
    cdef Py_ssize_t ix = <Py_ssize_t>(u * <double>(N - 1) + 0.5)
    if ix < 0:    ix = 0
    elif ix >= N: ix = N - 1
    return src[ix]


cdef inline double _sample_1d_linear(
    const double[:] src,
    double u,
    Py_ssize_t N,
) noexcept nogil:
    cdef double xf = u * <double>(N - 1)
    cdef Py_ssize_t x0 = <Py_ssize_t>xf
    cdef double dx = xf - <double>x0
    cdef Py_ssize_t x1 = x0 + 1 if x0 + 1 < N else x0
    return src[x0] + (src[x1] - src[x0]) * dx


cdef inline double _cubic_kernel(double t) noexcept nogil:
    cdef double at = fabs(t), at2 = at * at, at3 = at2 * at
    if at < 1.0:   return 1.5 * at3 - 2.5 * at2 + 1.0
    if at < 2.0:   return -0.5 * at3 + 2.5 * at2 - 4.0 * at + 2.0
    return 0.0


cdef inline double _sample_1d_cubic(
    const double[:] src,
    double u,
    Py_ssize_t N,
) noexcept nogil:
    cdef double xf = u * <double>(N - 1)
    cdef Py_ssize_t x0 = <Py_ssize_t>xf - 1
    cdef double wx, total, norm
    cdef Py_ssize_t ix, sx

    total = 0.0; norm = 0.0
    for ix in range(4):
        wx = _cubic_kernel(xf - <double>(x0 + ix))
        sx = x0 + ix
        if sx < 0:     sx = 0
        elif sx >= N:  sx = N - 1
        total += src[sx] * wx
        norm += wx

    return 0.0 if norm == 0.0 else total / norm


cdef inline double _lanczos_kernel(double t, int a) noexcept nogil:
    cdef double at = fabs(t), pix
    if at < 1.0e-15:     return 1.0
    if at < <double>a:
        pix = 3.141592653589793 * t
        return <double>a * sin(pix) * sin(pix / <double>a) / (pix * pix)
    return 0.0


cdef inline double _sample_1d_lanczos(
    const double[:] src,
    double u,
    Py_ssize_t N,
    int a,
) noexcept nogil:
    cdef double xf = u * <double>(N - 1)
    cdef Py_ssize_t x0 = <Py_ssize_t>xf - (a - 1)
    cdef double wx, total, norm
    cdef Py_ssize_t ix, sx

    total = 0.0; norm = 0.0
    for ix in range(2 * a):
        wx = _lanczos_kernel(xf - <double>(x0 + ix), a)
        sx = x0 + ix
        if sx < 0:     sx = 0
        elif sx >= N:  sx = N - 1
        total += src[sx] * wx
        norm += wx

    return 0.0 if norm == 0.0 else total / norm


cdef inline double _sample_1ch(
    const double[:] src,
    double u,
    Py_ssize_t N,
    int interp_mode,
) noexcept nogil:
    if interp_mode == 0:
        return _sample_1d_nearest(src, u, N)
    elif interp_mode == 1:
        return _sample_1d_linear(src, u, N)
    elif interp_mode == 2:
        return _sample_1d_cubic(src, u, N)
    elif interp_mode == 3:
        return _sample_1d_lanczos(src, u, N, 3)
    elif interp_mode == 4:
        return _sample_1d_lanczos(src, u, N, 4)
    return _sample_1d_linear(src, u, N)


# ---------------------------------------------------------------------------
# Kernel loop
# ---------------------------------------------------------------------------
cdef void _remap_kernel_1d_impl(
    const double[:, ::1] src,    # (N, C)
    const double[::1] map_x,     # (N,)
    double[:, ::1] dst,          # (N, C)
    int border_mode,
    double border_const,
    const double[:, ::1] border_array,
    double feather_w,
    double under_x, double over_x,
    const double[::1] feather_d,
    int interp_mode,
    Py_ssize_t N, Py_ssize_t C,
) noexcept nogil:
    """Main 1-D remap kernel loop."""
    cdef Py_ssize_t n, ch
    cdef double u, extra, val, blend, border_val, fw
    cdef bint oob
    cdef bint use_feather_d = feather_d.shape[0] > 0

    for n in prange(N, nogil=True, schedule='static'):
        u = map_x[n]

        extra = 0.0
        oob = _apply_border_1d(&u, &extra, border_mode, under_x, over_x)

        if oob and (border_mode == 1 or border_mode == 5):
            for ch in range(C):
                val = _sample_1ch(src[:, ch], u, N, interp_mode)
                fw = feather_d[ch] if use_feather_d else feather_w
                if fw <= 0.0 or extra >= fw:
                    if border_mode == 1:
                        dst[n, ch] = border_const
                    else:
                        dst[n, ch] = border_array[n, ch]
                else:
                    blend = extra / fw
                    if border_mode == 1:
                        border_val = border_const
                    else:
                        border_val = border_array[n, ch]
                    dst[n, ch] = val + blend * (border_val - val)
        else:
            for ch in range(C):
                dst[n, ch] = _sample_1ch(src[:, ch], u, N, interp_mode)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------
import os
def _detect_cpu_count() -> int:
    try:  return os.cpu_count() or 1
    except:  return 1


def remap_tensor_1d(
    src: np.ndarray,
    map_x: np.ndarray,
    *,
    interpolation: int = 1,
    border_config=None,
    num_threads: int = -1,
) -> np.ndarray:
    """Remap a 1-D signal using unit-space coordinates.

    Args:
        src: Input array, shape (N,) or (N, C). float64 preferred.
        map_x: Coordinate map, shape (N,), values in [0, 1].
        interpolation: 0=nearest, 1=linear, 2=cubic, 3=lanczos3, 4=lanczos4.
        border_config: BorderConfig instance (or None for clamp default).
        num_threads: Thread count (-1 = auto).

    Returns:
        Remapped array, same shape and dtype as src.
    """
    if src.ndim < 1 or src.ndim > 2:
        raise ValueError(f"src must be 1-D or 2-D, got ndim={src.ndim}")
    if map_x.ndim != 1:
        raise ValueError("map_x must be 1-dimensional")
    if len(src) != len(map_x):
        raise ValueError("src and map_x must have the same length")

    from ..core.border_config import BorderConfig, compute_feather_d
    from ..core.enums import BorderMode

    if border_config is None:
        bc = BorderConfig.clamp()
    else:
        bc = border_config

    src_contig = np.ascontiguousarray(src, dtype=np.float64)
    if src_contig.ndim == 1:
        src_contig = src_contig[:, np.newaxis]
    mx = np.ascontiguousarray(map_x, dtype=np.float64)

    N = len(src)
    C = 1 if src.ndim == 1 else src.shape[1]
    dst = np.empty((N, C), dtype=np.float64)

    border_array = np.empty((0, 0), dtype=np.float64)
    if bc.mode == BorderMode.ARRAY and bc.array is not None:
        ba = np.ascontiguousarray(bc.array, dtype=np.float64)
        if ba.ndim == 1:
            border_array = ba[:, np.newaxis]
        else:
            border_array = ba

    feather_d = compute_feather_d(bc, C)

    nthreads = num_threads if num_threads > 0 else _detect_cpu_count()

    border_const = bc.constant_value
    if isinstance(border_const, np.ndarray):
        border_const = float(border_const) if border_const.ndim == 0 else 0.0
    elif border_const is None:
        border_const = 0.0
    _remap_kernel_1d_impl(
        src_contig, mx, dst,
        bc.mode.value,
        border_const,
        border_array,
        bc.feathering_width,
        bc.feathering_x_undershoot_multiplier,
        bc.feathering_x_overshoot_multiplier,
        feather_d,
        interpolation,
        N, C,
    )

    if src.ndim == 1:
        return dst[:, 0].astype(src.dtype, copy=False)
    return dst.astype(src.dtype, copy=False)
