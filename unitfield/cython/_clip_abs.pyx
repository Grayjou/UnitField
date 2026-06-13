# cython: boundscheck=False, wraparound=False, nonecheck=False
# cython: cdivision=True, initializedcheck=False
"""
Sign-preserving absolute-value clip kernel for float32/float64 arrays.
"""

import numpy as np
cimport numpy as cnp
from cython.parallel cimport prange
from libc.math cimport fabs, isnan

from ._zero_sign import ZERO_SIGN

cnp.import_array()


# ---------------------------------------------------------------------------
# Float64 kernel (serial)
# ---------------------------------------------------------------------------
cdef void _clip_abs_f64_kernel(
    const cnp.float64_t[::1] src,
    cnp.float64_t[::1] dst,
    cnp.float64_t threshold,
    cnp.float64_t zero_val,
    Py_ssize_t N,
) noexcept nogil:
    cdef Py_ssize_t i
    cdef cnp.float64_t val
    for i in range(N):
        val = src[i]
        if isnan(val):
            dst[i] = val
        elif fabs(val) < threshold:
            if val == 0.0:
                dst[i] = zero_val
            elif val > 0.0:
                dst[i] = threshold
            else:
                dst[i] = -threshold
        else:
            dst[i] = val


# ---------------------------------------------------------------------------
# Float32 kernel (serial)
# ---------------------------------------------------------------------------
cdef void _clip_abs_f32_kernel(
    const cnp.float32_t[::1] src,
    cnp.float32_t[::1] dst,
    cnp.float32_t threshold,
    cnp.float32_t zero_val,
    Py_ssize_t N,
) noexcept nogil:
    cdef Py_ssize_t i
    cdef cnp.float32_t val
    for i in range(N):
        val = src[i]
        if isnan(val):
            dst[i] = val
        elif fabs(val) < threshold:
            if val == 0.0:
                dst[i] = zero_val
            elif val > 0.0:
                dst[i] = threshold
            else:
                dst[i] = -threshold
        else:
            dst[i] = val


# ---------------------------------------------------------------------------
# Float64 kernel (parallel, OpenMP)
# ---------------------------------------------------------------------------
cdef void _clip_abs_f64_parallel(
    const cnp.float64_t[::1] src,
    cnp.float64_t[::1] dst,
    cnp.float64_t threshold,
    cnp.float64_t zero_val,
    Py_ssize_t N,
) noexcept nogil:
    cdef Py_ssize_t i
    cdef cnp.float64_t val
    for i in prange(N, nogil=True, schedule='static'):
        val = src[i]
        if isnan(val):
            dst[i] = val
        elif fabs(val) < threshold:
            if val == 0.0:
                dst[i] = zero_val
            elif val > 0.0:
                dst[i] = threshold
            else:
                dst[i] = -threshold
        else:
            dst[i] = val


# ---------------------------------------------------------------------------
# Float32 kernel (parallel, OpenMP)
# ---------------------------------------------------------------------------
cdef void _clip_abs_f32_parallel(
    const cnp.float32_t[::1] src,
    cnp.float32_t[::1] dst,
    cnp.float32_t threshold,
    cnp.float32_t zero_val,
    Py_ssize_t N,
) noexcept nogil:
    cdef Py_ssize_t i
    cdef cnp.float32_t val
    for i in prange(N, nogil=True, schedule='static'):
        val = src[i]
        if isnan(val):
            dst[i] = val
        elif fabs(val) < threshold:
            if val == 0.0:
                dst[i] = zero_val
            elif val > 0.0:
                dst[i] = threshold
            else:
                dst[i] = -threshold
        else:
            dst[i] = val


# ---------------------------------------------------------------------------
# Public Python entry point
# ---------------------------------------------------------------------------
def clip_abs(
    arr: np.ndarray,
    threshold: float,
    zero_sign: ZERO_SIGN = ZERO_SIGN.POSITIVE,
    *,
    out: np.ndarray | None = None,
    in_place: bool = False,
) -> np.ndarray:
    if threshold < 0.0:
        raise ValueError(f"threshold must be >= 0, got {threshold}")
    if not isinstance(zero_sign, ZERO_SIGN):
        raise TypeError(
            f"zero_sign must be a ZERO_SIGN enum value, "
            f"got {type(zero_sign).__name__}"
        )
    if out is not None and in_place:
        raise ValueError("out and in_place are mutually exclusive")

    zero_val = threshold if zero_sign is ZERO_SIGN.POSITIVE else -threshold
    N = arr.size
    use_parallel = N > 1_000_000

    if arr.dtype == np.float64:
        return _clip_abs_f64_dispatch(
            arr, threshold, zero_val, out, in_place, use_parallel
        )
    elif arr.dtype == np.float32:
        return _clip_abs_f32_dispatch(
            arr, threshold, zero_val, out, in_place, use_parallel
        )
    else:
        raise TypeError(
            f"Unsupported dtype {arr.dtype}; expected float32 or float64"
        )


# ---------------------------------------------------------------------------
# Type-specific dispatchers (avoid cdef function pointer limitations)
# ---------------------------------------------------------------------------
cdef _clip_abs_f64_dispatch(
    arr,
    cnp.float64_t threshold,
    cnp.float64_t zero_val,
    out,
    bint in_place,
    bint use_parallel,
):
    cdef cnp.float64_t[::1] src_view = arr.ravel()
    cdef Py_ssize_t N = arr.size

    if in_place:
        if not arr.flags.writeable:
            raise ValueError("arr is not writeable; cannot use in_place=True")
        if use_parallel:
            _clip_abs_f64_parallel(src_view, src_view, threshold, zero_val, N)
        else:
            _clip_abs_f64_kernel(src_view, src_view, threshold, zero_val, N)
        return arr

    if out is not None:
        if out.shape != arr.shape or out.dtype != arr.dtype:
            raise ValueError("out must have the same shape and dtype as arr")
        dst = out
    else:
        dst = np.empty_like(arr, dtype=np.float64)

    cdef cnp.float64_t[::1] dst_view = dst.ravel()
    if use_parallel:
        _clip_abs_f64_parallel(src_view, dst_view, threshold, zero_val, N)
    else:
        _clip_abs_f64_kernel(src_view, dst_view, threshold, zero_val, N)
    return dst


cdef _clip_abs_f32_dispatch(
    arr,
    cnp.float32_t threshold,
    cnp.float32_t zero_val,
    out,
    bint in_place,
    bint use_parallel,
):
    cdef cnp.float32_t[::1] src_view = arr.ravel()
    cdef Py_ssize_t N = arr.size

    if in_place:
        if not arr.flags.writeable:
            raise ValueError("arr is not writeable; cannot use in_place=True")
        if use_parallel:
            _clip_abs_f32_parallel(src_view, src_view, threshold, zero_val, N)
        else:
            _clip_abs_f32_kernel(src_view, src_view, threshold, zero_val, N)
        return arr

    if out is not None:
        if out.shape != arr.shape or out.dtype != arr.dtype:
            raise ValueError("out must have the same shape and dtype as arr")
        dst = out
    else:
        dst = np.empty_like(arr, dtype=np.float32)

    cdef cnp.float32_t[::1] dst_view = dst.ravel()
    if use_parallel:
        _clip_abs_f32_parallel(src_view, dst_view, threshold, zero_val, N)
    else:
        _clip_abs_f32_kernel(src_view, dst_view, threshold, zero_val, N)
    return dst
