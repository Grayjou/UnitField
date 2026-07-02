# cython: boundscheck=False, wraparound=False, nonecheck=False
# cython: cdivision=True, initializedcheck=False
"""Nearest-neighbor resize kernel for UnitField map/preview arrays."""

import numpy as np
cimport numpy as cnp
from cython.parallel cimport prange

cnp.import_array()


cdef inline Py_ssize_t _nearest_index(
    Py_ssize_t dst_i,
    Py_ssize_t src_size,
    Py_ssize_t dst_size,
) noexcept nogil:
    cdef double pos
    cdef Py_ssize_t idx
    if src_size == 1 or dst_size == 1:
        return 0
    pos = (<double>dst_i) * <double>(src_size - 1) / <double>(dst_size - 1)
    idx = <Py_ssize_t>(pos + 0.5)
    if idx < 0:
        return 0
    if idx >= src_size:
        return src_size - 1
    return idx


cdef void _resize2d_impl(
    const double[:, ::1] src,
    double[:, ::1] dst,
    Py_ssize_t src_h,
    Py_ssize_t src_w,
    Py_ssize_t dst_h,
    Py_ssize_t dst_w,
) noexcept nogil:
    cdef Py_ssize_t y, x, sy, sx
    for y in prange(dst_h, schedule='static'):
        sy = _nearest_index(y, src_h, dst_h)
        for x in range(dst_w):
            sx = _nearest_index(x, src_w, dst_w)
            dst[y, x] = src[sy, sx]


cdef void _resize3d_impl(
    const double[:, :, ::1] src,
    double[:, :, ::1] dst,
    Py_ssize_t src_h,
    Py_ssize_t src_w,
    Py_ssize_t dst_h,
    Py_ssize_t dst_w,
    Py_ssize_t channels,
) noexcept nogil:
    cdef Py_ssize_t y, x, ch, sy, sx
    for y in prange(dst_h, schedule='static'):
        sy = _nearest_index(y, src_h, dst_h)
        for x in range(dst_w):
            sx = _nearest_index(x, src_w, dst_w)
            for ch in range(channels):
                dst[y, x, ch] = src[sy, sx, ch]


def resize_nearest(
    image: np.ndarray,
    int width,
    int height,
) -> np.ndarray:
    """Resize a 2-D or 3-D array using endpoint-aligned nearest sampling."""
    if image.ndim not in (2, 3):
        raise ValueError(f"image must be 2-D or 3-D, got {image.ndim}-D")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError(f"image spatial dimensions must be positive, got {image.shape[:2]}")
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")
    if height <= 0:
        raise ValueError(f"height must be positive, got {height}")

    cdef Py_ssize_t src_h = image.shape[0]
    cdef Py_ssize_t src_w = image.shape[1]
    cdef Py_ssize_t channels

    src = np.ascontiguousarray(image, dtype=np.float64)
    if image.ndim == 2:
        dst2 = np.empty((height, width), dtype=np.float64)
        _resize2d_impl(src, dst2, src_h, src_w, height, width)
        return dst2.astype(image.dtype, copy=False)

    channels = image.shape[2]
    if channels <= 0:
        raise ValueError("image must have at least one channel")
    dst3 = np.empty((height, width, channels), dtype=np.float64)
    _resize3d_impl(src, dst3, src_h, src_w, height, width, channels)
    return dst3.astype(image.dtype, copy=False)
