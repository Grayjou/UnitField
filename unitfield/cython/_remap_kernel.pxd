"""Cython declarations for the remap kernel."""

cimport numpy as cnp


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
    int interp_mode,
    Py_ssize_t H, Py_ssize_t W, Py_ssize_t C,
) noexcept nogil
