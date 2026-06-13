"""Cython declarations for sign-preserving absolute-value clip kernel."""

cimport numpy as cnp


cdef void _clip_abs_f64_kernel(
    const cnp.float64_t[::1] src,
    cnp.float64_t[::1] dst,
    cnp.float64_t threshold,
    cnp.float64_t zero_val,
    Py_ssize_t N,
) noexcept nogil


cdef void _clip_abs_f32_kernel(
    const cnp.float32_t[::1] src,
    cnp.float32_t[::1] dst,
    cnp.float32_t threshold,
    cnp.float32_t zero_val,
    Py_ssize_t N,
) noexcept nogil


cdef void _clip_abs_f64_parallel(
    const cnp.float64_t[::1] src,
    cnp.float64_t[::1] dst,
    cnp.float64_t threshold,
    cnp.float64_t zero_val,
    Py_ssize_t N,
) noexcept nogil


cdef void _clip_abs_f32_parallel(
    const cnp.float32_t[::1] src,
    cnp.float32_t[::1] dst,
    cnp.float32_t threshold,
    cnp.float32_t zero_val,
    Py_ssize_t N,
) noexcept nogil
