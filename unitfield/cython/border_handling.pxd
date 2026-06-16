"""Border-handling inline functions for Cython remap kernel."""

from libc.math cimport fabs, fmod, sqrt


cdef inline double _tri2(double t) noexcept nogil:
    """Triangle wave for REFLECT border mode."""
    cdef double m = fmod(t, 2.0)
    if m < 0.0:
        m += 2.0
    return 1.0 - fabs(m - 1.0)


cdef inline double _wrap_1d(double t) noexcept nogil:
    """Wrap to [0, 1) for WRAP border mode."""
    cdef double r = fmod(t, 1.0)
    if r < 0.0:
        r += 1.0
    return r


cdef inline double _ref101(double t) noexcept nogil:
    """REFLECT_101: mirror without repeating edge pixel."""
    cdef double m = fmod(t, 2.0)
    if m < 0.0:
        m += 2.0
    if m <= 1.0:
        return m
    else:
        return 2.0 - m


cdef inline double _extra_1d(double u) noexcept nogil:
    """Overshoot distance: 0 if inside [0, 1], else distance to nearest edge."""
    if u < 0.0:
        return -u
    if u > 1.0:
        return u - 1.0
    return 0.0


cdef inline void _feather_dist_2d(
    double ux, double uy,
    double* dx, double* dy,
    double under_x, double over_x,
    double under_y, double over_y,
) noexcept nogil:
    """Compute per-edge feather distances for x and y axes."""
    if ux < 0.0:
        dx[0] = -ux * under_x
    elif ux > 1.0:
        dx[0] = (ux - 1.0) * over_x
    else:
        dx[0] = 0.0

    if uy < 0.0:
        dy[0] = -uy * under_y
    elif uy > 1.0:
        dy[0] = (uy - 1.0) * over_y
    else:
        dy[0] = 0.0


cdef inline bint _apply_border(
    double* u_x, double* u_y, double* extra,
    int border_mode,
    double under_x, double over_x,
    double under_y, double over_y,
) noexcept nogil:
    """Transform coordinates in-place and compute OOB feather distance.

    Returns True if the original coordinate was out-of-bounds.
    Sets ``extra[0]`` to the Euclidean feather distance for CONSTANT/ARRAY
    modes; sets it to 0.0 for CLAMP/REFLECT/WRAP/REFLECT_101.

    Feather blend logic is now the caller's responsibility — this function
    only computes the OOB geometry.
    """
    cdef bint oob
    cdef double dx, dy

    oob = (u_x[0] < 0.0 or u_x[0] > 1.0 or
           u_y[0] < 0.0 or u_y[0] > 1.0)

    if border_mode == 0:  # CLAMP
        if u_x[0] < 0.0:
            u_x[0] = 0.0
        elif u_x[0] > 1.0:
            u_x[0] = 1.0
        if u_y[0] < 0.0:
            u_y[0] = 0.0
        elif u_y[0] > 1.0:
            u_y[0] = 1.0
        extra[0] = 0.0

    elif border_mode == 1:  # CONSTANT
        _feather_dist_2d(u_x[0], u_y[0], &dx, &dy, under_x, over_x, under_y, over_y)
        extra[0] = sqrt(dx * dx + dy * dy)
        if u_x[0] < 0.0:
            u_x[0] = 0.0
        elif u_x[0] > 1.0:
            u_x[0] = 1.0
        if u_y[0] < 0.0:
            u_y[0] = 0.0
        elif u_y[0] > 1.0:
            u_y[0] = 1.0

    elif border_mode == 2:  # REFLECT
        u_x[0] = _tri2(u_x[0])
        u_y[0] = _tri2(u_y[0])
        extra[0] = 0.0

    elif border_mode == 3:  # WRAP
        u_x[0] = _wrap_1d(u_x[0])
        u_y[0] = _wrap_1d(u_y[0])
        extra[0] = 0.0

    elif border_mode == 4:  # REFLECT_101
        u_x[0] = _ref101(u_x[0])
        u_y[0] = _ref101(u_y[0])
        extra[0] = 0.0

    elif border_mode == 5:  # ARRAY
        _feather_dist_2d(u_x[0], u_y[0], &dx, &dy, under_x, over_x, under_y, over_y)
        extra[0] = sqrt(dx * dx + dy * dy)
        if u_x[0] < 0.0:
            u_x[0] = 0.0
        elif u_x[0] > 1.0:
            u_x[0] = 1.0
        if u_y[0] < 0.0:
            u_y[0] = 0.0
        elif u_y[0] > 1.0:
            u_y[0] = 1.0

    else:
        extra[0] = 0.0

    return oob
