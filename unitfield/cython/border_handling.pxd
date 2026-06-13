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


cdef inline void _apply_border(
    double* u_x, double* u_y,
    double* blend, bint* use_border,
    int border_mode, double feather_w,
    double fx, double fy,
) noexcept nogil:
    """Modify u_x, u_y in-place; set blend factor and use_border flag."""
    cdef bint oob = (u_x[0] < 0.0 or u_x[0] > 1.0 or
                     u_y[0] < 0.0 or u_y[0] > 1.0)
    cdef double dx, dy, extra

    if not oob:
        blend[0] = 0.0
        use_border[0] = False
        return

    if border_mode == 0:  # CLAMP
        if u_x[0] < 0.0:
            u_x[0] = 0.0
        elif u_x[0] > 1.0:
            u_x[0] = 1.0
        if u_y[0] < 0.0:
            u_y[0] = 0.0
        elif u_y[0] > 1.0:
            u_y[0] = 1.0
        blend[0] = 0.0
        use_border[0] = False

    elif border_mode == 1:  # CONSTANT
        dx = _extra_1d(u_x[0])
        dy = _extra_1d(u_y[0])
        extra = sqrt(dx * dx * fx * fx + dy * dy * fy * fy)
        if u_x[0] < 0.0:
            u_x[0] = 0.0
        elif u_x[0] > 1.0:
            u_x[0] = 1.0
        if u_y[0] < 0.0:
            u_y[0] = 0.0
        elif u_y[0] > 1.0:
            u_y[0] = 1.0
        if feather_w <= 0.0 or extra >= feather_w:
            use_border[0] = True
            blend[0] = 1.0
        else:
            use_border[0] = False
            blend[0] = extra / feather_w

    elif border_mode == 2:  # REFLECT
        u_x[0] = _tri2(u_x[0])
        u_y[0] = _tri2(u_y[0])
        blend[0] = 0.0
        use_border[0] = False

    elif border_mode == 3:  # WRAP
        u_x[0] = _wrap_1d(u_x[0])
        u_y[0] = _wrap_1d(u_y[0])
        blend[0] = 0.0
        use_border[0] = False

    elif border_mode == 4:  # REFLECT_101
        u_x[0] = _ref101(u_x[0])
        u_y[0] = _ref101(u_y[0])
        blend[0] = 0.0
        use_border[0] = False

    elif border_mode == 5:  # ARRAY
        dx = _extra_1d(u_x[0])
        dy = _extra_1d(u_y[0])
        extra = sqrt(dx * dx * fx * fx + dy * dy * fy * fy)
        if u_x[0] < 0.0:
            u_x[0] = 0.0
        elif u_x[0] > 1.0:
            u_x[0] = 1.0
        if u_y[0] < 0.0:
            u_y[0] = 0.0
        elif u_y[0] > 1.0:
            u_y[0] = 1.0
        if feather_w <= 0.0 or extra >= feather_w:
            use_border[0] = True
            blend[0] = 1.0
        else:
            use_border[0] = False
            blend[0] = extra / feather_w
