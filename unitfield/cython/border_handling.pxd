"""Border-handling inline functions for Cython remap kernel."""

from libc.math cimport fabs, fmod, fmax, fmin, sqrt, INFINITY


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


DEF FEATHER_SENTINEL = -1.0
DEF FEATHER_BEYOND_BORDER = 0
DEF FEATHER_TO_BORDER = 1


cdef inline void _feather_dist_2d(
    double ux, double uy,
    double* dx, double* dy,
    double under_x, double over_x,
    double under_y, double over_y,
    int feather_mode,
) noexcept nogil:
    """Compute per-edge feather distances for x and y axes.

    BEYOND_BORDER uses outside overshoot distance.  TO_BORDER uses distance
    to the nearest enabled edge while inside, and full border once outside.
    Negative multipliers saturate when OOB and contribute nothing inside.
    """
    if ux < 0.0:
        dx[0] = -ux * under_x if under_x >= 0.0 else INFINITY
    elif ux > 1.0:
        dx[0] = (ux - 1.0) * over_x if over_x >= 0.0 else INFINITY
    elif feather_mode == FEATHER_TO_BORDER:
        if under_x >= 0.0 and over_x >= 0.0:
            dx[0] = fmin(ux * under_x, (1.0 - ux) * over_x)
        elif under_x >= 0.0:
            dx[0] = ux * under_x
        elif over_x >= 0.0:
            dx[0] = (1.0 - ux) * over_x
        else:
            dx[0] = FEATHER_SENTINEL
    else:
        dx[0] = 0.0

    if uy < 0.0:
        dy[0] = -uy * under_y if under_y >= 0.0 else INFINITY
    elif uy > 1.0:
        dy[0] = (uy - 1.0) * over_y if over_y >= 0.0 else INFINITY
    elif feather_mode == FEATHER_TO_BORDER:
        if under_y >= 0.0 and over_y >= 0.0:
            dy[0] = fmin(uy * under_y, (1.0 - uy) * over_y)
        elif under_y >= 0.0:
            dy[0] = uy * under_y
        elif over_y >= 0.0:
            dy[0] = (1.0 - uy) * over_y
        else:
            dy[0] = FEATHER_SENTINEL
    else:
        dy[0] = 0.0


cdef inline double _combine_2d_distances(double dx, double dy, int mode) noexcept nogil:
    if dx == INFINITY or dy == INFINITY:
        return INFINITY
    if mode == 1:      # MAX_NORM
        return fmax(dx, dy)
    elif mode == 2:    # MANHATTAN
        return dx + dy
    elif mode == 3:    # SCALED_MANHATTAN
        return (dx + dy) * 0.7071067811865476
    elif mode == 4:    # ALPHA_MAX
        return fmax(dx, dy) + 0.41421356237309503 * fmin(dx, dy)
    elif mode == 5:    # ALPHA_MAX_SIMPLE
        return fmax(dx, dy) + 0.5 * fmin(dx, dy)
    elif mode == 6:    # TAYLOR
        return fmax(dx, dy) + 0.5 * fmin(dx, dy) - 0.25 * dx * dx - 0.25 * dy * dy
    elif mode == 8:    # WEIGHTED_MINMAX
        return 0.9604 * fmax(dx, dy) + 0.3981 * fmin(dx, dy)
    else:              # EUCLIDEAN and unknown-safe fallback
        return sqrt(dx * dx + dy * dy)


cdef inline bint _apply_border(
    double* u_x, double* u_y, double* extra,
    int border_mode,
    double under_x, double over_x,
    double under_y, double over_y,
    int feather_mode,
    int distance_mode,
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
        _feather_dist_2d(u_x[0], u_y[0], &dx, &dy, under_x, over_x, under_y, over_y, feather_mode)
        if feather_mode == FEATHER_TO_BORDER:
            if dx == FEATHER_SENTINEL and dy == FEATHER_SENTINEL:
                extra[0] = INFINITY
            elif dx == FEATHER_SENTINEL:
                extra[0] = dy
            elif dy == FEATHER_SENTINEL:
                extra[0] = dx
            else:
                extra[0] = fmin(dx, dy)
        else:
            extra[0] = _combine_2d_distances(dx, dy, distance_mode)
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
        _feather_dist_2d(u_x[0], u_y[0], &dx, &dy, under_x, over_x, under_y, over_y, feather_mode)
        if feather_mode == FEATHER_TO_BORDER:
            if dx == FEATHER_SENTINEL and dy == FEATHER_SENTINEL:
                extra[0] = INFINITY
            elif dx == FEATHER_SENTINEL:
                extra[0] = dy
            elif dy == FEATHER_SENTINEL:
                extra[0] = dx
            else:
                extra[0] = fmin(dx, dy)
        else:
            extra[0] = _combine_2d_distances(dx, dy, distance_mode)
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
