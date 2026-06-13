"""ZERO_SIGN enum for clip_abs: resolves sign ambiguity at zero."""

import enum


class ZERO_SIGN(enum.Enum):
    """Determines the sign of ``clip_abs(x, t)`` when ``x == 0``.

    ``POSITIVE`` → ``+threshold``
    ``NEGATIVE`` → ``-threshold``
    """
    POSITIVE = 1
    NEGATIVE = -1
