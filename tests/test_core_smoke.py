"""Smoke tests for UnitField -- do NOT import any compiled Cython extension."""

import numpy as np
import unitfield


def test_import():
    """Package imports without error."""
    assert unitfield.__version__


def test_border_enum_accessible():
    """BorderMode enum is importable without compiled extensions."""
    from unitfield import BorderMode
    assert BorderMode.CLAMP is not None
    assert BorderMode.CLAMP == 0
    assert BorderMode.REFLECT_101 == 4


def test_border_config_defaults():
    """BorderConfig dataclass works without compiled extensions."""
    from unitfield import BorderConfig, BorderMode
    bc = BorderConfig()
    assert bc.mode == BorderMode.CLAMP
    assert bc.feathering_width == 0.0

    bc2 = BorderConfig.clamp()
    assert bc2.mode == BorderMode.CLAMP

    bc3 = BorderConfig.constant(0.5)
    assert bc3.mode == BorderMode.CONSTANT
    assert bc3.constant_value == 0.5


def test_clip_abs_fallback():
    """clip_abs works via pure-Python path (before Cython is built)."""
    from unitfield import clip_abs
    x = np.array([0.3, -0.3, 5.0], dtype=np.float64)
    result = clip_abs(x, 1.0)
    expected = np.array([1.0, -1.0, 5.0])
    np.testing.assert_array_equal(result, expected)


def test_clip_abs_zero_positive():
    """clip_abs(0, t) with default POSITIVE sign."""
    from unitfield import clip_abs
    result = clip_abs(np.zeros(3, dtype=np.float64), 0.5)
    np.testing.assert_array_equal(result, [0.5, 0.5, 0.5])


def test_clip_abs_zero_negative():
    """clip_abs(0, t) with NEGATIVE sign."""
    from unitfield import clip_abs, ZERO_SIGN
    result = clip_abs(np.zeros(3, dtype=np.float64), 0.5, ZERO_SIGN.NEGATIVE)
    np.testing.assert_array_equal(result, [-0.5, -0.5, -0.5])


def test_clip_abs_nan():
    """NaN propagates unchanged; finite values above threshold are unchanged."""
    from unitfield import clip_abs
    x = np.array([float('nan'), 1.0], dtype=np.float64)
    result = clip_abs(x, 0.5)
    assert np.isnan(result[0])
    assert result[1] == 1.0  # |1.0| >= 0.5, unchanged


def test_clip_abs_inf():
    """Inf passes through unchanged."""
    from unitfield import clip_abs
    x = np.array([float('inf'), -float('inf')], dtype=np.float64)
    result = clip_abs(x, 0.5)
    assert np.isinf(result[0]) and result[0] > 0
    assert np.isinf(result[1]) and result[1] < 0


def test_clip_abs_negative_threshold_raises():
    """Negative threshold raises ValueError."""
    from unitfield import clip_abs
    x = np.array([0.5], dtype=np.float64)
    try:
        clip_abs(x, -1.0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_clip_abs_integer_raises():
    """Integer array raises TypeError."""
    from unitfield import clip_abs
    try:
        clip_abs(np.array([1, 2, 3], dtype=np.int32), 1.0)
        assert False, "Should have raised TypeError"
    except TypeError:
        pass


def test_clip_abs_in_place():
    """In-place modifies the original array."""
    from unitfield import clip_abs
    x = np.array([0.3, -0.3, 5.0], dtype=np.float64)
    result = clip_abs(x, 1.0, in_place=True)
    assert result is x
    np.testing.assert_array_equal(x, [1.0, -1.0, 5.0])


def test_clip_abs_out():
    """Pre-allocated output array."""
    from unitfield import clip_abs
    x = np.array([0.3, -0.3, 5.0], dtype=np.float64)
    out = np.empty_like(x)
    result = clip_abs(x, 1.0, out=out)
    assert result is out
    np.testing.assert_array_equal(out, [1.0, -1.0, 5.0])
