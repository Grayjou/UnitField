"""Tests for clip_abs — sign-preserving absolute-value clamp."""

import numpy as np
import pytest

from unitfield import ZERO_SIGN, clip_abs


class TestClipAbsNumerical:
    """Numerical correctness for float64 and float32."""

    @pytest.mark.parametrize("dtype", [np.float64, np.float32])
    def test_above_threshold(self, dtype):
        """Values above threshold are unchanged."""
        x = np.array([5.0, -5.0, 100.0], dtype=dtype)
        result = clip_abs(x, 1.0)
        np.testing.assert_array_equal(result, x)

    @pytest.mark.parametrize("dtype", [np.float64, np.float32])
    def test_below_threshold(self, dtype):
        """Values below threshold are raised to threshold with sign preserved."""
        x = np.array([0.3, -0.3, 0.0], dtype=dtype)
        result = clip_abs(x, 1.0)
        expected = np.array([1.0, -1.0, 1.0], dtype=dtype)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.float64, np.float32])
    def test_equal_to_threshold(self, dtype):
        """Values equal to threshold pass through."""
        x = np.array([1.0, -1.0], dtype=dtype)
        result = clip_abs(x, 1.0)
        np.testing.assert_array_equal(result, x)

    def test_zero_positive_default(self):
        """clip_abs(0, t) → +t with default ZERO_SIGN.POSITIVE."""
        result = clip_abs(np.zeros(3, dtype=np.float64), 0.5)
        np.testing.assert_array_equal(result, [0.5, 0.5, 0.5])

    def test_zero_negative(self):
        """clip_abs(0, t) → -t with ZERO_SIGN.NEGATIVE."""
        result = clip_abs(np.zeros(3, dtype=np.float64), 0.5, ZERO_SIGN.NEGATIVE)
        np.testing.assert_array_equal(result, [-0.5, -0.5, -0.5])


class TestClipAbsEdgeCases:
    """Edge cases: NaN, inf, threshold=0."""

    def test_nan_propagates(self):
        """NaN propagates unchanged."""
        x = np.array([float('nan'), 1.0], dtype=np.float64)
        result = clip_abs(x, 0.5)
        assert np.isnan(result[0])
        assert result[1] == 1.0

    def test_inf_propagates(self):
        """Inf passes through unchanged."""
        x = np.array([float('inf'), -float('inf')], dtype=np.float64)
        result = clip_abs(x, 0.5)
        assert np.isinf(result[0]) and result[0] > 0
        assert np.isinf(result[1]) and result[1] < 0

    def test_threshold_zero(self):
        """threshold=0 leaves all values unchanged."""
        x = np.array([0.3, 0.0, -0.3], dtype=np.float64)
        result = clip_abs(x, 0.0)
        np.testing.assert_array_equal(result, x)


class TestClipAbsValidation:
    """Input validation."""

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError, match="threshold must be >= 0"):
            clip_abs(np.array([0.5], dtype=np.float64), -1.0)

    def test_invalid_zero_sign_raises(self):
        with pytest.raises(TypeError, match="zero_sign must be a ZERO_SIGN"):
            clip_abs(np.array([0.5], dtype=np.float64), 1.0, zero_sign=0)

    def test_integer_array_raises(self):
        with pytest.raises(TypeError, match="Unsupported dtype"):
            clip_abs(np.array([1, 2, 3], dtype=np.int32), 1.0)

    def test_both_out_and_inplace_raises(self):
        x = np.array([0.5], dtype=np.float64)
        out = np.empty_like(x)
        with pytest.raises(ValueError, match="mutually exclusive"):
            clip_abs(x, 1.0, out=out, in_place=True)


class TestClipAbsOutputModes:
    """in_place and out modes."""

    def test_in_place_returns_same_array(self):
        x = np.array([0.3, -0.3, 5.0], dtype=np.float64)
        result = clip_abs(x, 1.0, in_place=True)
        assert result is x

    def test_in_place_modifies(self):
        x = np.array([0.3, -0.3, 5.0], dtype=np.float64)
        clip_abs(x, 1.0, in_place=True)
        np.testing.assert_array_equal(x, [1.0, -1.0, 5.0])

    def test_in_place_non_writeable_raises(self):
        x = np.array([0.5], dtype=np.float64)
        x.flags.writeable = False
        with pytest.raises((ValueError, TypeError)):
            clip_abs(x, 1.0, in_place=True)

    def test_out_returns_same_array(self):
        x = np.array([0.3, -0.3, 5.0], dtype=np.float64)
        out = np.empty_like(x)
        result = clip_abs(x, 1.0, out=out)
        assert result is out

    def test_out_shape_mismatch_raises(self):
        x = np.array([0.3, -0.3], dtype=np.float64)
        out = np.empty(3, dtype=np.float64)
        with pytest.raises(ValueError, match="out must have the same shape"):
            clip_abs(x, 1.0, out=out)

    def test_out_dtype_mismatch_raises(self):
        x = np.array([0.3, -0.3], dtype=np.float64)
        out = np.empty(2, dtype=np.float32)
        with pytest.raises(ValueError, match="out must have the same shape"):
            clip_abs(x, 1.0, out=out)

    def test_non_contiguous(self):
        """Strided array slice works."""
        x = np.array([0.1, 0.2, 0.3, 0.4, 5.0, 6.0], dtype=np.float64)
        view = x[::2]  # [0.1, 0.3, 5.0]
        result = clip_abs(view, 1.0)
        expected = np.array([1.0, 1.0, 5.0], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)


class TestClipAbsMultiDim:
    """Multi-dimensional arrays."""

    def test_2d_array(self):
        x = np.array([[0.3, -0.3], [5.0, -5.0]], dtype=np.float64)
        result = clip_abs(x, 1.0)
        expected = np.array([[1.0, -1.0], [5.0, -5.0]], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

    def test_3d_array(self):
        x = np.ones((2, 3, 4), dtype=np.float64) * 0.5
        result = clip_abs(x, 1.0)
        np.testing.assert_array_equal(result, np.ones((2, 3, 4), dtype=np.float64))

    def test_preserves_shape(self):
        x = np.random.rand(4, 5, 6).astype(np.float64)
        result = clip_abs(x, 0.5)
        assert result.shape == x.shape
