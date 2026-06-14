"""Tests for the Cython remap kernel — numerical correctness and border modes."""

import numpy as np
import pytest

from unitfield import BorderConfig, BorderMode, remap_tensor

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def identity_maps():
    """Identity mapping in unit space: map_y[i,j] = i/(H-1), map_x[i,j] = j/(W-1)."""
    H, W = 16, 24
    ys, xs = np.meshgrid(
        np.linspace(0, 1, H), np.linspace(0, 1, W), indexing='ij'
    )
    return ys.astype(np.float64), xs.astype(np.float64)


@pytest.fixture
def checkerboard():
    """8x8 checkerboard pattern, shape (16, 24, 1)."""
    H, W = 16, 24
    check = (np.arange(H).reshape(-1, 1) + np.arange(W)) % 2
    return check.reshape(H, W, 1).astype(np.float64)


@pytest.fixture
def gradient():
    """Horizontal + vertical gradient, shape (16, 24, 1)."""
    H, W = 16, 24
    grad = np.linspace(0, 1, W).reshape(1, -1) * np.linspace(0, 1, H).reshape(-1, 1)
    return grad.reshape(H, W, 1).astype(np.float64)

@pytest.fixture
def gradient_2d(gradient):
    """Same gradient but 2D (H, W)."""
    return gradient.reshape(16, 24)


# ---------------------------------------------------------------------------
# Numerical correctness
# ---------------------------------------------------------------------------

class TestRemapNumerical:
    """Verify correct output for known inputs."""

    def test_identity_bilinear(self, checkerboard, identity_maps):
        """Identity mapping with bilinear should reproduce the source."""
        ys, xs = identity_maps
        src = checkerboard
        result = remap_tensor(src, xs, ys, interpolation=1)
        np.testing.assert_allclose(result, src, atol=1e-14)

    def test_identity_nearest(self):
        """Identity mapping with nearest should reproduce the source pixel-perfect."""
        H, W = 8, 8
        src = ((np.arange(H).reshape(-1, 1) + np.arange(W)) % 2).astype(np.float64)
        ys, xs = np.meshgrid(
            np.linspace(0, 1, H), np.linspace(0, 1, W), indexing='ij'
        )
        result = remap_tensor(src, xs.astype(np.float64), ys.astype(np.float64), interpolation=0)
        np.testing.assert_array_equal(result, src)

    def test_single_channel(self, gradient, identity_maps):
        """Grayscale (H, W) input — should work."""
        ys, xs = identity_maps
        src = gradient.reshape(16, 24)  # 2D
        result = remap_tensor(src, xs, ys, interpolation=1)
        assert result.shape == (16, 24)
        assert result.dtype == np.float64
        np.testing.assert_allclose(result, src, atol=1e-14)

    def test_multi_channel(self, gradient, identity_maps):
        """Multi-channel (H, W, 4) input."""
        ys, xs = identity_maps
        src = np.dstack([gradient] * 4)  # (16, 24, 4)
        result = remap_tensor(src, xs, ys, interpolation=1)
        assert result.shape == (16, 24, 4)
        np.testing.assert_allclose(result, src, atol=1e-14)

    def test_multi_channel_7(self, identity_maps):
        """Arbitrary channel count (7)."""
        ys, xs = identity_maps
        H, W = 16, 24
        src = np.random.rand(H, W, 7).astype(np.float64)
        result = remap_tensor(src, xs, ys, interpolation=1)
        assert result.shape == (H, W, 7)


class TestRemapValidation:
    """Input validation."""

    def test_shape_mismatch_raises(self):
        src = np.random.rand(10, 10, 3).astype(np.float64)
        mx = np.random.rand(5, 5).astype(np.float64)
        my = np.random.rand(5, 5).astype(np.float64)
        with pytest.raises(ValueError, match="must match"):
            remap_tensor(src, mx, my)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-zero"):
            mx = np.zeros((0, 10), dtype=np.float64)
            my = np.zeros((0, 10), dtype=np.float64)
            remap_tensor(np.zeros((0, 10, 3)), mx, my)


# ---------------------------------------------------------------------------
# Border modes
# ---------------------------------------------------------------------------

class TestBorderModes:
    """All 6 border modes."""

    @pytest.fixture
    def src(self):
        """3×3 image, used for border tests where map == src shape."""
        return np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ], dtype=np.float64).reshape(3, 3, 1)

    def _make_oob_maps(self, H=3, W=3):
        """Coordinates partially outside [0, 1], matching (H, W)."""
        ys, xs = np.meshgrid(
            np.linspace(-0.1, 1.1, H), np.linspace(-0.1, 1.1, W), indexing='ij'
        )
        return ys.astype(np.float64), xs.astype(np.float64)

    def test_clamp(self, src):
        """CLAMP: OOB → nearest edge pixel."""
        ys, xs = self._make_oob_maps()
        result = remap_tensor(src, xs, ys, interpolation=0,
                              border_config=BorderConfig.clamp())
        assert result[0, 0, 0] == 1.0
        assert result[-1, -1, 0] == 9.0

    def test_constant_default(self, src):
        """CONSTANT with default value (0.0)."""
        ys, xs = self._make_oob_maps()
        bc = BorderConfig.constant(0.0)
        result = remap_tensor(src, xs, ys, interpolation=0, border_config=bc)
        assert result[0, 0, 0] == 0.0

    def test_constant_custom_value(self, src):
        """CONSTANT with custom value."""
        ys, xs = self._make_oob_maps()
        bc = BorderConfig.constant(42.0)
        result = remap_tensor(src, xs, ys, interpolation=0, border_config=bc)
        assert result[0, 0, 0] == 42.0

    def test_reflect(self, src):
        """REFLECT: mirror at edge — sample just outside left edge."""
        ys = np.full((3, 3), 0.5, dtype=np.float64)
        xs = np.full((3, 3), -0.1, dtype=np.float64)
        bc = BorderConfig(mode=BorderMode.REFLECT)
        result = remap_tensor(src, xs, ys, interpolation=1, border_config=bc)
        # -0.1 reflected → 0.1, y=0.5
        # yf = 0.5*2 = 1.0, xf = 0.1*2 = 0.2
        # Bilinear at src[1, 0] + 0.2 * (src[1,1] - src[1,0]) = 4 + 0.2*1 = 4.2
        assert result[0, 0, 0] == pytest.approx(4.2, abs=0.01)

    def test_wrap(self, src):
        """WRAP: tile / repeat — sample just past right edge."""
        ys = np.full((3, 3), 0.5, dtype=np.float64)
        xs = np.full((3, 3), 1.1, dtype=np.float64)
        bc = BorderConfig(mode=BorderMode.WRAP)
        result = remap_tensor(src, xs, ys, interpolation=1, border_config=bc)
        assert result[0, 0, 0] == pytest.approx(4.2, abs=0.01)

    def test_reflect_101(self, src):
        """REFLECT_101: mirror without repeating edge pixel."""
        ys = np.full((3, 3), 0.5, dtype=np.float64)
        xs = np.full((3, 3), -0.1, dtype=np.float64)
        bc = BorderConfig(mode=BorderMode.REFLECT_101)
        result = remap_tensor(src, xs, ys, interpolation=1, border_config=bc)
        assert result[0, 0, 0] == pytest.approx(4.2, abs=0.01)

    def test_array_mode(self, src):
        """ARRAY mode: OOB returns corresponding pixel from border_array."""
        ys, xs = self._make_oob_maps()
        border_arr = np.ones((3, 3, 1), dtype=np.float64) * 77.0
        bc = BorderConfig.from_array(border_arr)
        result = remap_tensor(src, xs, ys, interpolation=0, border_config=bc)
        assert result[0, 0, 0] == 77.0


# ---------------------------------------------------------------------------
# Feathering
# ---------------------------------------------------------------------------

class TestFeathering:
    """Feathering transitions between border and sampled values."""

    @pytest.fixture
    def src(self):
        return np.ones((3, 3, 1), dtype=np.float64) * 10.0

    def _make_map(self, val_x, val_y):
        """Create (3,3) map filled with val_x, val_y."""
        H, W = 3, 3
        xs = np.full((H, W), val_x, dtype=np.float64)
        ys = np.full((H, W), val_y, dtype=np.float64)
        return xs, ys

    def test_hard_edge(self, src):
        """feathering_width=0.0 → hard edge, no blend."""
        xs, ys = self._make_map(-0.5, 0.5)
        bc = BorderConfig.constant(0.0, feathering_width=0.0)
        result = remap_tensor(src, xs, ys, interpolation=1, border_config=bc)
        assert result[0, 0, 0] == 0.0

    def test_soft_edge(self, src):
        """feathering_width > 0 → blend between sample and border."""
        xs, ys = self._make_map(-0.1, 0.5)
        # dx=0.1, dy=0, extra=0.1, feather_w=0.2 → blend=0.5
        # result = 10 + 0.5 * (0 - 10) = 5.0
        bc = BorderConfig.constant(0.0, feathering_width=0.2)
        result = remap_tensor(src, xs, ys, interpolation=1, border_config=bc)
        assert result[0, 0, 0] == pytest.approx(5.0, abs=0.01)

    def test_asymmetric_multipliers(self, src):
        """Asymmetric feather multipliers."""
        xs, ys = self._make_map(0.5, -0.2)
        # dy=0.2, dx=0, extra = 0.2*2.0 = 0.4
        # feather_w=0.4 → blend=1.0 → pure border
        bc = BorderConfig.constant(
            0.0, feathering_width=0.4,
            feathering_x_multiplier=1.0, feathering_y_multiplier=2.0,
        )
        result = remap_tensor(src, xs, ys, interpolation=1, border_config=bc)
        assert result[0, 0, 0] == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# Dtype handling
# ---------------------------------------------------------------------------

class TestDtype:
    """dtype conversion behaviour."""

    def test_float64_in_out(self, identity_maps):
        ys, xs = identity_maps
        src = np.random.rand(16, 24).astype(np.float64)
        result = remap_tensor(src, xs, ys, interpolation=1)
        assert result.dtype == np.float64

    def test_float32_in_out(self, identity_maps):
        ys, xs = identity_maps
        src = np.random.rand(16, 24).astype(np.float32)
        result = remap_tensor(src, xs, ys, interpolation=1)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# Edge-sized arrays
# ---------------------------------------------------------------------------

class TestEdgeSizes:
    """1x1, 1xN, Nx1 images."""

    def test_1x1(self):
        src = np.array([[42.0]], dtype=np.float64)
        mx = np.array([[0.5]], dtype=np.float64)
        my = np.array([[0.5]], dtype=np.float64)
        result = remap_tensor(src, mx, my, interpolation=1)
        assert result[0, 0] == 42.0

    def test_1xN(self):
        src = np.arange(5, dtype=np.float64).reshape(1, 5)
        mx = np.linspace(0, 1, 5, dtype=np.float64).reshape(1, 5)
        my = np.zeros((1, 5), dtype=np.float64)
        result = remap_tensor(src, mx, my, interpolation=0)
        np.testing.assert_array_equal(result, src)

    def test_Nx1(self):
        src = np.arange(5, dtype=np.float64).reshape(5, 1)
        mx = np.zeros((5, 1), dtype=np.float64)
        my = np.linspace(0, 1, 5, dtype=np.float64).reshape(5, 1)
        result = remap_tensor(src, mx, my, interpolation=0)
        np.testing.assert_array_equal(result, src)
