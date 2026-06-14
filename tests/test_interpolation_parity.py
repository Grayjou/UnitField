"""Interpolation parity tests: Cython kernel vs OpenCV/reference implementations."""

import numpy as np
import pytest
import cv2

from unitfield import remap_tensor, remap_tensor_1d, InterpMethod


# ---------------------------------------------------------------------------
# InterpMethod enum acceptance
# ---------------------------------------------------------------------------

class TestInterpMethodEnum:
    """remap_tensor accepts InterpMethod enum values."""

    @pytest.fixture
    def img(self):
        H, W = 16, 24
        return np.random.rand(H, W).astype(np.float64)

    @pytest.fixture
    def maps(self):
        H, W = 16, 24
        ys, xs = np.meshgrid(
            np.linspace(0, 1, H), np.linspace(0, 1, W), indexing='ij'
        )
        return xs.astype(np.float64), ys.astype(np.float64)

    def test_nearest_manhattan(self, img, maps):
        xs, ys = maps
        r = remap_tensor(img, xs, ys, interpolation=InterpMethod.NEAREST_MANHATTAN)
        assert r.shape == img.shape

    def test_nearest_euclidean(self, img, maps):
        xs, ys = maps
        r = remap_tensor(img, xs, ys, interpolation=InterpMethod.NEAREST_EUCLIDEAN)
        assert r.shape == img.shape

    def test_linear(self, img, maps):
        xs, ys = maps
        r = remap_tensor(img, xs, ys, interpolation=InterpMethod.LINEAR)
        assert r.shape == img.shape

    def test_cubic(self, img, maps):
        xs, ys = maps
        r = remap_tensor(img, xs, ys, interpolation=InterpMethod.CUBIC)
        assert r.shape == img.shape

    def test_lanczos4(self, img, maps):
        xs, ys = maps
        r = remap_tensor(img, xs, ys, interpolation=InterpMethod.LANCZOS4)
        assert r.shape == img.shape


# ---------------------------------------------------------------------------
# Nearest + bilinear: exact parity with cv2
# ---------------------------------------------------------------------------

def _warped_maps(H, W, shift=0.3):
    """Non-identity maps: slightly shifted to sample between pixels."""
    ys, xs = np.meshgrid(
        np.linspace(0, 1, H), np.linspace(0, 1, W), indexing='ij'
    )
    xs = xs + shift / (W - 1)
    ys = ys + shift / (H - 1)
    np.clip(xs, 0, 1, out=xs)
    np.clip(ys, 0, 1, out=ys)
    return xs.astype(np.float64), ys.astype(np.float64)


def _cv2_remap(src, interp_int, shift=0.3):
    """Warped cv2.remap for comparison."""
    H, W = src.shape[:2]
    xs, ys = _warped_maps(H, W, shift)
    map_x = (xs * (W - 1)).astype(np.float32)
    map_y = (ys * (H - 1)).astype(np.float32)
    src_f32 = src.astype(np.float32) if src.ndim == 2 else src.astype(np.float32)
    return cv2.remap(src_f32, map_x, map_y, interp_int, borderMode=cv2.BORDER_REPLICATE)


class TestParityNearestBilinear:
    """Nearest and bilinear should match cv2 exactly (same math)."""

    @pytest.fixture(params=['gradient', 'checkerboard', 'random'])
    def src(self, request):
        H, W = 16, 24
        if request.param == 'gradient':
            g = np.linspace(0, 1, W).reshape(1, -1) * np.linspace(0, 1, H).reshape(-1, 1)
            return g.astype(np.float64)
        elif request.param == 'checkerboard':
            return ((np.arange(H).reshape(-1, 1) + np.arange(W)) % 2).astype(np.float64)
        else:
            return np.random.rand(H, W).astype(np.float64)

    @pytest.fixture
    def shift_val(self):
        return 0.3

    @pytest.fixture
    def maps(self, src, shift_val):
        return _warped_maps(*src.shape, shift=shift_val)

    def test_nearest(self, src, maps):
        xs, ys = maps
        ours = remap_tensor(src, xs, ys, interpolation=0)
        theirs = _cv2_remap(src, cv2.INTER_NEAREST, shift=0.3)
        np.testing.assert_allclose(ours, theirs, atol=1e-6)

    def test_bilinear(self, src, maps):
        xs, ys = maps
        ours = remap_tensor(src, xs, ys, interpolation=1)
        theirs = _cv2_remap(src, cv2.INTER_LINEAR, shift=0.3)
        # cv2 uses 5-bit fixed-point (dx/dy quantized to 1/32) for INTER_LINEAR;
        # our Cython kernel uses full float64 precision. Max diff ~0.013.
        np.testing.assert_allclose(ours, theirs, atol=0.02)


# ---------------------------------------------------------------------------
# Cubic + Lanczos: validated via numpy reference implementations
#   (OpenCV uses different kernel parameters, so exact cv2 parity is
#    not expected for these methods.)
# ---------------------------------------------------------------------------

def _catmull_rom_kernel(t):
    """Catmull-Rom (a=-0.5) cubic kernel."""
    at = np.abs(t)
    at2 = at ** 2
    at3 = at2 * at
    out = np.where(at < 1, 1.5 * at3 - 2.5 * at2 + 1.0,
                   np.where(at < 2, -0.5 * at3 + 2.5 * at2 - 4.0 * at + 2.0, 0.0))
    return out


def _lanczos_kernel(t, a):
    """Sinc-windowed sinc kernel of radius a."""
    eps = 1e-15
    pix = np.pi * t
    with np.errstate(all='ignore'):
        result = np.where(np.abs(t) < eps, 1.0,
                          np.where(np.abs(t) < a, a * np.sin(pix) * np.sin(pix / a) / (pix * pix), 0.0))
    return result


def _ref_interp(src, u_x, u_y, kernel_func, radius):
    """Reference 2D separable interpolation with given kernel."""
    H, W = src.shape
    xf = u_x * (W - 1)
    yf = u_y * (H - 1)
    x0 = np.floor(xf).astype(np.intp) - (radius - 1)
    y0 = np.floor(yf).astype(np.intp) - (radius - 1)

    total = np.zeros_like(u_x, dtype=np.float64)
    norm = np.zeros_like(u_x, dtype=np.float64)

    for j in range(2 * radius):
        wy = kernel_func(yf - (y0 + j))
        for i in range(2 * radius):
            wx = kernel_func(xf - (x0 + i))
            sx = np.clip(x0 + i, 0, W - 1)
            sy = np.clip(y0 + j, 0, H - 1)
            total += src[sy, sx] * wx * wy
            norm += wx * wy

    return np.where(norm > 0, total / norm, 0.0)


class TestParityCubicLanczos:
    """Cubic (Catmull-Rom) and Lanczos vs numpy reference."""

    @pytest.fixture(params=['gradient', 'checkerboard', 'random'])
    def src(self, request):
        H, W = 16, 24
        if request.param == 'gradient':
            g = np.linspace(0, 1, W).reshape(1, -1) * np.linspace(0, 1, H).reshape(-1, 1)
            return g.astype(np.float64)
        elif request.param == 'checkerboard':
            return ((np.arange(H).reshape(-1, 1) + np.arange(W)) % 2).astype(np.float64)
        else:
            return np.random.rand(H, W).astype(np.float64)

    @pytest.fixture
    def maps(self, src):
        return _warped_maps(*src.shape, shift=0.3)

    def test_cubic(self, src, maps):
        xs, ys = maps
        ours = remap_tensor(src, xs, ys, interpolation=2)
        expected = _ref_interp(src, xs, ys, _catmull_rom_kernel, 2)
        np.testing.assert_allclose(ours, expected, atol=1e-12)

    def test_lanczos3(self, src, maps):
        xs, ys = maps
        ours = remap_tensor(src, xs, ys, interpolation=3)
        expected = _ref_interp(src, xs, ys, lambda t: _lanczos_kernel(t, 3), 3)
        np.testing.assert_allclose(ours, expected, atol=1e-12)

    def test_lanczos4(self, src, maps):
        xs, ys = maps
        ours = remap_tensor(src, xs, ys, interpolation=4)
        expected = _ref_interp(src, xs, ys, lambda t: _lanczos_kernel(t, 4), 4)
        np.testing.assert_allclose(ours, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Lanczos3 basic smoke
# ---------------------------------------------------------------------------

class TestLanczos3Basic:
    """Lanczos3 (kernel ID 3) basic smoke tests."""

    def test_identity(self):
        H, W = 8, 8
        src = np.random.rand(H, W).astype(np.float64)
        ys, xs = np.meshgrid(
            np.linspace(0, 1, H), np.linspace(0, 1, W), indexing='ij'
        )
        result = remap_tensor(src, xs.astype(np.float64), ys.astype(np.float64), interpolation=3)
        assert result.shape == (H, W)

    def test_lanczos3_vs_lanczos4_different(self):
        """Lanczos3 and Lanczos4 should give different results on a warped map."""
        H, W = 16, 24
        src = np.random.rand(H, W).astype(np.float64)
        xs_w, ys_w = _warped_maps(H, W, shift=0.37)
        r3 = remap_tensor(src, xs_w, ys_w, interpolation=3)
        r4 = remap_tensor(src, xs_w, ys_w, interpolation=4)
        assert not np.allclose(r3, r4, atol=1e-10, rtol=0)


# ---------------------------------------------------------------------------
# 1D interpolation parity: Cython kernel vs numpy reference
# ---------------------------------------------------------------------------

def _ref_interp_1d(src, map_x, kernel_func, radius):
    """Reference 1D interpolation with given kernel."""
    N = len(src)
    xf = map_x * (N - 1)
    x0 = np.floor(xf).astype(np.intp) - (radius - 1)

    total = np.zeros_like(map_x, dtype=np.float64)
    norm = np.zeros_like(map_x, dtype=np.float64)

    for i in range(2 * radius):
        wx = kernel_func(xf - (x0 + i))
        sx = np.clip(x0 + i, 0, N - 1)
        total += src[sx] * wx
        norm += wx

    return np.where(norm > 0, total / norm, 0.0)


def _warped_1d_map(N, shift=0.3):
    """Non-identity 1-D map: slightly shifted to sample between samples."""
    m = np.linspace(0, 1, N) + shift / (N - 1)
    return np.clip(m, 0, 1)


class Test1DParityNearestLinear:
    """1D nearest/linear vs numpy reference."""

    @pytest.fixture(params=['sine', 'ramp', 'random'])
    def src(self, request):
        N = 50
        if request.param == 'sine':
            return np.sin(np.linspace(0, 4*np.pi, N)).astype(np.float64)
        elif request.param == 'ramp':
            return np.linspace(0, 1, N).astype(np.float64)
        else:
            return np.random.rand(N).astype(np.float64)

    @pytest.fixture
    def map_x(self, src):
        return _warped_1d_map(len(src), shift=0.3)

    def test_nearest(self, src, map_x):
        ours = remap_tensor_1d(src, map_x, interpolation=0)
        # nearest reference
        N = len(src)
        ix = np.round(map_x * (N - 1)).astype(np.intp)
        np.clip(ix, 0, N - 1, out=ix)
        expected = src[ix]
        np.testing.assert_allclose(ours, expected, atol=1e-14)

    def test_linear(self, src, map_x):
        ours = remap_tensor_1d(src, map_x, interpolation=1)
        # linear reference
        N = len(src)
        xf = map_x * (N - 1)
        x0 = np.floor(xf).astype(np.intp)
        dx = xf - x0
        x1 = np.clip(x0 + 1, 0, N - 1)
        expected = src[x0] + (src[x1] - src[x0]) * dx
        np.testing.assert_allclose(ours, expected, atol=1e-14)


class Test1DParityCubicLanczos:
    """1D cubic/lanczos vs numpy reference."""

    @pytest.fixture(params=['sine', 'ramp', 'random'])
    def src(self, request):
        N = 50
        if request.param == 'sine':
            return np.sin(np.linspace(0, 4*np.pi, N)).astype(np.float64)
        elif request.param == 'ramp':
            return np.linspace(0, 1, N).astype(np.float64)
        else:
            return np.random.rand(N).astype(np.float64)

    @pytest.fixture
    def map_x(self, src):
        return _warped_1d_map(len(src), shift=0.3)

    def test_cubic(self, src, map_x):
        ours = remap_tensor_1d(src, map_x, interpolation=2)
        expected = _ref_interp_1d(src, map_x, _catmull_rom_kernel, 2)
        np.testing.assert_allclose(ours, expected, atol=1e-12)

    def test_lanczos3(self, src, map_x):
        ours = remap_tensor_1d(src, map_x, interpolation=3)
        expected = _ref_interp_1d(src, map_x, lambda t: _lanczos_kernel(t, 3), 3)
        np.testing.assert_allclose(ours, expected, atol=1e-12)

    def test_lanczos4(self, src, map_x):
        ours = remap_tensor_1d(src, map_x, interpolation=4)
        expected = _ref_interp_1d(src, map_x, lambda t: _lanczos_kernel(t, 4), 4)
        np.testing.assert_allclose(ours, expected, atol=1e-12)
