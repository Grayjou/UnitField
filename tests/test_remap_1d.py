"""
Tests for 1D remap kernel and Unit1DMappedEndomorphism.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np

from unitfield import (
    remap_tensor_1d,
    Unit1DMappedEndomorphism,
    InterpMethod,
    BorderConfig,
    BorderMode,
    U1DE,
    MappedUnitField,
)


# ---------------------------------------------------------------------------
# remap_tensor_1d
# ---------------------------------------------------------------------------

class TestRemapTensor1D:
    def test_identity_linear(self):
        N = 100
        src = np.sin(np.linspace(0, 4*np.pi, N)).astype(np.float64)
        identity = np.linspace(0, 1, N).astype(np.float64)
        out = remap_tensor_1d(src, identity, interpolation=1)
        np.testing.assert_allclose(out, src, atol=1e-14)

    def test_identity_nearest(self):
        N = 100
        src = np.sin(np.linspace(0, 4*np.pi, N)).astype(np.float64)
        identity = np.linspace(0, 1, N).astype(np.float64)
        out = remap_tensor_1d(src, identity, interpolation=0)
        np.testing.assert_allclose(out, src, atol=1e-14)

    def test_interpolation_modes(self):
        N = 50
        src = np.sin(np.linspace(0, 2*np.pi, N)).astype(np.float64)
        identity = np.linspace(0, 1, N).astype(np.float64)
        for mode in range(5):
            out = remap_tensor_1d(src, identity, interpolation=mode)
            assert out.shape == (N,)
            assert out.dtype == np.float64

    def test_interp_method_enum(self):
        N = 50
        src = np.random.rand(N).astype(np.float64)
        identity = np.linspace(0, 1, N).astype(np.float64)
        out = remap_tensor_1d(src, identity, interpolation=InterpMethod.LINEAR)
        assert out.shape == (N,)

    def test_multi_channel(self):
        N = 50
        src = np.random.rand(N, 3).astype(np.float64)
        identity = np.linspace(0, 1, N).astype(np.float64)
        out = remap_tensor_1d(src, identity, interpolation=1)
        assert out.shape == (N, 3)

    def test_border_config(self):
        N = 50
        src = np.ones(N, dtype=np.float64) * 42.0
        shift = np.zeros(N, dtype=np.float64) + 2.0  # all OOB
        bc = BorderConfig.constant(value=0.0)
        out = remap_tensor_1d(src, shift, interpolation=1, border_config=bc)
        np.testing.assert_allclose(out, np.zeros(N), atol=1e-14)

    def test_border_clamp(self):
        N = 50
        src = np.arange(N, dtype=np.float64)
        oob = np.full(N, -0.5, dtype=np.float64)
        out = remap_tensor_1d(src, oob, interpolation=0, border_config=BorderConfig.clamp())
        np.testing.assert_allclose(out, np.full(N, src[0]), atol=1e-14)

    def test_border_reflect(self):
        N = 50
        src = np.arange(N, dtype=np.float64)
        # map 1.2 → reflected to 0.8 → pixel 0.8*(N-1) = 39.2 → ~src[39]
        oob = np.full(N, 1.2, dtype=np.float64)
        bc = BorderConfig(mode=BorderMode.REFLECT)
        out = remap_tensor_1d(src, oob, interpolation=0, border_config=bc)
        expected_pos = (1 - (1.2 - 1.0)) * (N - 1)
        ix = int(round(expected_pos))
        np.testing.assert_allclose(out[0], float(ix), atol=1e-14)

    def test_border_wrap(self):
        N = 50
        src = np.arange(N, dtype=np.float64)
        oob = np.full(N, 1.5, dtype=np.float64)
        bc = BorderConfig(mode=BorderMode.WRAP)
        out = remap_tensor_1d(src, oob, interpolation=1, border_config=bc)
        # 1.5 wraps to 0.5 → pixel 0.5*(N-1) = 24.5
        expected = 24.5
        assert abs(out[0] - expected) < 1.0

    def test_constant_time_warp(self):
        N = 200
        t = np.linspace(0, 1, N)
        src = np.sin(2 * np.pi * 10 * t)
        # speed up in middle: map (t) → (t + 0.2*sin(2π*t))
        warp = t + 0.2 * np.sin(2 * np.pi * t)
        warp = np.clip(warp, 0, 1)
        out = remap_tensor_1d(src, warp, interpolation=1)
        assert out.shape == (N,)
        assert out.dtype == np.float64
        assert np.all(np.isfinite(out))

    def test_empty(self):
        out = remap_tensor_1d(np.array([]), np.array([]))
        assert out.shape == (0,)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            remap_tensor_1d(np.ones(10), np.ones(5))


# ---------------------------------------------------------------------------
# Unit1DMappedEndomorphism
# ---------------------------------------------------------------------------

class TestUnit1DMappedEndomorphism:
    def test_valid_initialization(self):
        data = np.linspace(0, 1, 50).reshape(-1, 1)
        endo = Unit1DMappedEndomorphism(data, InterpMethod.LINEAR)
        assert endo.ndim == 1
        assert endo.spatial_shape == (50,)
        assert endo.interp_method == InterpMethod.LINEAR

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError):
            Unit1DMappedEndomorphism(np.random.rand(10, 10, 2))
        with pytest.raises(ValueError):
            Unit1DMappedEndomorphism(np.random.rand(10, 3))
        with pytest.raises(ValueError):  # noqa: E501
            Unit1DMappedEndomorphism(np.random.rand(10))

    def test_get_value(self):
        data = np.linspace(0, 1, 50).reshape(-1, 1)
        endo = Unit1DMappedEndomorphism(data, InterpMethod.LINEAR)
        v = endo.get_value((0.5,))
        assert isinstance(v, tuple)
        assert len(v) == 1
        assert abs(v[0] - 0.5) < 0.02

    def test_get_value_nearest(self):
        data = np.linspace(0, 1, 50).reshape(-1, 1)
        endo = Unit1DMappedEndomorphism(data, InterpMethod.NEAREST_MANHATTAN)
        v = endo.get_value((0.5,))
        assert abs(v[0] - 0.5) < 0.02

    def test_get_values(self):
        data = np.linspace(0, 1, 50).reshape(-1, 1)
        endo = Unit1DMappedEndomorphism(data, InterpMethod.LINEAR)
        coords = np.array([[0.0], [0.5], [1.0]])
        vals = endo.get_values(coords)
        assert vals.shape == (3, 1)
        np.testing.assert_allclose(vals[:, 0], [0.0, 0.5, 1.0], atol=0.02)

    def test_remap_identity(self):
        data = np.linspace(0, 1, 50).reshape(-1, 1)
        endo = Unit1DMappedEndomorphism(data, InterpMethod.LINEAR)
        src = np.sin(np.linspace(0, 4*np.pi, 50)).astype(np.float64)
        out = endo.remap(src, interpolation=1)
        np.testing.assert_allclose(out, src, atol=1e-12)

    def test_remap_multi_channel(self):
        data = np.linspace(0, 1, 50).reshape(-1, 1)
        endo = Unit1DMappedEndomorphism(data, InterpMethod.LINEAR)
        src = np.random.rand(50, 4).astype(np.float64)
        out = endo.remap(src, interpolation=1)
        np.testing.assert_allclose(out, src, atol=1e-12)

    def test_remap_with_border(self):
        shift = np.ones(50).reshape(-1, 1) * 0.5
        endo = Unit1DMappedEndomorphism(shift, InterpMethod.NEAREST_MANHATTAN)
        src = np.arange(50, dtype=np.float64)
        bc = BorderConfig.constant(value=-1.0)
        out = endo.remap(src, border_config=bc)
        # All mapped to position 0.5 → pixel 24.5 → value ~24 or 25
        assert np.all(np.abs(out - [24.5]) < 1.0)

    def test_remap_interpolation_modes(self):
        data = np.linspace(0, 1, 50).reshape(-1, 1)
        endo = Unit1DMappedEndomorphism(data, InterpMethod.LINEAR)
        src = np.sin(np.linspace(0, 2*np.pi, 50)).astype(np.float64)
        for mode in range(5):
            out = endo.remap(src, interpolation=mode)
            assert out.shape == src.shape

    def test_caching(self):
        data = np.linspace(0, 1, 50).reshape(-1, 1)
        endo = Unit1DMappedEndomorphism(data, InterpMethod.LINEAR, cache_size=10)
        # warm up
        endo.get_value((0.5,))
        # second call should be cached
        v = endo.get_value((0.5,))
        assert abs(v[0] - 0.5) < 0.02

    def test_alias(self):
        data = np.linspace(0, 1, 50).reshape(-1, 1)
        e = U1DE(data)
        assert isinstance(e, Unit1DMappedEndomorphism)

    def test_with_interp_method(self):
        data = np.linspace(0, 1, 50).reshape(-1, 1)
        endo = Unit1DMappedEndomorphism(data, InterpMethod.LINEAR)
        modified = endo.with_interp_method(InterpMethod.NEAREST_MANHATTAN)
        assert modified.interp_method == InterpMethod.NEAREST_MANHATTAN

    def test_sign_change_warp(self):
        N = 100
        data = np.linspace(1, 0, N).reshape(-1, 1)  # decreasing
        endo = Unit1DMappedEndomorphism(data, InterpMethod.LINEAR)
        src = np.sin(np.linspace(0, 4*np.pi, N)).astype(np.float64)
        out = endo.remap(src, interpolation=1)
        # decreasing map flips the signal
        assert abs(out[0] - src[-1]) < 0.1
        assert abs(out[-1] - src[0]) < 0.1

    def test_repr(self):
        data = np.linspace(0, 1, 50).reshape(-1, 1)
        endo = Unit1DMappedEndomorphism(data)
        r = repr(endo)
        assert "Unit1DMappedEndomorphism" in r
        assert "50" in r
