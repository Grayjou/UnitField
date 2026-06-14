"""
Tests for the remap_tensor function (formerly remap_tensor_cv2).
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np

from unitfield import BorderConfig, remap_tensor


class TestRemapTensor:
    """Tests for remap_tensor with various configurations."""

    def test_identity_bilinear(self):
        H, W = 8, 8
        src = np.random.rand(H, W).astype(np.float64)
        ys, xs = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing='ij')
        result = remap_tensor(src, xs.astype(np.float64), ys.astype(np.float64), interpolation=1)
        np.testing.assert_allclose(result, src, atol=1e-14)

    def test_border_config(self):
        H, W = 8, 8
        src = np.random.rand(H, W).astype(np.float64)
        ys, xs = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing='ij')
        result = remap_tensor(src, xs, ys, border_config=BorderConfig.clamp())
        assert result.shape == src.shape

    def test_interp_method_enum(self):
        from unitfield import InterpMethod
        H, W = 8, 8
        src = np.random.rand(H, W).astype(np.float64)
        ys, xs = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing='ij')
        for method in [InterpMethod.LINEAR, InterpMethod.NEAREST_MANHATTAN,
                       InterpMethod.CUBIC, InterpMethod.LANCZOS4]:
            result = remap_tensor(src, xs, ys, interpolation=method)
            assert result.shape == src.shape
