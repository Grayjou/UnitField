"""Benchmarks comparing Cython remap kernel vs cv2.remap."""

import cv2
import numpy as np

from unitfield import remap_tensor


def _make_src(H, W, channels=1):
    src = np.random.rand(H, W, channels).astype(np.float64)
    if channels == 1:
        return src.reshape(H, W)
    return src


def _make_maps(H, W):
    ys, xs = np.meshgrid(
        np.linspace(0, 1, H), np.linspace(0, 1, W), indexing='ij'
    )
    return xs.astype(np.float64), ys.astype(np.float64)


# ---------------------------------------------------------------------------
# Cython kernel — all interpolation modes at 512x512
# ---------------------------------------------------------------------------

class TestBenchCython:
    """Throughput across all 5 interpolation modes."""

    def setup_method(self):
        self.H, self.W = 512, 512
        self.src = _make_src(self.H, self.W)
        self.mx, self.my = _make_maps(self.H, self.W)

    def test_nearest(self, benchmark):
        benchmark(remap_tensor, self.src, self.mx, self.my, interpolation=0)

    def test_bilinear(self, benchmark):
        benchmark(remap_tensor, self.src, self.mx, self.my, interpolation=1)

    def test_cubic(self, benchmark):
        benchmark(remap_tensor, self.src, self.mx, self.my, interpolation=2)

    def test_lanczos3(self, benchmark):
        benchmark(remap_tensor, self.src, self.mx, self.my, interpolation=3)

    def test_lanczos4(self, benchmark):
        benchmark(remap_tensor, self.src, self.mx, self.my, interpolation=4)


# ---------------------------------------------------------------------------
# Cython vs cv2 — single-threaded
# ---------------------------------------------------------------------------

class TestBenchVsCV2:
    """Comparison against cv2.remap at 512x512."""

    def setup_method(self):
        self.H, self.W = 512, 512
        self.src = _make_src(self.H, self.W)
        self.mx, self.my = _make_maps(self.H, self.W)
        self.mx_cv = (self.mx * (self.W - 1)).astype(np.float32)
        self.my_cv = (self.my * (self.H - 1)).astype(np.float32)
        self.src_f32 = self.src.astype(np.float32)
        self.bm = cv2.BORDER_REPLICATE

    def test_cython_nearest(self, benchmark):
        benchmark(remap_tensor, self.src, self.mx, self.my, interpolation=0, num_threads=1)

    def test_cv2_nearest(self, benchmark):
        benchmark(cv2.remap, self.src_f32, self.mx_cv, self.my_cv,
                  cv2.INTER_NEAREST, borderMode=self.bm)

    def test_cython_bilinear(self, benchmark):
        benchmark(remap_tensor, self.src, self.mx, self.my, interpolation=1, num_threads=1)

    def test_cv2_bilinear(self, benchmark):
        benchmark(cv2.remap, self.src_f32, self.mx_cv, self.my_cv,
                  cv2.INTER_LINEAR, borderMode=self.bm)

    def test_cython_cubic(self, benchmark):
        benchmark(remap_tensor, self.src, self.mx, self.my, interpolation=2, num_threads=1)

    def test_cv2_cubic(self, benchmark):
        benchmark(cv2.remap, self.src_f32, self.mx_cv, self.my_cv,
                  cv2.INTER_CUBIC, borderMode=self.bm)

    def test_cython_lanczos4(self, benchmark):
        benchmark(remap_tensor, self.src, self.mx, self.my, interpolation=4, num_threads=1)

    def test_cv2_lanczos4(self, benchmark):
        benchmark(cv2.remap, self.src_f32, self.mx_cv, self.my_cv,
                  cv2.INTER_LANCZOS4, borderMode=self.bm)


# ---------------------------------------------------------------------------
# Thread scaling — 1024x1024, 3-channels
# ---------------------------------------------------------------------------

class TestBenchThreadScaling:
    """1 thread vs auto-detect at 1024x1024 with 3 channels."""

    def setup_method(self):
        self.H, self.W = 1024, 1024
        self.src = _make_src(self.H, self.W, channels=3)
        self.mx, self.my = _make_maps(self.H, self.W)

    def test_single_thread(self, benchmark):
        benchmark(remap_tensor, self.src, self.mx, self.my, interpolation=1, num_threads=1)

    def test_auto_threads(self, benchmark):
        benchmark(remap_tensor, self.src, self.mx, self.my, interpolation=1, num_threads=-1)


# ---------------------------------------------------------------------------
# Multi-channel scaling
# ---------------------------------------------------------------------------

class TestMultiChannel:
    """Channel count impact at 512x512 bilinear."""

    def test_1ch(self, benchmark):
        src = _make_src(512, 512, 1)
        mx, my = _make_maps(512, 512)
        benchmark(remap_tensor, src, mx, my, interpolation=1)

    def test_3ch(self, benchmark):
        src = _make_src(512, 512, 3)
        mx, my = _make_maps(512, 512)
        benchmark(remap_tensor, src, mx, my, interpolation=1)

    def test_7ch(self, benchmark):
        src = _make_src(512, 512, 7)
        mx, my = _make_maps(512, 512)
        benchmark(remap_tensor, src, mx, my, interpolation=1)
