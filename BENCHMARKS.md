# Benchmark Notes

Measured on Windows, CPython 3.14.0, NumPy available in the development
environment, with the in-place Cython/OpenMP extensions built by
`python setup.py build_ext --inplace`. Values below are the mean of three
bilinear `remap_tensor(..., num_threads=1)` calls after one warm-up. Peak memory
comes from `tracemalloc`; it excludes allocator and native OpenMP overhead.

| Source | Destination | Source dtype | Mean seconds | Peak MiB |
| --- | --- | --- | ---: | ---: |
| `(512, 512)` contiguous | `(640, 768)` | `float64` | 0.004776 | 7.50 |
| `(512, 512)` contiguous | `(640, 768)` | `float32` | 0.005534 | 9.50 |
| `(512, 512, 3)` contiguous | `(640, 768, 3)` | `float64` | 0.011814 | 22.50 |
| `(384, 512)` noncontiguous view | `(640, 768)` | `float64` | 0.004880 | 9.00 |

Reproduce with:

```powershell
python -c "import time, tracemalloc, numpy as np; from unitfield import remap_tensor; src=np.random.default_rng(0).random((512,512)); ys,xs=np.meshgrid(np.linspace(0,1,640),np.linspace(0,1,768),indexing='ij'); remap_tensor(src,xs,ys,interpolation=1,num_threads=1); tracemalloc.start(); start=time.perf_counter(); [remap_tensor(src,xs,ys,interpolation=1,num_threads=1) for _ in range(3)]; elapsed=(time.perf_counter()-start)/3; current,peak=tracemalloc.get_traced_memory(); tracemalloc.stop(); print(elapsed, peak/1024/1024)"
pytest tests/test_benchmark_remap.py -m benchmark
```

The one-line timing command used for these values is intentionally not a
benchmark framework; use the maintained `pytest-benchmark` suite for comparison
and regression tracking. No float32/fused-type optimization was added: every
path currently converts source and coordinates to contiguous float64, so a
specialized kernel needs demonstrated benefit and full parity coverage first.
