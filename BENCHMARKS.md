# Benchmark Notes

Measured on Windows 11, CPython 3.14.0, NumPy 2.2.6, with the in-place
Cython/OpenMP extensions built by `python setup.py build_ext --inplace`.
Values below are the mean of three bilinear `remap_tensor(..., num_threads=1)`
calls after one warm-up.

| Source | Destination | Source dtype | Mean seconds | Retained output MiB | Temporary Python-tracked MiB |
| --- | --- | --- | ---: | ---: | ---: |
| `(512, 512)` contiguous | `(640, 768)` | `float64` | 0.004961 | 3.75 | 0.00 |

`tracemalloc` reports Python-tracked allocations only. The benchmark script
reports both retained output bytes after the call and peak temporary bytes
(`peak - current_after_call`), so the output allocation is not incorrectly
reported as a temporary buffer. Native allocator and OpenMP runtime allocations
remain outside this measurement.

Reproduce with:

```powershell
python benchmarks/remap_memory.py
pytest tests/test_benchmark_remap.py -m benchmark
```

Use the maintained `pytest-benchmark` suite for comparison and regression
tracking. No float32/fused-type optimization was added: every path currently
converts source and coordinates to contiguous float64, so a specialized kernel
needs demonstrated benefit and full parity coverage first.
