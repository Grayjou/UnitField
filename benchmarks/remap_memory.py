"""Measure remap time plus Python-tracked retained and temporary memory."""

from __future__ import annotations

import gc
import platform
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unitfield import remap_tensor  # noqa: E402


def measure(call):
    """Return one result and allocations retained by it versus temporary allocations."""
    gc.collect()
    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    result = call()
    after_current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    retained = after_current - before_current
    temporary = max(peak - after_current, 0)
    return result, retained, temporary


def main() -> None:
    rng = np.random.default_rng(0)
    source = rng.random((512, 512), dtype=np.float64)
    destination_shape = (640, 768)
    map_y, map_x = np.meshgrid(
        np.linspace(0.0, 1.0, destination_shape[0]),
        np.linspace(0.0, 1.0, destination_shape[1]),
        indexing="ij",
    )
    def call():
        return remap_tensor(source, map_x, map_y, interpolation=1, num_threads=1)

    call()  # Warm-up excludes Cython import and OpenMP startup from timing.
    start = time.perf_counter()
    for _ in range(3):
        output = call()
    mean_seconds = (time.perf_counter() - start) / 3
    del output

    output, retained, temporary = measure(call)
    print(f"Python: {sys.version.split()[0]}")
    print(f"NumPy: {np.__version__}")
    print(f"Platform: {platform.platform()}")
    print(f"Source: {source.shape} {source.dtype}; destination: {output.shape} {output.dtype}")
    print(f"Mean seconds (3 calls): {mean_seconds:.6f}")
    print(f"Retained output MiB: {retained / 1024 / 1024:.2f}")
    print(f"Temporary Python-tracked MiB: {temporary / 1024 / 1024:.2f}")
    print("tracemalloc excludes native allocator and OpenMP runtime allocations.")


if __name__ == "__main__":
    main()
