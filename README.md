# UnitField

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Build Status](https://github.com/Grayjou/UnitField/actions/workflows/ci.yml/badge.svg)](https://github.com/Grayjou/UnitField/actions)

**UnitField** is a high-performance N-dimensional coordinate remapping library built on a Cython kernel with OpenMP parallelism. It maps unit-space coordinates ([0, 1]) through arbitrary displacement fields with configurable interpolation and border handling — purpose-built for image warping, morphing, and nonlinear coordinate transformations.

## Why UnitField?

- **Cython-accelerated kernel** — 2D and 1D remap loops compiled to C with OpenMP threading. Significantly faster than pure NumPy for large images.
- **Asymmetric per-edge feathering** — Control feather blend independently on left, right, top, and bottom borders. Useful for seamless compositing and panorama blending.
- **Per-channel feather masks** — Feather only specific channels (e.g., alpha-only) via the `feather_dims` parameter.
- **Multiple border modes** — CLAMP, CONSTANT, REFLECT, WRAP, REFLECT_101, and ARRAY compositing.
- **Multiple interpolation methods** — Nearest-neighbor, bilinear, bicubic (Catmull-Rom), Lanczos-3/4.
- **1-D signal remapping** — The same kernel operates on 1-D signals, useful for audio, time-series, and look-up table applications.
- **Endomorphism composition** — `Unit2DMappedEndomorphism` supports composition (`f ∘ g`) for chaining transformations.
- **Endomorphism interpolation** — `Unit1DEndoInterpolator` and `Unit2DEndoInterpolator` blend transforms over time, even when endpoint maps use different sampled sizes.
- **Fast nearest map resize** — `resize_unit_field_nearest` upsamples or downsamples sampled remap fields for low-resolution interactive previews.

## Installation

```bash
pip install unitfield
```

For the fastest installation with a pre-built wheel, ensure you have the `cv2` extras (optional — used for comparison benchmarks only):

```bash
pip install "unitfield[cv2]"
```

### From Source (Cython)

```bash
git clone https://github.com/Grayjou/UnitField.git
cd UnitField
pip install -e ".[dev]"
```

Requires: Python ≥ 3.10, NumPy ≥ 1.20, Cython ≥ 3.0 (for source builds), a C99 compiler with OpenMP support.

## Quick Start

### 2-D Image Remapping

```python
import numpy as np
from unitfield import (
    BorderConfig, BorderMode, InterpMethod,
    Unit2DMappedEndomorphism, remap_tensor,
)

# Simple identity field
H, W = 256, 256
xs, ys = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H), indexing="xy")
identity = np.stack([xs, ys], axis=-1)
endo = Unit2DMappedEndomorphism(identity, interp_method=InterpMethod.LINEAR)

# Remap an image with asymmetric feathering
bc = BorderConfig(
    mode=BorderMode.CONSTANT,
    constant_value=0.0,
    feathering_width=0.2,
    feathering_x_overshoot_multiplier=3.0,   # heavy feather on right
    feathering_x_undershoot_multiplier=0.0,   # hard edge on left
    feather_dims=[True, True, True, False],   # RGB feathers, alpha hard
)
result = endo.remap(image, interpolation=1, border_config=bc)
```

### Direct remap with coordinate maps

```python
map_x = np.random.rand(H, W).astype(np.float64)
map_y = np.random.rand(H, W).astype(np.float64)

result = remap_tensor(
    image, map_x, map_y,
    interpolation=1,
    border_config=BorderConfig.constant(0.0, feathering_width=0.1),
)
```

### Fast Preview From A Low-Resolution Map

```python
from unitfield import InterpMethod, remap_tensor, resize_unit_field_nearest

# low_res_map shape: (preview_h, preview_w, 2), storing x/y unit coordinates
full_res_map = resize_unit_field_nearest(
    low_res_map,
    width=image.shape[1],
    height=image.shape[0],
)

preview = remap_tensor(
    image,
    full_res_map[..., 0],
    full_res_map[..., 1],
    interpolation=InterpMethod.NEAREST_MANHATTAN,
)
```

`resize_unit_field_nearest` is the preferred helper for preview remap fields. `resize_nearest` is also available for generic 2-D or 3-D arrays when a nearest resize is needed directly.

### 1-D Signal Remapping

```python
from unitfield import remap_tensor_1d

signal = np.sin(np.linspace(0, 4 * np.pi, 1000))
map_x = np.linspace(0, 1, 800) ** 2  # nonlinear time warp
warped = remap_tensor_1d(signal, map_x, interpolation=1)
```

### Endomorphism Interpolation

```python
import numpy as np
from unitfield import Unit1DEndoInterpolator

ease = np.array([0.0, 0.125, 0.25, 0.5625, 1.0])
linear = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

interpolator = Unit1DEndoInterpolator(ease, linear)
halfway = interpolator.interpolate(t=0.5)

value = halfway.get_value((0.5,))[0]
remapped_signal = halfway.remap(signal, interpolation=1)
```

Endpoint maps can be raw 1-D arrays, `Unit1DMappedEndomorphism`, `Unit2DMappedEndomorphism`, or generic `UnitMappedEndomorphism` instances. If endpoint maps use different sampled sizes, both are evaluated on the requested output grid before blending.

## Asymmetric Feathering

`BorderConfig` now exposes four independent feather multipliers — one per edge:

| Field | Edge | Applies when |
|-------|------|-------------|
| `feathering_x_undershoot_multiplier` | left | `u_x < 0.0` |
| `feathering_x_overshoot_multiplier` | right | `u_x > 1.0` |
| `feathering_y_undershoot_multiplier` | top | `u_y < 0.0` |
| `feathering_y_overshoot_multiplier` | bottom | `u_y > 1.0` |

All default to `1.0`. Set to `0.0` for a hard edge, or higher for a softer blend.

## API Overview

| Module | Key exports |
|--------|-------------|
| `unitfield` | `BorderConfig`, `BorderMode`, `InterpMethod`, `remap_tensor`, `remap_tensor_1d`, `resize_unit_field_nearest`, `resize_nearest`, `Unit2DMappedEndomorphism`, `Unit1DMappedEndomorphism`, `Unit1DEndoInterpolator`, `Unit2DEndoInterpolator`, `MappedUnitField` |
| `unitfield.core` | Same + `UnitNdimField`, `UnitMappedEndomorphism`, `UnitEndomorphismInterpolator`, `UnitArray`, `UnitSpaceVector` |
| `unitfield.utilities` | `pbm_2d`, `upbm_2d`, `flat_1d_pbm` — positional basematrix generators |

## Performance

The kernel is written in Cython with:
- **OpenMP-accelerated** inner loops (`prange`)
- **No Python overhead** at runtime (`nogil`)
- **Bicubic (Catmull-Rom)** and **Lanczos** interpolation with efficient separable sampling
- **Per-edge feather distance** computed inline with the border handler

Run benchmarks locally:

```bash
pytest tests/ -v -m benchmark
```

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT — see [LICENSE](LICENSE).

## Citation

```bibtex
@software{unitfield2026,
  author = {GrayJou},
  title = {UnitField: N-dimensional Unit Field Transformations},
  year = {2026},
  url = {https://github.com/Grayjou/UnitField},
}
```
