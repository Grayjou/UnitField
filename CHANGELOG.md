# Changelog

All notable changes to UnitField will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-06-15

### Added
- **Asymmetric per-edge feather multipliers** — `feathering_x_undershoot_multiplier`,
  `feathering_x_overshoot_multiplier`, `feathering_y_undershoot_multiplier`,
  `feathering_y_overshoot_multiplier` replace the old axis-level multipliers.
  Each edge can now be feathered independently.
- **1-D border feather multipliers** — the 1-D kernel now respects `under_x`/`over_x`
  multipliers (was missing them entirely — isotropic feathering regardless of config).

### Changed
- **Breaking:** `feathering_x_multiplier` and `feathering_y_multiplier` removed.
  Migrate to the four per-edge fields above.
- `_apply_border` signature changed from `(..., fx, fy)` to
  `(..., under_x, over_x, under_y, over_y)`.

## [0.1.3] - 2026-05-??

- Added `flat_1d_pbm` utility for 1-D unit positional basematrix generation.

## [0.1.2] - 2026-05-??

- Introduced positional basematrix utilities (`pbm_2d`, `upbm_2d`, `pbm_ndim`,
  `upbm_ndim`, `flat_1d_upbm`) with unit normalization.
- Added tests in `tests/test_pbm.py`.

## [0.1.0] - 2025-12-15

### Added — First Release

#### Core Functionality
- N-dimensional unit field transformations with flexible interpolation
- `MappedUnitField` for general N-dimensional unit field operations
- `UnitMappedEndomorphism` for endomorphism transformations
- `Unit2DMappedEndomorphism` with Cython-accelerated 2-D remap kernel

#### Interpolation
- Nearest-neighbor (Manhattan / Euclidean distance)
- Linear (bilinear)
- Cubic (Catmull-Rom bicubic)
- Lanczos-3 / Lanczos-4

#### Performance
- Cython kernel with OpenMP parallelism
- Configurable thread count
- LRU caching for single-coordinate queries
- Vectorized batch processing

#### Border Handling
- CLAMP, CONSTANT, REFLECT, WRAP, REFLECT_101, ARRAY modes
- Smooth feather blending with configurable width
- Per-channel `feather_dims` mask (e.g., alpha-only feathering)

[0.1.0]: https://github.com/Grayjou/UnitField/releases/tag/v0.1.0
[0.1.2]: https://github.com/Grayjou/UnitField/releases/tag/v0.1.2
[0.1.3]: https://github.com/Grayjou/UnitField/releases/tag/v0.1.3
[0.2.0]: https://github.com/Grayjou/UnitField/releases/tag/v0.2.0
