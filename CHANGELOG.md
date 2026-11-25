# Changelog

All notable changes to UnitField will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- `Unit2DMappedEndomorphism.rasterize_mapping()` now evaluates sampled fields
  in array-axis `(y, x)` order and returns correct rectangular mappings.
- `rasterize_mapping(dtype=...)` now honors its documented output dtype.
- Linear interpolation of sampled fields handles singleton spatial axes.
- `num_threads` now controls the OpenMP worker count in both remap kernels.

### Added
- 1-D and 2-D remappers support independent source and destination resolutions;
  coordinate-map shape now defines destination shape.
- `output_shape` and `output_length` on 2-D and 1-D mapped endomorphism
  remapping convenience methods.
- Public coordinate conversion, absolute/displacement conversion, identity-map,
  and interpolating coordinate-field resampling utilities.

### Migration Notes
- Equal source/map dimensions remain supported. Existing callers need no change.
- Array-valued borders are destination-aligned and must match the coordinate-map
  shape when source and destination dimensions differ.
- A caller that relied on `rasterize_mapping(dtype=np.float64)` returning
  `float32` now receives `float64`, matching the parameter contract.

## [0.2.12] - 2026-06-29

### Added
- `resize_unit_field_nearest` for endpoint-aligned nearest-neighbor resizing of sampled unit fields/maps shaped `(height, width, channels)`.
- `resize_nearest` for endpoint-aligned nearest-neighbor resizing of 2-D and 3-D arrays.
- Cython source kernel for nearest resize, with vectorized NumPy fallback when the extension has not been built yet.
- Tests covering upscaling, downscaling, channel preservation, dtype preservation/conversion, contiguous output, and validation errors.

### Notes
- The intended preview workflow is to evaluate a remap field at low resolution, resize the field to the viewport/document shape, then call `remap_tensor(..., interpolation=InterpMethod.NEAREST_MANHATTAN)`.
- This prioritizes resizing UnitField maps over owning a general-purpose image resize feature.
- This release is intended to support smoother Chromatica2 drag-preview rendering.

## [0.2.11] - 2026-06-22

### Added
- `UnitEndomorphismInterpolator` for interpolating between mapped unit endomorphisms.
- `Unit1DEndoInterpolator` and `Unit2DEndoInterpolator` convenience classes, plus lowercase aliases `Unit1dEndoInterpolator` and `Unit2dEndoInterpolator`.
- Raw 1-D array/list inputs for 1-D endomorphism interpolation, automatically coerced to `Unit1DMappedEndomorphism`.
- Interpolation across endpoint maps with different sampled sizes by evaluating both endpoints on the requested output grid before blending.
- Public exports from both `unitfield` and `unitfield.core`.

### Notes
- No Cython kernel changes are required for this feature. It uses the existing mapped endomorphism sampling path and returns normal endomorphism instances that support `get_value`, `get_values`, and `remap`.
- This release is intended to support time-varying transforms in downstream tools such as Chromatica2.

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
[0.2.11]: https://github.com/Grayjou/UnitField/releases/tag/v0.2.11
[0.2.12]: https://github.com/Grayjou/UnitField/releases/tag/v0.2.12
