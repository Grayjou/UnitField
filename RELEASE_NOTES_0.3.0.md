# UnitField 0.3.0 Release Notes

Release date: 2026-11-25

## Summary

UnitField 0.3.0 corrects 2-D mapping rasterization, supports independently
sampled source and destination grids, and makes coordinate-field contracts
explicit for interoperable numerical remapping.

## Highlights

- `remap_tensor` and `remap_tensor_1d` now use coordinate-map shape as the
  destination shape, independent of source dimensions.
- `rasterize_mapping(width, height, source_shape=(Hs, Ws))` emits source pixel
  coordinates for an independently sized source grid. Its default behavior is
  unchanged when `source_shape` is omitted.
- Singleton source axes rasterize to pixel coordinate zero, matching
  `unit_to_pixel`.
- Added public unit/pixel, axis-order, identity-map, displacement, and
  coordinate-field resampling utilities.
- `compose` retains its existing `other(self(q))` direction and now documents
  clipping of out-of-domain intermediate sampled coordinates.

## Migration

- Equal source and destination shapes continue to work unchanged.
- Array-valued borders must match the destination coordinate-map shape.
- `rasterize_mapping(dtype=np.float64)` now returns `float64` as documented.
- Coordinate utilities reject nonfinite values. Identity-map and requested
  resampling output dtypes must be floating point; integer coordinate vectors
  remain accepted and are converted to `float64`.
- OpenCV remains optional; importing the core package no longer requires it.

## Verification

```powershell
python setup.py build_ext --inplace
python -m pytest
ruff check unitfield tests benchmarks
python -m build --wheel --no-isolation
```
