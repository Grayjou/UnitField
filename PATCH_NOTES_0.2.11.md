# UnitField 0.2.11 Patch Notes

Release date: 2026-06-22

## Summary

UnitField 0.2.11 adds interpolation for unit endomorphisms. This makes it possible to represent time-varying coordinate transforms, such as blending from one 1-D or 2-D remap field to another, while still returning normal UnitField endomorphism objects that can be sampled or used for remapping.

## What Changed

- Added `UnitEndomorphismInterpolator` for generic mapped unit endomorphisms.
- Added `Unit1DEndoInterpolator` for 1-D transforms.
- Added `Unit2DEndoInterpolator` for 2-D transforms.
- Added aliases `Unit1dEndoInterpolator` and `Unit2dEndoInterpolator`.
- Exported the new interpolators from both `unitfield` and `unitfield.core`.
- Added support for raw 1-D array/list endpoint maps.
- Added support for endpoint maps with different sampled sizes.

## Chromatica2 Integration Notes

This is intended for animated or evolving transforms. Chromatica2 can keep transform endpoints as unit endomorphisms and request an intermediate transform at a normalized time parameter.

```python
import numpy as np
from unitfield import Unit1DEndoInterpolator

start = np.array([0.0, 0.125, 0.25, 0.5625, 1.0])
end = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

interpolator = Unit1DEndoInterpolator(start, end)
endo = interpolator.interpolate(t=0.5)

value = endo.get_value((0.5,))[0]
warped = endo.remap(signal, interpolation=1)
```

For 2-D transforms, use `Unit2DEndoInterpolator` with `Unit2DMappedEndomorphism` instances or raw arrays shaped `(height, width, 2)`.

## Behavior

- `t` must be finite and in `[0, 1]`.
- If `output_shape` is omitted, the interpolated endomorphism uses the first endpoint's sampled shape.
- If endpoint maps have different sampled sizes, both are sampled over the output grid and then linearly blended.
- The result is a normal `Unit1DMappedEndomorphism`, `Unit2DMappedEndomorphism`, or `UnitMappedEndomorphism`.
- No new Cython-level feature is required for mapped endomorphism interpolation.

## Compatibility

- This is a patch-level additive release.
- Existing endomorphism, remap, and Cython kernel APIs remain unchanged.
- Existing imports continue to work.

## Verification

Validated with the full test suite:

```bash
pytest tests -q
```
