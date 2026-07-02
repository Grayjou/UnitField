# UnitField 0.2.12 Patch Notes

Release date: 2026-06-29

## Summary

UnitField 0.2.12 adds fast nearest-neighbor resizing for sampled unit fields/maps. This is aimed at interactive preview rendering: build or sample a low-resolution remap field, resize that field to the full viewport/document shape, then apply the remap with nearest sampling.

## What Changed

- Added `resize_unit_field_nearest(field, width, height, preserve_dtype=True)`.
- Added `resize_nearest(image, width, height, preserve_dtype=True)` as the underlying 2-D/3-D nearest resize helper.
- Added Cython source for the nearest resize kernel.
- Added a vectorized NumPy fallback for environments where the Cython extension has not been built yet.
- Exported the new helpers from both `unitfield` and `unitfield.core`.

## Chromatica2 Preview Workflow

Use this for low-resolution drag previews:

```python
from unitfield import InterpMethod, remap_tensor, resize_unit_field_nearest

# low_res_map shape: (preview_h, preview_w, 2)
# channel 0 = x unit coordinate, channel 1 = y unit coordinate
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

This avoids evaluating the expensive morph field at full resolution during every drag event while still producing a full-size texture for the UI.

## Behavior

- Supports upscaling and downscaling.
- Uses nearest-neighbor sampling only.
- Uses endpoint-aligned sampling to match UnitField's unit-space coordinate convention.
- Preserves channel count.
- Preserves dtype by default.
- Returns contiguous arrays.
- Rejects non-positive target dimensions.
- `resize_nearest` accepts arrays shaped `(height, width)` or `(height, width, channels)`.
- `resize_unit_field_nearest` requires `(height, width, channels)` and is the preferred map/field API.

## Compatibility

- This is an additive patch release.
- Existing remap and endomorphism APIs are unchanged.
- The Cython extension is optional at import time because the Python wrapper has a NumPy fallback.

## Verification

Validated with:

```bash
pytest tests/test_resize_nearest.py -q
pytest tests -q
```
