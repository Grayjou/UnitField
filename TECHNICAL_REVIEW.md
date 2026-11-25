# Technical Review

## Release Recommendation

Recommend version `0.3.0`. Independent destination dimensions, new public
utilities, and explicit coordinate contracts are additive. The rasterizer dtype
correction is a narrow behavioral fix for callers that passed `float64` but
received `float32`; it is documented in the migration notes.

Review base commit: `dddbfee4cec87408184e5300b983b56567bb3ce8`.

## Disposition

| Request | Original observation | Verification result | Decision | Implementation | Tests | Compatibility impact | Remaining limitations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Rasterization | Rectangular identity transposed | Confirmed | Implemented | Query grid now uses `(y, x)` and separately sized source scaling | `test_coordinate_contracts.py` | Corrects erroneous output | Query order differs from field vector order by design |
| Coordinate conventions | Ambiguous array/geometry order | Confirmed | Implemented | Docs, type comments, explicit converters | Coordinate contract tests | Additive | Generic N-D vectors remain application-defined |
| Composition | `A.compose(B)` evaluates `B(A(q))` | Confirmed | Implemented | Documented unchanged | Noncommuting test | None | Composed-field and sequential remap differ after discrete interpolation |
| Independent dimensions | Source/map shapes must match | Confirmed | Implemented | 1-D/2-D kernels separate source and destination sizes | Remap tests | Existing same-size calls unchanged | Source must remain nonempty when destination is nonempty |
| Conversion utilities | No explicit public conversion suite | Partially supported | Implemented | Pixel/unit, axis order, absolute/displacement, identity utilities | Round-trip tests | Additive | Endpoint-aligned convention only |
| Field resampling | Nearest-only public helper | Confirmed | Implemented | `resample_coordinate_field` evaluates fields with selected interpolator | Affine test | Additive | 2-D absolute coordinate fields only |
| Nearest ties | Half-integer difference | Confirmed | Documented | Existing half-up rule retained | Exact tie test | None | No selectable ties-to-even policy |
| Precision/memory | Float64 conversion overhead | Confirmed | Deferred | Measured, no speculative kernel split | Benchmarks | None | Float32 sources still use float64 work buffers |
| NumPy fallback | Compiled extension required | Confirmed | Deferred | Design rationale recorded | N/A | None | Source builds require Cython/compiler/OpenMP |
| Packaging/CI | Claimed versions not all CI-tested | Confirmed | Partially implemented | CI matrix covers 3.10-3.14; local wheel smoke test added to release verification | Local wheel smoke test | Broader CI time | Remote Actions status requires GitHub authentication |
| Partial fields | Validity/coverage semantics unclear | Confirmed | Deferred | Design note | N/A | None | No masked interpolation/composition API |
| Differential diagnostics | Jacobian API unspecified | Confirmed | Deferred | Design note | N/A | None | No diagnostics module |

## Additional Findings

- `num_threads` was computed but not passed to `prange`; this is corrected.
- Field-query clipping was documented but not applied for linear/cubic/Lanczos;
  queries are now clipped before interpolation.
- Linear field interpolation had no singleton-axis contract; it now uses the
  sole sample on singleton axes.
- Rasterizing a nonidentity field to a singleton source axis previously left a
  normalized coordinate in pixel-space output; it now produces zero.
- The OpenCV comparison backend was imported despite being optional. Package
  import now works without OpenCV and the backend reports an installation hint.
- The current kernels still differ from some graphics-library border semantics:
  higher-order interpolation edge taps clamp, and reflect variants are not
  independently characterized. These are existing behavior and require a
  dedicated border-semantics release rather than an untested compatibility
  change.

## Verification Commands

```powershell
python setup.py build_ext --inplace
python -m pytest
ruff check unitfield tests benchmarks
pytest tests/test_benchmark_remap.py -m benchmark
python -m build --wheel --no-isolation
```
