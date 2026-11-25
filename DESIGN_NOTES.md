# Design Notes

## Partially Defined Coordinate Fields

Partially defined mappings should remain a separate layer from ordinary
`MappedUnitField` instances. A useful future representation is a coordinate
array plus a boolean validity mask and optional nonnegative coverage array.
Coordinates select source positions; validity says whether a mapping exists;
coverage is fractional occupancy. Neither is equivalent to an out-of-bounds
coordinate or a source border mode.

For linear interpolation with nonnegative weights, valid/covered samples could
be normalized as `sum(weight * coverage * coordinate) / sum(weight * coverage)`.
No contributing sample should produce an explicit invalid result rather than an
ordinary coordinate sentinel. Cubic and Lanczos need a separately specified
policy because their weights can be negative. Composition should combine
validity from the first field with validity of the second field evaluated at the
first field's coordinate; coverage composition is application-dependent.

Decision: deferred. The policy is valuable but cannot be added to existing
field classes without changing their value contracts, interpolation APIs,
composition behavior, and remapping output semantics. Consumers can currently
maintain masks independently and select a border behavior explicitly.

## Differential Diagnostics

A future diagnostics module can evaluate finite-difference Jacobians for
sampled absolute coordinate fields. For `field[y, x] = (Fx, Fy)`, derivatives
must use geometric axes, yielding `[[dFx/dx, dFx/dy], [dFy/dx, dFy/dy]]`.
Normalized-grid derivatives need scaling by `(width - 1, height - 1)` to report
derivatives in pixel coordinates; singleton axes have no derivative and should
report `NaN` or an explicit undefined mask. The determinant indicates local
area scaling and orientation, but does not prove global invertibility.

Useful initial outputs are Jacobian, determinant, displacement magnitude,
orientation-reversal mask, and 1-D derivative/monotonicity masks. They should
be diagnostic arrays only, not image-specific rendering features.

Decision: deferred. The API requires a clear choice between normalized and
physical units, boundary stencils, and invalid/singleton representation. It is
not required for the corrected remapping contract.

## NumPy Reference Backend

A remap fallback was not added. Existing NumPy field interpolation does not
match kernel support for border/feather behavior or Lanczos-3. A fallback must
be a dedicated reference implementation with parity tests; silently selecting
the existing interpolator would change semantics. Source installations continue
to require a Cython build toolchain and OpenMP-capable C compiler.
