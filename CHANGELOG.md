# Changelog

All notable changes to UnitField will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-12-15

### Added - First Release Features

#### Core Functionality
- **N-dimensional unit field transformations** with flexible interpolation support
- **MappedUnitField** class for general N-dimensional unit field operations
- **UnitMappedEndomorphism** class for endomorphism transformations (same input/output dimensions)
- **Unit2DMappedEndomorphism** class for optimized 2D operations with OpenCV backend

#### Interpolation Methods
- **Nearest neighbor interpolation** with two distance metrics:
  - Manhattan distance (NEAREST_MANHATTAN)
  - Euclidean distance (NEAREST_EUCLIDEAN)
- **Linear interpolation** (bilinear for 2D, trilinear for 3D, etc.)
- **Cubic interpolation** for smoother results
- **Lanczos-4 interpolation** for high-quality resampling

#### Dual Backend System
- **NumPy backend** for N-dimensional field operations
- **OpenCV backend** for optimized 2D image operations
- Automatic backend selection based on dimensionality

#### Performance Features
- **LRU caching** for repeated single-coordinate queries (configurable cache size)
- **Vectorized operations** for batch coordinate processing
- **Efficient rasterization** for converting unit fields to pixel-space mappings

#### Type Safety
- Comprehensive type hints throughout the codebase
- Type aliases for clarity: `UnitArray`, `Coordinate`, `UnitSpaceVector`
- Full mypy compatibility

#### Image Processing
- **Image remapping** with arbitrary unit field transformations
- **Endomorphism composition** for combining transformations
- Support for arbitrary tensor shapes (H, W, C) in remapping
- Configurable border modes and interpolation methods

#### API Features
- `get_value(coords)` - Query single coordinate
- `get_values(coords_array)` - Batch query multiple coordinates
- `rasterize_mapping(width, height)` - Convert to pixel-space mapping
- `remap(data)` - Apply transformation to images/tensors
- `compose(other)` - Compose two endomorphisms
- `with_interp_method(method)` - Create field with different interpolation

#### Testing
- Comprehensive test suite with 100+ test cases
- Tests for all interpolation methods
- Edge case and boundary condition tests
- Backend comparison tests (NumPy vs OpenCV)
- Performance benchmarks
- Thread safety tests
- Serialization compatibility tests

#### Documentation
- Complete API documentation with docstrings
- Comprehensive README with examples
- Quick start guide
- Performance tips
- Known limitations documented

### Known Limitations

#### Coordinate Constraints
- **Inf and NaN coordinates are NOT supported** in remapping functions
  - This is intentional to avoid unnecessary overhead in performance-critical code
  - The functions are designed for normalized coordinate spaces [0, 1]
  - Users should validate coordinates externally if they may contain special values
  
- **Out-of-bounds behavior** (coordinates < 0 or > 1):
  - Coordinates are clipped to the [0, 1] range
  - This simple behavior is easy to understand and implement
  - More complex boundary handling can be implemented externally if needed
  
- **Rationale**: Unit field transformations are intended for normalized coordinate spaces where inf/NaN are not expected. Checking for these on every coordinate would add overhead without benefit for the intended use case.

#### Performance Considerations
- Large N-dimensional fields may consume significant memory
- Only 2D operations benefit from OpenCV backend optimization
- Higher-dimensional operations use NumPy backend exclusively

#### API Stability
- This is the first release (0.1.0), API may change in minor version updates
- Breaking changes will be documented in future releases

### Technical Details

#### Dependencies
- Python >= 3.8
- NumPy >= 1.20.0
- OpenCV (cv2) >= 4.5.0
- boundednumbers >= 0.1.0
- typing-extensions >= 4.0.0

#### Architecture
- Abstract base class `UnitNdimField` for extensibility
- Inheritance hierarchy: `MappedUnitField` → `UnitMappedEndomorphism` → `Unit2DMappedEndomorphism`
- Pluggable interpolation system via `InterpMethod` enum
- Separate backends for NumPy and OpenCV operations

#### Quality Assurance
- Type hints throughout
- Comprehensive input validation
- Detailed error messages
- Extensive test coverage
- Modern Python packaging with pyproject.toml

### Migration Notes

This is the first release, no migration needed.

### Future Considerations

Potential future enhancements (not yet implemented):
- GPU acceleration support
- Additional interpolation methods
- Sparse field representations
- Field compression/serialization
- JIT compilation for performance

---



---

[0.1.0]: https://github.com/Grayjou/UnitField/releases/tag/v0.1.0

# [0.1.2]

- Introduced utilities such as generation of positional basematrices with functions, which include unit normalization as well. 
- Introduced tests for the utilities. See tests/test_pbm.py

## [Unreleased]

No unreleased changes yet.