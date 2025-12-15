# UnitField

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**UnitField** is a high-performance Python library for N-dimensional unit field transformations with interpolation. It provides efficient tools for mapping unit-space coordinates (values in [0, 1]) to transformed coordinates, with support for various interpolation methods and optimized backends.

## Features

- **N-dimensional unit field transformations** with flexible interpolation
- **Multiple interpolation methods**: Nearest neighbor (Manhattan/Euclidean), Linear, Cubic, Lanczos4
- **Dual backends**: NumPy for N-dimensional fields, OpenCV for optimized 2D operations
- **2D image remapping** with endomorphism composition
- **Type-safe** with comprehensive type hints
- **Well-tested** with extensive test coverage
- **Performance optimized** with LRU caching and vectorized operations

## Installation

### From Source

```bash
git clone https://github.com/Grayjou/UnitField.git
cd UnitField
pip install -e .
```

### Requirements

- Python >= 3.8
- NumPy >= 1.20.0
- OpenCV (cv2) >= 4.5.0
- boundednumbers >= 0.1.0
- typing-extensions >= 4.0.0

## Quick Start

### Basic Unit Field Usage

```python
import numpy as np
from unitfield.core.unitfield import MappedUnitField
from unitfield.core.enums import InterpMethod

# Create a 2D unit field (5x5 grid mapping to 2D vectors)
x, y = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
data = np.stack([x, y], axis=-1)

# Create the field with linear interpolation
field = MappedUnitField(data=data, interp_method=InterpMethod.LINEAR)

# Query single coordinate
result = field.get_value((0.5, 0.5))
print(f"Value at (0.5, 0.5): {result}")

# Query multiple coordinates
coords = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
results = field.get_values(coords)
print(f"Batch results shape: {results.shape}")
```

### 2D Image Remapping with Endomorphisms

```python
import numpy as np
import cv2
from unitfield.core.unitfield import Unit2DMappedEndomorphism
from unitfield.core.enums import InterpMethod

# Create an identity endomorphism
height, width = 100, 100
xs, ys = np.meshgrid(
    np.linspace(0, 1, width),
    np.linspace(0, 1, height),
    indexing='xy'
)
identity_data = np.stack([xs, ys], axis=-1)

# Create endomorphism with cubic interpolation
endo = Unit2DMappedEndomorphism(
    data=identity_data,
    interp_method=InterpMethod.CUBIC
)

# Remap an image
image = cv2.imread('input.jpg')
remapped = endo.remap(image, interpolation=cv2.INTER_LINEAR)

# Compose two endomorphisms
endo2 = Unit2DMappedEndomorphism(data=other_data)
composed = endo.compose(endo2)
```

## API Reference

### Core Classes

#### `MappedUnitField`

N-dimensional unit field with interpolation.

**Parameters:**
- `data` (UnitArray): N+1 dimensional array of shape (*spatial_dims, N)
- `interp_method` (InterpMethod): Interpolation strategy (default: NEAREST_MANHATTAN)
- `cache_size` (int, optional): LRU cache size for single queries (default: 128)

**Methods:**
- `get_value(coords)`: Get value at single coordinate
- `get_values(coords_array)`: Get values at multiple coordinates
- `with_interp_method(method)`: Create copy with different interpolation

#### `Unit2DMappedEndomorphism`

2D unit field endomorphism with optimized OpenCV backend.

**Parameters:**
- `data` (UnitArray): 3-dimensional array of shape (H, W, 2)
- `interp_method` (InterpMethod): Interpolation strategy
- `cache_size` (int, optional): LRU cache size

**Methods:**
- `get_value(coords)`: Get single coordinate value
- `get_values(coords_array)`: Get multiple coordinate values
- `rasterize_mapping(width, height)`: Convert to pixel-space mapping
- `remap(data)`: Remap arbitrary (H, W, J) array
- `compose(other)`: Compose with another endomorphism

### Interpolation Methods

```python
from unitfield.core.enums import InterpMethod

# Available methods:
InterpMethod.NEAREST_MANHATTAN  # Nearest neighbor (Manhattan distance)
InterpMethod.NEAREST_EUCLIDEAN  # Nearest neighbor (Euclidean distance)
InterpMethod.LINEAR             # Linear/bilinear interpolation
InterpMethod.CUBIC              # Cubic/bicubic interpolation
InterpMethod.LANCZOS4           # Lanczos-4 interpolation
```

### Utility Functions

#### `remap_tensor_cv2`

Remap arbitrary tensors using pixel-space mappings.

```python
from unitfield.core.unitfield import remap_tensor_cv2

result = remap_tensor_cv2(
    data=tensor,
    mapping=pixel_mapping,
    interpolation=cv2.INTER_LINEAR,
    border_mode=cv2.BORDER_REPLICATE,
    border_value=0.0
)
```

## Coordinate System

UnitField operates in **unit space** where coordinates are in the range [0, 1]:
- `0.0` represents the start of each dimension
- `1.0` represents the end of each dimension
- Coordinates are automatically scaled to array indices

### Coordinate Constraints and Behavior

**Important Limitations:**

1. **Infinite and NaN coordinates are NOT supported** and will produce undefined behavior. This is by design to keep the functions simple and avoid overhead:
   ```python
   # ❌ AVOID - undefined behavior
   field.get_value((np.inf, 0.5))  
   field.get_value((np.nan, 0.5))  
   ```

2. **Out-of-bounds coordinates** (< 0 or > 1) are handled via clipping to [0, 1] range:
   ```python
   # ✓ OK - will be clipped to valid range
   field.get_value((-0.5, 1.5))  # Treated as (0.0, 1.0)
   ```

3. **Why this design?** 
   - Unit field transformations are intended for normalized coordinate spaces
   - Checking for inf/NaN on every coordinate adds unnecessary overhead
   - Out-of-bounds handling beyond [0, 1] is simple to implement externally if needed
   - Keeps the core functions fast and focused

**Best Practices:**
- Validate coordinates before passing to UnitField methods if they may contain special values
- For out-of-bounds behavior beyond simple clipping, preprocess coordinates externally

## Examples

### Creating Different Field Types

```python
# 1D field
data_1d = np.linspace(0, 1, 100).reshape(-1, 1)
field_1d = MappedUnitField(data=data_1d, interp_method=InterpMethod.LINEAR)

# 3D field
data_3d = np.random.rand(10, 10, 10, 3)
field_3d = MappedUnitField(data=data_3d, interp_method=InterpMethod.CUBIC)
```

### Switching Interpolation Methods

```python
# Create field with one method
field = MappedUnitField(data=data, interp_method=InterpMethod.LINEAR)

# Switch to another method
cubic_field = field.with_interp_method(InterpMethod.CUBIC)
```

### Batch Processing

```python
# Process large batches efficiently
batch_size = 1000
coords = np.random.rand(batch_size, 2)
results = field.get_values(coords)
```

### Image Warping Example

```python
import numpy as np
import cv2
from unitfield.core.unitfield import Unit2DMappedEndomorphism

# Create a simple distortion field (barrel distortion)
h, w = 256, 256
y, x = np.ogrid[:h, :w]
center_x, center_y = w / 2, h / 2

# Calculate distance from center
dx = (x - center_x) / center_x
dy = (y - center_y) / center_y
r = np.sqrt(dx**2 + dy**2)

# Apply barrel distortion
distortion = 1 + 0.3 * r**2
new_x = center_x + dx * distortion * center_x
new_y = center_y + dy * distortion * center_y

# Normalize to unit space
unit_x = new_x / (w - 1)
unit_y = new_y / (h - 1)
distortion_field = np.stack([unit_x, unit_y], axis=-1).astype(np.float32)

# Create endomorphism and apply
endo = Unit2DMappedEndomorphism(data=distortion_field)
image = cv2.imread('input.jpg')
warped = endo.remap(image)
cv2.imwrite('output.jpg', warped)
```

## Performance Tips

1. **Use caching for repeated queries**: Set appropriate `cache_size` when creating fields
2. **Batch operations**: Use `get_values()` instead of multiple `get_value()` calls
3. **Choose the right backend**: Use `Unit2DMappedEndomorphism` for 2D operations (faster via OpenCV)
4. **Appropriate interpolation**: Nearest neighbor is fastest, cubic/Lanczos are slower but smoother

## Development

### Running Tests

```bash
pip install pytest pytest-benchmark
pytest tests/ -v
```

### Code Style

This project follows modern Python conventions:
- PEP 8 style guide
- Type hints throughout
- Comprehensive docstrings (Google style)

## Known Limitations

1. **Inf/NaN coordinates**: Not supported in remapping functions (see Coordinate Constraints section)
2. **Memory usage**: Large N-dimensional fields may consume significant memory
3. **2D optimization**: Only 2D endomorphisms benefit from OpenCV backend optimization

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use UnitField in your research, please cite:

```bibtex
@software{unitfield2025,
  author = {GrayJou},
  title = {UnitField: N-dimensional Unit Field Transformations},
  year = {2025},
  url = {https://github.com/Grayjou/UnitField}
}
```

## Acknowledgments

- Built with NumPy and OpenCV
- Inspired by coordinate transformation needs in computer vision and graphics

## Support

For issues, questions, or contributions, please visit:
- **Issues**: https://github.com/Grayjou/UnitField/issues
- **Discussions**: https://github.com/Grayjou/UnitField/discussions
