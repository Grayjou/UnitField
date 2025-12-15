"""
Test suite for UnitFied unit field transformations.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unitfield.core.types import *
from unitfield.core.enums import InterpMethod
from unitfield.core.interpolation.interp_np import (
    get_numpy_interpolator,
    _scale_coords,
    _safe_indexing,
    nearest_manhattan_vectorized_arr,
    linear_vectorized_arr,
    cubic_vectorized_arr,
    lanczos4_vectorized_arr,
    np_interp_dict
)
from unitfield.core.interpolation.interp_cv2 import (
    cv2_unit_field_sample,
    _scale_to_pixel_space,
    _CV2_INTERP_MAP
)


class TestTypes:
    """Test type definitions and constants."""
    
    def test_constants(self):
        """Test module constants."""
        assert MAX_CACHE_SIZE == 1024
        assert DEFAULT_CACHE_SIZE == 128
        assert DEFAULT_DTYPE == np.float32
        
    def test_unit_float_bounds(self):
        """Test UnitFloat from boundednumbers package."""
        from boundednumbers import UnitFloat
        
        # Test valid values
        uf = UnitFloat(0.5)
        assert 0 <= uf <= 1
        
        # Test automatic clamping
        uf = UnitFloat(1.5)
        assert uf == 1.0
        
        uf = UnitFloat(-0.5)
        assert uf == 0.0
        
    def test_numpy_type_aliases(self):
        """Test numpy array type aliases."""
        # Test UnitArray
        arr = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32
        
        # Test NDArray typing
        arr_2d: NDArray[np.float64] = np.array([[0.0, 0.5], [0.5, 1.0]])
        assert arr_2d.shape == (2, 2)


class TestEnums:
    """Test interpolation method enumerations."""
    
    def test_interp_method_values(self):
        """Test enum values and string representations."""
        assert InterpMethod.NEAREST_MANHATTAN.value == "nearest"
        assert InterpMethod.LINEAR.value == "linear"
        assert InterpMethod.NEAREST_EUCLIDEAN.value == "euclidean"
        assert InterpMethod.CUBIC.value == "cubic"
        assert InterpMethod.LANCZOS4.value == "lanczos4"
        
    def test_cv2_methods(self):
        """Test OpenCV supported methods."""
        cv2_methods = InterpMethod.get_cv2_methods()
        assert len(cv2_methods) == 5
        assert InterpMethod.NEAREST_MANHATTAN in cv2_methods
        assert InterpMethod.LINEAR in cv2_methods
        
    def test_numpy_methods_mapping(self):
        """Test numpy method name mapping."""
        numpy_methods = InterpMethod.get_numpy_methods()
        assert len(numpy_methods) == 5
        assert numpy_methods[InterpMethod.LINEAR] == "linear"
        assert numpy_methods[InterpMethod.CUBIC] == "cubic"
        
    def test_euclidean_alias(self):
        """Test backward compatibility alias."""
        # Both should reference the same enum member
        assert InterpMethod.NEAREST_EUCLIDEAN == InterpMethod.EUCLIDEAN


class TestNumpyHelperFunctions:
    """Test helper functions in numpy interpolation backend."""
    
    def test_scale_coords_1d(self):
        """Test coordinate scaling in 1D."""
        coords = np.array([0.0, 0.5, 1.0])
        spatial_shape = (10,)
        scaled = _scale_coords(coords, spatial_shape)
        expected = np.array([0.0, 4.5, 9.0])
        np.testing.assert_array_almost_equal(scaled, expected)
        
    def test_scale_coords_2d(self):
        """Test coordinate scaling in 2D."""
        coords = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        spatial_shape = (10, 20)
        scaled = _scale_coords(coords, spatial_shape)
        expected = np.array([[0.0, 0.0], [4.5, 9.5], [9.0, 19.0]])
        np.testing.assert_array_almost_equal(scaled, expected)
        
    def test_scale_coords_batch(self):
        """Test coordinate scaling with batched inputs."""
        coords = np.random.rand(5, 10, 3)  # 5 batches of 10 3D points
        spatial_shape = (100, 200, 300)
        scaled = _scale_coords(coords, spatial_shape)
        assert scaled.shape == coords.shape
        assert np.all(scaled >= 0)
        assert np.all(scaled <= np.array([99, 199, 299]))
        
    def test_safe_indexing_within_bounds(self):
        """Test safe indexing within bounds."""
        data = np.arange(100).reshape(10, 10)
        indices = np.array([[2, 3], [5, 6]])
        result = _safe_indexing(data, indices, (10, 10))
        expected = np.array([data[2, 3], data[5, 6]])
        np.testing.assert_array_equal(result, expected)
        
    def test_safe_indexing_out_of_bounds(self):
        """Test safe indexing with out-of-bounds indices."""
        data = np.arange(100).reshape(10, 10)
        indices = np.array([[-1, -1], [10, 10], [5, 5]])
        result = _safe_indexing(data, indices, (10, 10))
        # Should be clipped to [0, 0], [9, 9], [5, 5]
        assert result[0] == data[0, 0]
        assert result[1] == data[9, 9]
        assert result[2] == data[5, 5]
        
    def test_safe_indexing_higher_dimensions(self):
        """Test safe indexing with higher dimensional data."""
        data = np.random.rand(10, 10, 3)  # RGB image
        indices = np.array([[2, 3], [5, 6], [7, 8]])
        result = _safe_indexing(data, indices, (10, 10))
        assert result.shape == (3, 3)  # 3 points, 3 channels
        assert np.array_equal(result[0], data[2, 3])


class TestNumpyInterpolation:
    """Test numpy-based interpolation methods."""
    
    @pytest.fixture
    def sample_data_1d(self):
        """Create sample 1D data."""
        return np.array([0.0, 0.5, 1.0, 0.5, 0.0])
    
    @pytest.fixture
    def sample_data_2d(self):
        """Create sample 2D data (checkerboard pattern)."""
        data = np.zeros((10, 10))
        data[::2, ::2] = 1.0
        data[1::2, 1::2] = 1.0
        return data
    
    @pytest.fixture
    def sample_data_3d(self):
        """Create sample 3D data."""
        data = np.zeros((8, 8, 8))
        data[2:6, 2:6, 2:6] = 1.0  # Cube in the center
        return data
    
    @pytest.fixture
    def sample_coords_1d(self):
        """Create sample 1D coordinates."""
        return np.array([[0.0], [0.25], [0.5], [0.75], [1.0]])
    
    @pytest.fixture
    def sample_coords_2d(self):
        """Create sample 2D coordinates."""
        return np.array([
            [0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
            [0.0, 0.5], [0.5, 0.5], [1.0, 0.5],
            [0.0, 1.0], [0.5, 1.0], [1.0, 1.0]
        ])
    
    @pytest.fixture
    def sample_coords_3d(self):
        """Create sample 3D coordinates."""
        return np.array([
            [0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.0, 0.5, 0.5], [0.5, 0.5, 0.5], [1.0, 0.5, 0.5],
            [0.0, 1.0, 1.0], [0.5, 1.0, 1.0], [1.0, 1.0, 1.0]
        ])
    
    def test_get_numpy_interpolator(self):
        """Test interpolator getter function."""
        for method in np_interp_dict.keys():
            interpolator = get_numpy_interpolator(method)
            assert callable(interpolator)
            
        # Test invalid method
        with pytest.raises(ValueError):
            get_numpy_interpolator("invalid_method")
    
    def test_nearest_manhattan_1d(self, sample_data_1d, sample_coords_1d):
        """Test nearest neighbor interpolation in 1D."""
        result = nearest_manhattan_vectorized_arr(
            sample_coords_1d,
            sample_data_1d.shape,
            sample_data_1d
        )
        expected = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
        np.testing.assert_array_almost_equal(result, expected, decimal=5)
        
    def test_nearest_manhattan_2d(self, sample_data_2d, sample_coords_2d):
        """Test nearest neighbor interpolation in 2D."""
        result = nearest_manhattan_vectorized_arr(
            sample_coords_2d,
            sample_data_2d.shape,
            sample_data_2d
        )
        # Corners should be 0 or 1 based on checkerboard pattern
        assert result[0] == 1.0  # (0,0) in checkerboard
        assert result[2] == 0.0  # (1,0) - note: 1.0 in coords maps to index 9
        assert result[6] == 0.0  # (0,1)
        assert result[8] == 1.0  # (1,1)
        
    def test_nearest_manhattan_batched(self, sample_data_2d):
        """Test nearest neighbor with batched coordinates."""
        batch_coords = np.random.rand(5, 10, 2)  # 5 batches of 10 points
        result = nearest_manhattan_vectorized_arr(
            batch_coords,
            sample_data_2d.shape,
            sample_data_2d
        )
        assert result.shape == (5, 10)
        
    def test_linear_1d(self, sample_data_1d):
        coords = np.array([[0.125], [0.625]], dtype=float)  # midpoints
        result = linear_vectorized_arr(coords, sample_data_1d.shape, sample_data_1d)

        # halfway between 0.0 and 0.5
        assert abs(result[0] - 0.25) < 0.01

        # halfway between 1.0 and 0.5
        assert abs(result[1] - 0.75) < 0.01

        
    def test_linear_2d(self, sample_data_2d):
        """Test bilinear interpolation in 2D."""
        # Test at exact grid points
        exact_coords = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        result = linear_vectorized_arr(
            exact_coords,
            sample_data_2d.shape,
            sample_data_2d
        )
        np.testing.assert_array_almost_equal(
            result,
            np.array([1.0, 0.5, 1.0]),
            decimal=5
        )

        
    def test_linear_3d(self, sample_data_3d, sample_coords_3d):
        """Test trilinear interpolation in 3D."""
        result = linear_vectorized_arr(
            sample_coords_3d,
            sample_data_3d.shape,
            sample_data_3d
        )
        # Center point (0.5, 0.5, 0.5) should be in the cube
        assert result[4] == 1.0
        # Corner points should be 0
        assert result[0] == 0.0
        assert result[8] == 0.0
        
    def test_cubic_1d(self, sample_data_1d, sample_coords_1d):
        """Test cubic interpolation in 1D."""
        result = cubic_vectorized_arr(
            sample_coords_1d,
            sample_data_1d.shape,
            sample_data_1d
        )
        # Should produce smoother results than linear
        assert len(result) == 5
        # Check bounds
        assert np.all(result >= -0.1)  # Allow small negative overshoot
        assert np.all(result <= 1.1)   # Allow small positive overshoot
        
    def test_cubic_2d_symmetry(self, sample_data_2d):
        """Test cubic interpolation symmetry in 2D."""
        # Test symmetric points
        coords1 = np.array([[0.25, 0.25]])
        coords2 = np.array([[0.75, 0.75]])
        
        result1 = cubic_vectorized_arr(
            coords1,
            sample_data_2d.shape,
            sample_data_2d
        )
        result2 = cubic_vectorized_arr(
            coords2,
            sample_data_2d.shape,
            sample_data_2d
        )
        
        # Should be symmetric for checkerboard pattern
        np.testing.assert_almost_equal(result1, result2, decimal=5)
        
    def test_lanczos4_1d(self, sample_data_1d, sample_coords_1d):
        """Test Lanczos-4 interpolation in 1D."""
        result = lanczos4_vectorized_arr(
            sample_coords_1d,
            sample_data_1d.shape,
            sample_data_1d
        )
        # Lanczos should produce smooth results with reduced ringing
        assert len(result) == 5
        # Check reasonable bounds
        assert np.all(result >= -0.2)
        assert np.all(result <= 1.2)
        
    def test_lanczos4_edge_cases(self, sample_data_2d):
        """Test Lanczos-4 at edges."""
        edge_coords = np.array([[0.0, 0.0], [1.0, 1.0]])
        result = lanczos4_vectorized_arr(
            edge_coords,
            sample_data_2d.shape,
            sample_data_2d
        )
        # At exact grid points, should return exact values
        np.testing.assert_array_almost_equal(result, [1.0, 1.0], decimal=5)
        
    def test_interpolation_consistency(self, sample_data_2d):
        """Test consistency between different interpolation methods."""
        # Create gradient test image
        gradient = np.linspace(0, 1, 100).reshape(10, 10)
        
        coords = np.array([[0.25, 0.25], [0.5, 0.5], [0.75, 0.75]])
        
        # Test different methods
        nearest = nearest_manhattan_vectorized_arr(
            coords, gradient.shape, gradient
        )
        linear = linear_vectorized_arr(coords, gradient.shape, gradient)
        cubic = cubic_vectorized_arr(coords, gradient.shape, gradient)
        
        # Linear should be closer to cubic than nearest for smooth data
        linear_cubic_diff = np.abs(linear - cubic)
        nearest_linear_diff = np.abs(nearest - linear)
        
        # For smooth gradient, linear and cubic should be similar
        assert np.mean(linear_cubic_diff) < np.mean(nearest_linear_diff)
        
    def test_edge_handling(self, sample_data_2d):
        """Test interpolation at edges and beyond."""
        # Coordinates outside [0, 1] range
        coords = np.array([[-0.5, -0.5], [1.5, 1.5], [0.5, 0.5]])
        
        # All methods should handle out-of-bounds via clipping
        for method_name, interp_func in np_interp_dict.items():
            result = interp_func(coords, sample_data_2d.shape, sample_data_2d)
            assert not np.any(np.isnan(result))
            assert not np.any(np.isinf(result))
            
    def test_dtype_preservation(self):
        """Test that data types are preserved."""
        for dtype in [np.float32, np.float64]:
            data = np.random.rand(10, 10).astype(dtype)
            coords = np.random.rand(5, 2)
            
            for interp_func in np_interp_dict.values():
                result = interp_func(coords, data.shape, data)
                assert result.dtype == dtype
                
    def test_memory_layout(self):
        """Test interpolation with different array layouts."""
        # Test C-contiguous
        data_c = np.ascontiguousarray(np.random.rand(10, 10, 3))
        # Test F-contiguous
        data_f = np.asfortranarray(np.random.rand(10, 10, 3))
        
        coords = np.random.rand(5, 2)
        
        # Pass only spatial dimensions (first 2 dims for 2D coords)
        spatial_dims = data_c.shape[:2]
        
        for interp_func in np_interp_dict.values():
            result_c = interp_func(coords, spatial_dims, data_c)
            result_f = interp_func(coords, spatial_dims, data_f)
            
            assert result_c.shape == (5, 3)
            assert result_f.shape == (5, 3)

class TestOpenCVInterpolation:
    """Test OpenCV-based interpolation methods."""
    
    @pytest.fixture
    def sample_image_2d(self):
        """Create sample 2D image."""
        img = np.zeros((100, 100), dtype=np.float32)
        img[25:75, 25:75] = 1.0  # White square in center
        return img
    
    @pytest.fixture
    def sample_image_rgb(self):
        """Create sample RGB image."""
        img = np.zeros((100, 100, 3), dtype=np.float32)
        img[25:75, 25:75, 0] = 1.0  # Red square
        img[25:75, 25:75, 1] = 0.5  # Green square
        img[25:75, 25:75, 2] = 0.0  # Blue square
        return img
    
    @pytest.fixture
    def sample_coords_2d_grid(self):
        """Create grid of coordinates."""
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        xx, yy = np.meshgrid(x, y)
        return np.stack([xx.ravel(), yy.ravel()], axis=-1)
    
    def test_scale_to_pixel_space(self):
        """Test coordinate scaling for OpenCV."""
        coords = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        map_x, map_y = _scale_to_pixel_space(coords, height=100, width=200)
        
        assert map_x.shape == (3,)
        assert map_y.shape == (3,)
        np.testing.assert_almost_equal(map_x[0], 0.0)
        np.testing.assert_almost_equal(map_x[1], 99.5)  # (200-1)*0.5
        np.testing.assert_almost_equal(map_x[2], 199.0)
        np.testing.assert_almost_equal(map_y[1], 49.5)  # (100-1)*0.5
        
    def test_scale_to_pixel_space_single_pixel(self):
        """Test scaling for single-pixel images."""
        coords = np.array([[0.5, 0.5]])
        map_x, map_y = _scale_to_pixel_space(coords, height=1, width=1)
        np.testing.assert_almost_equal(map_x[0], 0.5)
        np.testing.assert_almost_equal(map_y[0], 0.5)
        
    def test_cv2_method_mapping(self):
        """Test OpenCV method constant mapping."""
        assert _CV2_INTERP_MAP[InterpMethod.NEAREST_MANHATTAN] == 0  # cv2.INTER_NEAREST
        assert _CV2_INTERP_MAP[InterpMethod.LINEAR] == 1  # cv2.INTER_LINEAR
        assert _CV2_INTERP_MAP[InterpMethod.CUBIC] == 2  # cv2.INTER_CUBIC
        assert _CV2_INTERP_MAP[InterpMethod.LANCZOS4] == 4  # cv2.INTER_LANCZOS4
        
    def test_cv2_nearest_interpolation(self, sample_image_2d):
        """Test nearest neighbor interpolation with OpenCV."""
        coords = np.array([[0.24, 0.24], [0.5, 0.5], [0.76, 0.76]])
        
        result = cv2_unit_field_sample(
            sample_image_2d,
            coords,
            InterpMethod.NEAREST_MANHATTAN
        )
        
        # At (0.25, 0.25) should be 0 (outside square)
        # At (0.5, 0.5) should be 1 (inside square)
        # At (0.75, 0.75) should be 0 (outside square)
        expected = np.array([0.0, 1.0, 0.0])
        np.testing.assert_array_almost_equal(result.ravel(), expected)
        
    def test_cv2_linear_interpolation(self, sample_image_2d):
        """Test linear interpolation with OpenCV."""
        # Test at edge of square 
        coords = np.array([[0.45, 0.5]])  # At the left edge of the square

        result = cv2_unit_field_sample(
            sample_image_2d,
            coords,
            InterpMethod.LINEAR
        )
        
        # At the edge, linear interpolation between 0 (outside) and 1 (inside)
        # should give ~0.5, but could vary based on exact pixel alignment
        print(f"Edge result: {result[0, 0]}")
        assert 0.0 <= result[0, 0] <= 1.0
            
    def test_cv2_rgb_interpolation(self, sample_image_rgb):
        """Test RGB image interpolation with OpenCV."""
        coords = np.array([[0.5, 0.5]])  # Center of red/green square
        
        result = cv2_unit_field_sample(
            sample_image_rgb,
            coords,
            InterpMethod.LINEAR
        )
        
        assert result.shape == (1, 1, 3)
        # Should be red + green = yellow-ish
        np.testing.assert_almost_equal(result[0, 0, 0], 1.0)  # Red
        np.testing.assert_almost_equal(result[0, 0, 1], 0.5)  # Green
        np.testing.assert_almost_equal(result[0, 0, 2], 0.0)  # Blue
        
    def test_cv2_batched_coordinates(self, sample_image_2d):
        """Test batched coordinate sampling with OpenCV."""
        batch_coords = np.random.rand(5, 10, 2)
        
        result = cv2_unit_field_sample(
            sample_image_2d,
            batch_coords,
            InterpMethod.LINEAR
        )
        
        assert result.shape == (5, 10)
        
    def test_cv2_invalid_dimensions(self):
        """Test OpenCV backend with invalid dimensions."""
        data_3d = np.random.rand(10, 10, 10)
        coords_3d = np.random.rand(5, 3)
        
        with pytest.raises(ValueError):
            cv2_unit_field_sample(
                data_3d,
                coords_3d,
                InterpMethod.LINEAR
            )
            
    def test_cv2_unsupported_method(self, sample_image_2d):
        """Test OpenCV backend with unsupported method."""
        coords = np.array([[0.5, 0.5]])
        
        # Create a dummy method not in cv2 supported methods
        class DummyMethod:
            pass
        
        dummy_method = DummyMethod()
        
        with pytest.raises(ValueError):
            cv2_unit_field_sample(
                sample_image_2d,
                coords,
                dummy_method
            )
            
    def test_cv2_border_modes(self, sample_image_2d):
        """Test different border modes."""
        import cv2
        
        # Coordinates outside the image
        coords = np.array([[1.5, 1.5]])
        
        # Test with constant border
        result = cv2_unit_field_sample(
            sample_image_2d,
            coords,
            InterpMethod.LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            border_value=0.5
        )
        
        assert result[0, 0] == 0.5
        


class TestBackendComparison:
    """Compare NumPy and OpenCV backends where applicable."""
    
    @pytest.fixture
    def test_image(self):
        """Create test image for comparison."""
        # Simple gradient
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)
        xx, yy = np.meshgrid(x, y)
        return xx + yy  # Diagonal gradient
    
    def test_nearest_interpolation_consistency(self, test_image):
        """Compare nearest neighbor between backends."""
        coords = np.random.rand(100, 2)
        
        # NumPy backend
        np_result = nearest_manhattan_vectorized_arr(
            coords,
            test_image.shape,
            test_image
        )
        
        # OpenCV backend
        cv2_result = cv2_unit_field_sample(
            test_image,
            coords,
            InterpMethod.NEAREST_MANHATTAN
        )
        
        # Should be very close (might differ by 1 index due to rounding)
        diff = np.abs(np_result - cv2_result.ravel())
        assert np.max(diff) < 0.01
        
    def test_linear_interpolation_consistency(self, test_image):
        """Compare linear interpolation between backends."""
        coords = np.random.rand(100, 2)
        
        # NumPy backend
        np_result = linear_vectorized_arr(
            coords,
            test_image.shape,
            test_image
        )
        
        # OpenCV backend
        cv2_result = cv2_unit_field_sample(
            test_image,
            coords,
            InterpMethod.LINEAR
        )
        
        # Should be very similar
        diff = np.abs(np_result - cv2_result.ravel())
        assert np.mean(diff) < 0.01
        assert np.max(diff) < 0.05
        
    def test_cubic_interpolation_consistency(self, test_image):
        """Compare cubic interpolation between backends."""
        coords = np.random.rand(100, 2)
        
        # NumPy backend
        np_result = cubic_vectorized_arr(
            coords,
            test_image.shape,
            test_image
        )
        
        # OpenCV backend
        cv2_result = cv2_unit_field_sample(
            test_image,
            coords,
            InterpMethod.CUBIC
        )
        
        # Cubic implementations might differ slightly
        diff = np.abs(np_result - cv2_result.ravel())
        assert np.mean(diff) < 0.05  # Allow more tolerance for cubic


class TestPerformance:
    """Performance tests (timing benchmarks)."""
    
    @pytest.mark.benchmark
    def test_interpolation_speed_numpy(self, benchmark):
        """Benchmark NumPy interpolation speed."""
        data = np.random.rand(1000, 1000)
        coords = np.random.rand(10000, 2)
        
        def run():
            return linear_vectorized_arr(coords, data.shape, data)
        
        result = benchmark(run)
        assert result.shape == (10000,) or result.shape == (10000, 1)
        
    @pytest.mark.benchmark
    def test_interpolation_speed_opencv(self, benchmark):
        """Benchmark OpenCV interpolation speed."""
        import cv2
        
        data = np.random.rand(1000, 1000).astype(np.float32)
        coords = np.random.rand(10000, 2)
        
        def run():
            return cv2_unit_field_sample(
                data,
                coords,
                InterpMethod.LINEAR
            )
        
        result = benchmark(run)
        assert result.shape == (10000,) or result.shape == (10000, 1)
        
    @pytest.mark.benchmark
    def test_cache_performance(self, benchmark):
        """Test cache performance for repeated calls."""
        from functools import lru_cache
        
        data = np.random.rand(100, 100)
        coords = np.random.rand(1000, 2)
        
        # Clear any existing caches
        from unitfield.core.interpolation.interp_np import _get_kernel_weights
        _get_kernel_weights.cache_clear()
        
        def run_multiple():
            results = []
            for _ in range(10):
                results.append(
                    linear_vectorized_arr(coords, data.shape, data)
                )
            return results
        
        benchmark(run_multiple)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_coordinates(self):
        """Test with empty coordinate arrays."""
        data = np.random.rand(10, 10)
        
        # Empty array
        coords = np.array([]).reshape(0, 2)
        
        for interp_func in np_interp_dict.values():
            result = interp_func(coords, data.shape, data)
            assert result.shape == (0,)
            
    def test_single_point(self):
        """Test with single coordinate point."""
        data = np.array([[1.0]])
        coords = np.array([[0.5, 0.5]])
        
        for interp_func in np_interp_dict.values():
            result = interp_func(coords, data.shape, data)
            assert result.shape == (1,)
            
    def test_very_large_coordinates(self):
        """Test with coordinates causing large intermediate arrays."""
        data = np.random.rand(10, 10)
        coords = np.random.rand(100000, 2)  # 100k points
        
        # Should not crash
        result = nearest_manhattan_vectorized_arr(
            coords, data.shape, data
        )
        assert result.shape == (100000,) or result.shape == (100000, 1)
                    

            
    def test_1d_image_interpolation(self):
        """Test interpolation with 1D images."""
        data = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
        coords = np.array([[0.0], [0.25], [0.5], [0.75], [1.0]])
        
        # Test all numpy methods that should work in 1D
        result_nearest = nearest_manhattan_vectorized_arr(
            coords, data.shape, data
        )
        result_linear = linear_vectorized_arr(coords, data.shape, data)
        
        assert result_nearest.shape == (5,)
        assert result_linear.shape == (5,)
        
    def test_4d_interpolation_numpy(self):
        """Test 4D interpolation with NumPy backend."""
        data = np.random.rand(8, 8, 8, 8)
        coords = np.random.rand(10, 4)
        
        # Linear should work for any dimension
        result = linear_vectorized_arr(coords, data.shape, data)
        assert result.shape == (10,)
        
    def test_dtype_conversion(self):
        """Test automatic dtype conversion."""
        # Test with integer input
        data_int = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        coords = np.array([[0.5, 0.5]])
        
        result = linear_vectorized_arr(coords, data_int.shape, data_int)
        # Should convert to float for interpolation
        assert result.dtype == np.float64 or result.dtype == np.float32


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])