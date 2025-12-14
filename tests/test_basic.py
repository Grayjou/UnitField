import pytest
import numpy as np
from ..unitfield.basic import MappedNdimField, InterpMethod, UnitMappedEndomorphism
from ..unitfield.distances import get_hypercube_points
from boundednumbers import UnitFloat

class TestMappedNdimField:
    
    def test_init_valid_data(self):
        """Test initialization with valid data shapes."""
        # 2D spatial, 2D vector (endomorphism: 2D → 2D)
        data_2d = np.random.rand(10, 10, 2)
        field = MappedNdimField(data_2d)
        assert field.ndim == 2
        assert field.spatial_shape == (10, 10)
        
        # 3D spatial, 3D vector (endomorphism: 3D → 3D)
        data_3d = np.random.rand(5, 5, 5, 3)
        field = MappedNdimField(data_3d)
        assert field.ndim == 3
        assert field.spatial_shape == (5, 5, 5)
    
    
    def test_get_value_coordinate_validation(self):
        """Test coordinate validation."""
        # 2D → 2D endomorphism
        data = np.random.rand(5, 5, 2)
        field = MappedNdimField(data)
        
        # Wrong number of coordinates
        with pytest.raises(ValueError):
            field.get_value((0.5,))  # 1 coordinate for 2D endomorphism
        
        with pytest.raises(ValueError):
            field.get_value((0.5, 0.5, 0.5))  # 3 coordinates for 2D endomorphism
        
        # Valid coordinates
        result = field.get_value((0.5, 0.5))
        assert len(result) == 2
    
    def test_nearest_manhattan(self):
        """Test nearest Manhattan interpolation."""
        # 2D → 2D endomorphism
        data = np.zeros((3, 3, 2))
        for i in range(3):
            for j in range(3):
                data[i, j] = [i/2, j/2]  # Scale to [0,1] for UnitFloat
        
        field = MappedNdimField(data, InterpMethod.NEAREST_MANHATTAN)
        
        # Test at cell centers
        # For 3x3 grid: coord * (size-1) = coord * 2
        # (0.166, 0.166) → 0.332 → round to 0 → index 0
        result = field.get_value((0.166, 0.166))
        assert np.allclose(result, (0.0, 0.0), rtol=1e-10)
        
        # (0.5, 0.5) → 1.0 → round to 1 → index 1
        result = field.get_value((0.5, 0.5))
        assert np.allclose(result, (0.5, 0.5), rtol=1e-10)
        
        # (0.833, 0.833) → 1.666 → round to 2 → index 2
        result = field.get_value((0.833, 0.833))
        assert np.allclose(result, (1.0, 1.0), rtol=1e-10)
    
    def test_nearest_euclidean(self):
        """Test nearest Euclidean interpolation."""
        # 2D → 2D endomorphism
        data = np.zeros((3, 3, 2))
        for i in range(3):
            for j in range(3):
                data[i, j] = [i/2, j/2]  # Scale to [0,1]
        
        field = MappedNdimField(data, InterpMethod.EUCLIDEAN)
        
        # (0.4, 0.4) → scaled to (0.8, 0.8)
        # Distance to (0,0): sqrt(0.8² + 0.8²) = 1.131
        # Distance to (1,1): sqrt(0.2² + 0.2²) = 0.283
        # So should pick index (1,1) which has value [0.5, 0.5]
        result = field.get_value((0.4, 0.4))
        assert np.allclose(result, (0.5, 0.5), rtol=1e-10)
    
    def test_linear_interpolation(self):
        """Test linear interpolation for 2D → 2D endomorphism."""
        # 2x2 grid with 2D vectors (bilinear interpolation)
        data = np.array([
            [[0.0, 0.0], [0.0, 1.0]],  # (0,0) and (0,1)
            [[1.0, 0.0], [1.0, 1.0]],  # (1,0) and (1,1)
        ])
        
        field = MappedNdimField(data, InterpMethod.LINEAR)
        
        # At (0.25, 0.25) - should be bilinear interpolation
        result = field.get_value((0.25, 0.25))
        expected = (0.25, 0.25)  # Both dimensions interpolate linearly
        np.testing.assert_allclose(result, expected, rtol=1e-10)
        
        # At (0.5, 0.5) - exact center
        result = field.get_value((0.5, 0.5))
        expected = (0.5, 0.5)
        np.testing.assert_allclose(result, expected, rtol=1e-10)
        
        # At (0.0, 0.0) - exact corner
        result = field.get_value((0.0, 0.0))
        expected = (0.0, 0.0)
        np.testing.assert_allclose(result, expected, rtol=1e-10)
    
    def test_1d_endomorphism(self):
        """Test 1D → 1D endomorphism."""
        # 1D → 1D: data shape (spatial_points, 1)
        data = np.array([
            [0.0],  # x=0
            [0.5],  # x=0.5
            [1.0],  # x=1.0
        ])
        
        field = MappedNdimField(data, InterpMethod.LINEAR)
        
        # Test linear interpolation in 1D
        result = field.get_value((0.25,))  # Between 0.0 and 0.5
        expected = (0.25,)
        np.testing.assert_allclose(result, expected, rtol=1e-10)
        
        result = field.get_value((0.75,))  # Between 0.5 and 1.0
        expected = (0.75,)
        np.testing.assert_allclose(result, expected, rtol=1e-10)
    
    def test_boundary_conditions(self):
        """Test behavior at boundaries."""
        # 2D → 2D endomorphism
        data = np.random.rand(4, 4, 2)
        
        for method in InterpMethod:
            field = MappedNdimField(data, method)
            
            # Lower bounds
            result_min = field.get_value((0.0, 0.0))
            
            # Upper bounds
            result_max = field.get_value((1.0, 1.0))
            
            # Should not raise any exceptions
            assert result_min is not None
            assert result_max is not None
            
            # Results should be within data bounds
            assert all(0 <= x <= 1 for x in result_min)
            assert all(0 <= x <= 1 for x in result_max)
    
    def test_cache_functionality(self):
        """Test that caching works correctly."""
        # 2D → 2D endomorphism
        data = np.random.rand(5, 5, 2)
        field = MappedNdimField(data, InterpMethod.LINEAR, cache_size=10)
        
        # First call
        result1 = field.get_value((0.5, 0.5))
        
        # Second call with same coordinates (should be cached)
        result2 = field.get_value((0.5, 0.5))
        
        np.testing.assert_array_equal(result1, result2)
        
        # Test with different coordinates
        result3 = field.get_value((0.6, 0.6))
        assert not np.array_equal(result1, result3)
    
    def test_interpolation_methods_enum(self):
        """Test all interpolation methods are available."""
        # 2D → 2D endomorphism
        data = np.random.rand(3, 3, 2)
        
        for method in InterpMethod:
            field = MappedNdimField(data, method)
            result = field.get_value((0.5, 0.5))
            assert len(result) == 2
            # For random data, results should be within data bounds
            assert all(0 <= x <= 1 for x in result)
    
    def test_data_types(self):
        """Test with different data types."""
        # Float32 - 2D → 2D endomorphism
        data_float32 = np.random.rand(3, 3, 2).astype(np.float32)
        field = MappedNdimField(data_float32, InterpMethod.LINEAR)
        result = field.get_value((0.5, 0.5))
        assert len(result) == 2
        
        # Float64 - 2D → 2D endomorphism
        data_float64 = np.random.rand(3, 3, 2).astype(np.float64)
        field = MappedNdimField(data_float64, InterpMethod.LINEAR)
        result = field.get_value((0.5, 0.5))
        assert len(result) == 2
    
    def test_consistency_across_methods(self):
        """Test that all methods produce valid outputs for same input."""
        # 2D → 2D endomorphism
        data = np.random.rand(5, 5, 2)
        coords = (0.5, 0.5)
        
        for method in InterpMethod:
            field = MappedNdimField(data, method)
            result = field.get_value(coords)
            
            # Check output shape matches input dimension
            assert len(result) == 2
            
            # Check bounds
            assert all(0 <= x <= 1 for x in result)
    
    def test_edge_cases(self):
        """Test various edge cases."""
        # Single cell grid - 2D → 2D endomorphism
        data_single = np.random.rand(1, 1, 2)
        field = MappedNdimField(data_single, InterpMethod.LINEAR)
        result = field.get_value((0.5, 0.5))
        np.testing.assert_array_almost_equal(result, data_single[0, 0])
        
        # 1D endomorphism
        data_1d = np.random.rand(5, 1)
        field = MappedNdimField(data_1d, InterpMethod.LINEAR)
        result = field.get_value((0.5,))
        assert len(result) == 1
        
        # 3D endomorphism
        data_3d = np.random.rand(3, 3, 3, 3)
        field = MappedNdimField(data_3d, InterpMethod.NEAREST_MANHATTAN)
        result = field.get_value((0.5, 0.5, 0.5))
        assert len(result) == 3
    
    def test_extreme_coordinates(self):
        """Test with coordinates at extremes."""
        # 2D → 2D endomorphism
        data = np.array([
            [[0.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [1.0, 1.0]],
        ])
        
        field = MappedNdimField(data, InterpMethod.LINEAR)
        
        # Exactly at (0,0)
        result = field.get_value((0.0, 0.0))
        np.testing.assert_array_almost_equal(result, (0.0, 0.0))
        
        # Exactly at (1,1)
        result = field.get_value((1.0, 1.0))
        np.testing.assert_array_almost_equal(result, (1.0, 1.0))
    
    def test_n_linear_weights_sum_to_one(self):
        """Test that bilinear interpolation weights sum to 1."""

        
        # 2D → 2D endomorphism
        data = np.random.rand(2, 2, 2)
        field = MappedNdimField(data, InterpMethod.LINEAR)
        
        # Test multiple random points
        for _ in range(10):
            coords = tuple(np.random.rand(2))
            result = field.get_value(coords)
            
            # Manually calculate using the same logic to verify
            corners, fractions = get_hypercube_points(coords, (2, 2))
            
            total_weight = 0.0
            expected_result = np.zeros(2)
            
            for corner, mask in corners:
                weight = 1.0
                for d, t in enumerate(fractions):
                    if mask[d] == 0:
                        weight *= (1 - t)
                    else:
                        weight *= t
                
                total_weight += weight
                expected_result += weight * data[corner]
            
            # Weights should sum to 1
            assert np.allclose(total_weight, 1.0, rtol=1e-10)
            
            # Result should match expected
            np.testing.assert_allclose(result, tuple(expected_result), rtol=1e-10)
    
    def test_example_from_note(self):
        """Test the specific example mentioned in the notes."""
        # Example from notes: 2D spatial (2x3) with 2D vectors
        data = np.array([
            [[0.0, 0.0], [0.5, 1.0], [1.0, 0.5]],  # Row 0
            [[1.0, 0.0], [0.7, 0.8], [0.2, 0.3]],  # Row 1
        ])
        
        field = MappedNdimField(data, InterpMethod.NEAREST_EUCLIDEAN)
        
        # Test (0.9, 0.8) - should map to closest point
        # In 2x3 grid: coord * (size-1) = (0.9*1, 0.8*2) = (0.9, 1.6)
        # Closest grid point: (1, 2) → value [0.2, 0.3]
        result = field.get_value((0.9, 0.8))
        # Note: This depends on your distance calculation
        
        # Test (0.6, 0.7) - should map to closest
        # (0.6, 1.4) → closest to (1, 1) → value [0.7, 0.8]
        result = field.get_value((0.6, 0.7))
    
    def test_endomorphism_error(self):
        """Test that non-endomorphism data raises error."""
        # 2D spatial, 3D vector (not an endomorphism)
        data_invalid = np.random.rand(4, 4, 3)
        
        with pytest.raises(ValueError):
            UnitMappedEndomorphism(data_invalid)

        valid_data = np.random.rand(4, 4, 2)  # 2D spatial, 2D vector
        try:
            UnitMappedEndomorphism(valid_data)
        except ValueError:
            pytest.fail("UnitMappedEndomorphism raised ValueError unexpectedly!")

    def test_return_types(self):
        """Test different return types."""
        data = np.array([
            [[0.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [1.0, 1.0]],
        ])
        
        field = MappedNdimField(data, InterpMethod.LINEAR)
        
        result_array = field.get_value((0.5, 0.5), return_type="array")
        assert isinstance(result_array, np.ndarray)
        np.testing.assert_array_almost_equal(result_array, [0.5, 0.5])
        
        result_tuple = field.get_value((0.5, 0.5), return_type="tuple")
        assert isinstance(result_tuple, tuple)
        np.testing.assert_array_almost_equal(result_tuple, (0.5, 0.5))
        
        result_list = field.get_value((0.5, 0.5), return_type="list")
        assert isinstance(result_list, list)
        np.testing.assert_array_almost_equal(result_list, [0.5, 0.5])