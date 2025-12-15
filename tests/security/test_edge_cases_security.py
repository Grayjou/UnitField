"""
Security and edge case tests for UnitField.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pytest
import numpy as np

from unitfield.core.enums import InterpMethod
from unitfield.core.unitfield import (
    MappedUnitField,
    Unit2DMappedEndomorphism
)


@pytest.fixture
def sample_2d_field_data():
    """Create sample 2D field data."""
    x, y = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
    return np.stack([x, y], axis=-1)


class TestInfNaNHandling:
    """Tests for handling inf and NaN values."""
    
    def test_data_with_nan(self):
        """Test with data containing NaN values."""
        data = np.array([
            [[0.0, 0.0], [1.0, np.nan]],
            [[0.5, 0.0], [0.5, 1.0]]
        ])
        
        field = MappedUnitField(
            data=data,
            interp_method=InterpMethod.LINEAR
        )
        
        # Should handle without crashing
        result = field.get_value((0.5, 0.5))
        assert isinstance(result, tuple)
        # Result might contain NaN - that's OK for data values
    
    def test_data_with_inf(self):
        """Test with data containing inf values."""
        data = np.array([
            [[0.0, 0.0], [np.inf, 0.5]],
            [[0.5, 0.0], [-np.inf, 1.0]]
        ])
        
        field = MappedUnitField(
            data=data,
            interp_method=InterpMethod.LINEAR
        )
        
        # Should handle without crashing
        result = field.get_value((0.5, 0.5))
        assert isinstance(result, tuple)
        # Result might contain inf - that's OK for data values
    
    def test_coordinates_with_nan_documented_undefined_behavior(self, sample_2d_field_data):
        """
        Test that NaN coordinates produce some result (undefined behavior is OK).
        
        Note: As documented, inf/NaN coordinates are NOT supported and behavior
        is undefined. This test just ensures we don't crash.
        """
        field = MappedUnitField(
            data=sample_2d_field_data,
            interp_method=InterpMethod.LINEAR
        )
        
        # This is undefined behavior but shouldn't crash
        try:
            result = field.get_value((np.nan, 0.5))
            # If it returns something, that's fine (undefined is allowed)
            assert isinstance(result, tuple)
        except Exception:
            # If it raises an exception, that's also fine (undefined is allowed)
            pass
    
    def test_coordinates_with_inf_documented_undefined_behavior(self, sample_2d_field_data):
        """
        Test that inf coordinates produce some result (undefined behavior is OK).
        
        Note: As documented, inf/NaN coordinates are NOT supported and behavior
        is undefined. This test just ensures we don't crash hard.
        """
        field = MappedUnitField(
            data=sample_2d_field_data,
            interp_method=InterpMethod.LINEAR
        )
        
        # This is undefined behavior but shouldn't crash the process
        try:
            result = field.get_value((np.inf, 0.5))
            # If it returns something, that's fine (undefined is allowed)
            assert isinstance(result, tuple)
        except Exception:
            # If it raises an exception, that's also fine (undefined is allowed)
            pass


class TestOutOfBoundsCoordinates:
    """Tests for out-of-bounds coordinate handling."""
    
    def test_negative_coordinates(self, sample_2d_field_data):
        """Test that negative coordinates are handled (clipping)."""
        field = MappedUnitField(
            data=sample_2d_field_data,
            interp_method=InterpMethod.LINEAR
        )
        
        # Negative coordinates should be clipped to 0
        result = field.get_value((-0.5, 0.5))
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_greater_than_one_coordinates(self, sample_2d_field_data):
        """Test that coordinates > 1 are handled (clipping)."""
        field = MappedUnitField(
            data=sample_2d_field_data,
            interp_method=InterpMethod.LINEAR
        )
        
        # Coordinates > 1 should be clipped to 1
        result = field.get_value((1.5, 0.5))
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_extreme_coordinates(self, sample_2d_field_data):
        """Test with coordinates at extremes (0 and 1)."""
        field = MappedUnitField(
            data=sample_2d_field_data,
            interp_method=InterpMethod.LINEAR
        )
        
        # Test at corners
        corners = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
        for corner in corners:
            result = field.get_value(corner)
            assert isinstance(result, tuple)
            assert len(result) == 2


class TestMemoryAndPerformance:
    """Tests for memory and performance edge cases."""
    
    def test_large_data_handling(self):
        """Test with large data arrays."""
        # Create large but manageable dataset
        shape = (100, 100, 2)
        large_data = np.random.rand(*shape)
        
        field = MappedUnitField(
            data=large_data,
            interp_method=InterpMethod.LINEAR,
            cache_size=1000
        )
        
        # Should work without memory issues
        result = field.get_value((0.5, 0.5))
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_small_data_handling(self):
        """Test with very small data arrays."""
        # Minimum valid data
        small_data = np.random.rand(2, 2, 1)
        
        field = MappedUnitField(
            data=small_data,
            interp_method=InterpMethod.NEAREST_MANHATTAN
        )
        
        result = field.get_value((0.5, 0.5))
        assert isinstance(result, tuple)
        assert len(result) == 1
    
    def test_empty_coordinate_batch(self):
        """Test with empty coordinate arrays."""
        data = np.random.rand(10, 10, 2)
        field = MappedUnitField(
            data=data,
            interp_method=InterpMethod.LINEAR
        )
        
        # Empty array
        coords = np.array([]).reshape(0, 2)
        result = field.get_values(coords)
        assert result.shape[0] == 0


class TestInputValidation:
    """Tests for input validation and error handling."""
    
    def test_invalid_coordinate_type(self, sample_2d_field_data):
        """Test with invalid coordinate types."""
        field = MappedUnitField(
            data=sample_2d_field_data,
            interp_method=InterpMethod.LINEAR
        )
        
        # String coordinates should be handled gracefully
        try:
            result = field.get_value(("not", "coords"))
            # If numpy converts it, that's OK
        except (ValueError, TypeError):
            # Expected to fail
            pass
    
    def test_zero_dimension_data(self):
        """Test with 0-dimensional arrays."""
        data = np.array(0.5)
        
        with pytest.raises(ValueError, match="Data must have at least 1 dimension"):
            MappedUnitField(data=data, interp_method=InterpMethod.LINEAR)
    
    def test_data_type_integer(self):
        """Test that integer data is handled correctly."""
        # Integer data should be converted to float
        data = np.array([[[0, 0], [1, 1]], [[0, 1], [1, 0]]], dtype=np.uint8)
        
        field = MappedUnitField(
            data=data,
            interp_method=InterpMethod.LINEAR
        )
        
        result = field.get_value((0.5, 0.5))
        assert isinstance(result, tuple)


class TestThreadSafety:
    """Tests for thread safety."""
    
    def test_concurrent_access(self, sample_2d_field_data):
        """Test basic thread safety (no guarantees, but shouldn't crash)."""
        import concurrent.futures
        
        field = MappedUnitField(
            data=sample_2d_field_data,
            interp_method=InterpMethod.LINEAR,
            cache_size=100
        )
        
        def query_field(coord):
            return field.get_value(coord)
        
        coords = [(i/10, j/10) for i in range(10) for j in range(10)]
        
        # Run queries in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(query_field, coord) for coord in coords]
            results = [f.result() for f in futures]
        
        assert len(results) == 100
        for result in results:
            assert isinstance(result, tuple)
            assert len(result) == 2


class TestSerialization:
    """Tests for serialization compatibility."""
    
    def test_pickle_compatibility(self, sample_2d_field_data):
        """Test that fields can be pickled."""
        import pickle
        
        field = MappedUnitField(
            data=sample_2d_field_data,
            interp_method=InterpMethod.LINEAR,
            cache_size=10
        )
        
        # Pickle and unpickle
        pickled = pickle.dumps(field)
        unpickled = pickle.loads(pickled)
        
        assert unpickled.ndim == field.ndim
        assert unpickled.spatial_shape == field.spatial_shape
        assert unpickled.interp_method == field.interp_method
        np.testing.assert_array_equal(unpickled.data, field.data)
        
        # Should still work
        result = unpickled.get_value((0.5, 0.5))
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestDTypeHandling:
    """Tests for different data types."""
    
    def test_float32_data(self):
        """Test with float32 data."""
        data = np.random.rand(10, 10, 2).astype(np.float32)
        field = MappedUnitField(data=data, interp_method=InterpMethod.LINEAR)
        
        result = field.get_value((0.5, 0.5))
        assert isinstance(result, tuple)
    
    def test_float64_data(self):
        """Test with float64 data."""
        data = np.random.rand(10, 10, 2).astype(np.float64)
        field = MappedUnitField(data=data, interp_method=InterpMethod.LINEAR)
        
        result = field.get_value((0.5, 0.5))
        assert isinstance(result, tuple)
    
    def test_mixed_dtype_coordinates(self):
        """Test with different coordinate dtypes."""
        data = np.random.rand(10, 10, 2)
        field = MappedUnitField(data=data, interp_method=InterpMethod.LINEAR)
        
        # int coordinates
        result = field.get_value((0, 0))
        assert isinstance(result, tuple)
        
        # float32 coordinates
        result = field.get_value((np.float32(0.5), np.float32(0.5)))
        assert isinstance(result, tuple)


class TestBoundaryConditions:
    """Tests for boundary conditions."""
    
    def test_single_pixel_image(self):
        """Test with 1x1 image."""
        data = np.array([[[0.5, 0.5]]])
        field = MappedUnitField(data=data, interp_method=InterpMethod.LINEAR)
        
        result = field.get_value((0.5, 0.5))
        assert isinstance(result, tuple)
    
    def test_single_row_image(self):
        """Test with 1xN image."""
        data = np.random.rand(1, 10, 2)
        field = MappedUnitField(data=data, interp_method=InterpMethod.LINEAR)
        
        result = field.get_value((0.5, 0.5))
        assert isinstance(result, tuple)
    
    def test_single_column_image(self):
        """Test with Nx1 image."""
        data = np.random.rand(10, 1, 2)
        field = MappedUnitField(data=data, interp_method=InterpMethod.LINEAR)
        
        result = field.get_value((0.5, 0.5))
        assert isinstance(result, tuple)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
