"""
Tests for the MappedUnitField class.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pytest
import numpy as np

from unitfield.core.enums import InterpMethod
from unitfield.core.types import DEFAULT_DTYPE
from unitfield.core.unitfield import MappedUnitField
from unitfield.interpolation.interp_np import np_interp_dict


@pytest.fixture
def sample_2d_field_data():
    """Create sample 2D field data."""
    x, y = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
    return np.stack([x, y], axis=-1)


@pytest.fixture
def sample_1d_field_data():
    """Create sample 1D field data."""
    return np.linspace(0, 1, 10).reshape(-1, 1)


@pytest.fixture
def sample_3d_field_data():
    """Create sample 3D field data."""
    shape = (4, 4, 4, 3)
    return np.random.rand(*shape)


class TestMappedUnitFieldInitialization:
    """Tests for MappedUnitField initialization."""
    
    def test_initialization_1d(self, sample_1d_field_data):
        """Test initialization of 1D field."""
        field = MappedUnitField(
            data=sample_1d_field_data,
            interp_method=InterpMethod.LINEAR,
            cache_size=10
        )
        
        assert field.ndim == 1
        assert field.spatial_shape == (10,)
        assert field.interp_method == InterpMethod.LINEAR
        assert field.cache_size == 10
        np.testing.assert_array_equal(field.data, sample_1d_field_data)
    
    def test_initialization_2d(self, sample_2d_field_data):
        """Test initialization of 2D field."""
        field = MappedUnitField(
            data=sample_2d_field_data,
            interp_method=InterpMethod.CUBIC
        )
        
        assert field.ndim == 2
        assert field.spatial_shape == (5, 5)
        assert field.interp_method == InterpMethod.CUBIC
        assert field.cache_size is not None
    
    def test_initialization_with_invalid_data(self):
        """Test initialization with invalid data type."""
        with pytest.raises(TypeError, match="Data must be numpy array"):
            MappedUnitField(data=[1, 2, 3], interp_method=InterpMethod.LINEAR)
    
    def test_initialization_with_empty_data(self):
        """Test initialization with empty array."""
        empty_data = np.array([])
        with pytest.raises(ValueError, match="Data array must not be empty"):
            MappedUnitField(data=empty_data, interp_method=InterpMethod.LINEAR)
    
    def test_initialization_with_invalid_method(self):
        """Test initialization with invalid interpolation method."""
        data = np.random.rand(10, 10, 2)
        with pytest.raises(TypeError, match="interp_method must be InterpMethod"):
            MappedUnitField(data=data, interp_method="invalid_method")
    
    def test_initialization_with_negative_cache(self):
        """Test initialization with negative cache size."""
        data = np.random.rand(10, 10, 2)
        with pytest.raises(ValueError, match="cache_size must be non-negative"):
            MappedUnitField(data=data, cache_size=-1)


class TestMappedUnitFieldQueries:
    """Tests for MappedUnitField query methods."""
    
    def test_get_value_single_coordinate(self, sample_2d_field_data):
        """Test getting value at single coordinate."""
        field = MappedUnitField(
            data=sample_2d_field_data,
            interp_method=InterpMethod.LINEAR
        )
        
        # Test at known coordinate (center)
        result = field.get_value((0.5, 0.5))
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert 0.0 <= result[0] <= 1.0
        assert 0.0 <= result[1] <= 1.0
        
        # Test near (0, 0) - should be near (0, 0)
        result = field.get_value((0.1, 0.1))
        assert result[0] < 0.2
        assert result[1] < 0.2
    
    def test_get_value_coordinate_validation(self, sample_2d_field_data):
        """Test coordinate dimension validation in get_value."""
        field = MappedUnitField(
            data=sample_2d_field_data,
            interp_method=InterpMethod.LINEAR
        )
        
        # Test with wrong number of coordinates
        with pytest.raises(ValueError, match="Expected 2 coordinates"):
            field.get_value((0.5, 0.5, 0.5))
        
        with pytest.raises(ValueError, match="Expected 2 coordinates"):
            field.get_value((0.5,))
    
    def test_get_values_multiple_coordinates(self, sample_2d_field_data):
        """Test getting values at multiple coordinates."""
        field = MappedUnitField(
            data=sample_2d_field_data,
            interp_method=InterpMethod.LINEAR
        )
        
        # Test single coordinate array
        coords = np.array([[0.5, 0.5]])
        result = field.get_values(coords)
        assert result.shape == (1, 2)
        
        # Test multiple coordinates
        coords = np.array([
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 1.0]
        ])
        result = field.get_values(coords)
        assert result.shape == (3, 2)
        
        # Test batched coordinates
        coords = np.random.rand(2, 3, 2)  # 2 batches of 3 points
        result = field.get_values(coords)
        assert result.shape == (2, 3, 2)
    
    def test_get_values_coordinate_validation(self, sample_2d_field_data):
        """Test coordinate dimension validation in get_values."""
        field = MappedUnitField(
            data=sample_2d_field_data,
            interp_method=InterpMethod.LINEAR
        )
        
        # Test with wrong coordinate dimension
        coords = np.array([[0.5, 0.5, 0.5]])
        with pytest.raises(ValueError, match="Expected coordinates with 2 dimensions"):
            field.get_values(coords)


class TestMappedUnitFieldCaching:
    """Tests for MappedUnitField caching behavior."""
    
    def test_caching_behavior(self, sample_2d_field_data):
        """Test that caching works correctly."""
        # Create field with small cache
        field = MappedUnitField(
            data=sample_2d_field_data,
            interp_method=InterpMethod.LINEAR,
            cache_size=2
        )
        
        # Call get_value multiple times with same coordinate
        result1 = field.get_value((0.5, 0.5))
        result2 = field.get_value((0.5, 0.5))
        
        # Results should be identical
        assert result1 == result2
        
        # Fill cache with different coordinates
        field.get_value((0.1, 0.1))
        field.get_value((0.2, 0.2))
        field.get_value((0.3, 0.3))  # This should evict (0.5, 0.5) from cache
        
        # Cache should still work for (0.5, 0.5) (recomputes)
        result3 = field.get_value((0.5, 0.5))
        assert result3 == result1
    
    def test_no_caching(self, sample_2d_field_data):
        """Test field with caching disabled."""
        field = MappedUnitField(
            data=sample_2d_field_data,
            interp_method=InterpMethod.LINEAR,
            cache_size=None
        )
        
        # get_value should still work
        result = field.get_value((0.5, 0.5))
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestMappedUnitFieldMethods:
    """Tests for other MappedUnitField methods."""
    
    def test_with_interp_method(self):
        """Test creating field with different interpolation method."""
        # Use structured non-linear data
        xs, ys = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5), indexing='ij')
        data = np.stack([
            xs + 0.3 * np.sin(2 * np.pi * xs) * np.sin(2 * np.pi * ys),
            ys + 0.3 * np.cos(2 * np.pi * xs) * np.cos(2 * np.pi * ys)
        ], axis=-1)
        
        field1 = MappedUnitField(
            data=data,
            interp_method=InterpMethod.LINEAR
        )
        
        field2 = field1.with_interp_method(InterpMethod.CUBIC)
        
        assert field2 is not field1
        assert field2.interp_method == InterpMethod.CUBIC
        assert field2.cache_size == field1.cache_size
        np.testing.assert_array_equal(field2.data, field1.data)
    
    def test_repr(self, sample_2d_field_data):
        """Test string representation."""
        field = MappedUnitField(
            data=sample_2d_field_data,
            interp_method=InterpMethod.LINEAR,
            cache_size=128
        )
        
        repr_str = repr(field)
        assert field.__class__.__name__ in repr_str
        assert "shape=" in repr_str
        assert "interp=" in repr_str
        assert "cache=" in repr_str
    
    def test_all_interpolation_methods(self, sample_2d_field_data):
        """Test that all interpolation methods work."""
        for method in np_interp_dict.keys():
            field = MappedUnitField(
                data=sample_2d_field_data,
                interp_method=method
            )
            
            # Should be able to get value without error
            result = field.get_value((0.5, 0.5))
            assert isinstance(result, tuple)
            assert len(result) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
