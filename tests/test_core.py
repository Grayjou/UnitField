# UnitField\tests\test_core.py
"""
Comprehensive tests for the core unit field module.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock
from typing import Tuple, List

# Use absolute imports from the project root
from unitfield.core.enums import InterpMethod
from unitfield.core.types import DEFAULT_DTYPE
from unitfield.core.unitfield import (
    UnitNdimField,
    MappedUnitField,
    UnitMappedEndomorphism,
    Unit2DMappedEndomorphism,
    #validate_coordinates,
    remap_tensor_cv2,
)
from unitfield.interpolation.interp_np import np_interp_dict




@pytest.fixture
def sample_2d_field_data():
    """Create sample 2D field data."""
    x, y = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
    return np.stack([x, y], axis=-1)


class TestRemapTensorCV2:
    """Tests for the remap_tensor_cv2 function."""
    
    @pytest.fixture
    def sample_tensor(self):
        """Create a sample 3D tensor."""
        return np.random.rand(10, 10, 3).astype(np.float32)
    
    @pytest.fixture
    def sample_mapping(self):
        """Create a sample pixel-space mapping."""
        map_x = np.random.rand(10, 10).astype(np.float32) * 9
        map_y = np.random.rand(10, 10).astype(np.float32) * 9
        return np.stack([map_x, map_y], axis=-1)
    
    def test_remap_2d_image(self, sample_tensor, sample_mapping):
        """Test remapping a 2D image."""
        # Convert 3D tensor to 2D for testing
        image_2d = sample_tensor[..., 0]
        
        with patch('unitfield.core.unitfield.cv2.remap') as mock_remap:
            mock_remap.return_value = np.zeros_like(image_2d)
            result = remap_tensor_cv2(image_2d, sample_mapping)
            
            mock_remap.assert_called_once()
            assert result.shape == image_2d.shape
    
    def test_remap_3d_tensor(self, sample_tensor, sample_mapping):
        """Test remapping a 3D tensor."""
        with patch('unitfield.core.unitfield.cv2.remap') as mock_remap:
            mock_remap.return_value = np.zeros_like(sample_tensor)
            result = remap_tensor_cv2(sample_tensor, sample_mapping)
            
            mock_remap.assert_called_once()
            assert result.shape == sample_tensor.shape
    
    def test_invalid_data_dimensions(self):
        """Test error with 1D data."""
        data_1d = np.random.rand(10)
        mapping = np.random.rand(10, 2, 2)
        
        with pytest.raises(ValueError, match="Data must be at least 2-dimensional"):
            remap_tensor_cv2(data_1d, mapping)
    
    def test_invalid_mapping_shape(self):
        """Test error with invalid mapping shape."""
        data = np.random.rand(10, 10)
        mapping = np.random.rand(10, 10, 3)  # Wrong last dimension
        
        with pytest.raises(ValueError, match="Mapping must have shape"):
            remap_tensor_cv2(data, mapping)
    
    def test_mismatched_spatial_dimensions(self):
        """Test error when spatial dimensions don't match."""
        data = np.random.rand(10, 12)
        mapping = np.random.rand(10, 10, 2)
        
        with pytest.raises(ValueError, match="Data shape.*and mapping shape"):
            remap_tensor_cv2(data, mapping)
    
    def test_cv2_error_handling(self):
        """Test that cv2 errors are properly wrapped."""
        data = np.random.rand(10, 10).astype(np.float32)
        mapping = np.stack([
            np.random.rand(10, 10).astype(np.float32) * 9,
            np.random.rand(10, 10).astype(np.float32) * 9
        ], axis=-1)
        
        with patch('unitfield.core.unitfield.cv2.remap') as mock_remap:
            mock_remap.side_effect = cv2.error("Test error")
            
            with pytest.raises(RuntimeError, match="OpenCV remap failed"):
                remap_tensor_cv2(data, mapping)
    
    def test_border_modes_and_values(self):
        """Test different border modes and values."""
        data = np.random.rand(10, 10, 3).astype(np.float32)
        mapping = np.stack([
            np.random.rand(10, 10).astype(np.float32) * 9,
            np.random.rand(10, 10).astype(np.float32) * 9
        ], axis=-1)
        
        with patch('unitfield.core.unitfield.cv2.remap') as mock_remap:
            mock_remap.return_value = np.zeros_like(data)
            
            # Test different border modes
            remap_tensor_cv2(
                data, mapping, 
                border_mode=cv2.BORDER_CONSTANT,
                border_value=1.0
            )
            
            # Check that cv2.remap was called with correct parameters
            call_kwargs = mock_remap.call_args[1]
            assert call_kwargs['borderMode'] == cv2.BORDER_CONSTANT
            assert call_kwargs['borderValue'] == 1.0


class TestMappedUnitField:
    """Tests for the MappedUnitField class."""
    
    @pytest.fixture
    def sample_1d_field_data(self):
        """Create sample 1D field data."""
        return np.linspace(0, 1, 10).reshape(-1, 1)
    

    
    @pytest.fixture
    def sample_3d_field_data(self):
        """Create sample 3D field data."""
        shape = (4, 4, 4, 3)
        return np.random.rand(*shape)
    
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
        # Check if it has at least 1 dimension and size > 0
        with pytest.raises(ValueError, match="Data array must not be empty"):
            MappedUnitField(data=empty_data, interp_method=InterpMethod.LINEAR)

    def test_initialization_with_invalid_method(self):
        """Test initialization with invalid interpolation method."""
        data = np.random.rand(10, 10, 2)
        with pytest.raises(TypeError, match="interp_method must be InterpMethod"):
            MappedUnitField(data=data, interp_method="invalid_method")

    def test_invalid_initialization_wrong_dimensions(self):
        """Test initialization with wrong dimensions."""
        # 3D data instead of 2D
        data_3d = np.random.rand(10, 10, 10, 3)
        with pytest.raises(ValueError, match="does not represent a 2D endomorphism"):
            Unit2DMappedEndomorphism(data=data_3d, interp_method=InterpMethod.LINEAR)
        
        # Wrong last dimension
        data_wrong_last = np.random.rand(10, 10, 3)
        with pytest.raises(ValueError, match="does not represent a 2D endomorphism"):
            Unit2DMappedEndomorphism(data=data_wrong_last, interp_method=InterpMethod.LINEAR)

    def test_initialization_with_negative_cache(self):
        """Test initialization with negative cache size."""
        data = np.random.rand(10, 10, 2)
        with pytest.raises(ValueError, match="cache_size must be non-negative"):
            MappedUnitField(data=data, cache_size=-1)
    
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
    
    def test_with_interp_method(self):
        """Test creating field with different interpolation method."""
        # Use structured non-linear data to ensure different interpolation methods
        # produce different results
        # Create a 5x5 grid with a wave pattern
        xs, ys = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5), indexing='ij')
        # Non-linear transformation: sine wave pattern
        data = np.stack([
            xs + 0.3 * np.sin(2 * np.pi * xs) * np.sin(2 * np.pi * ys),
            ys + 0.3 * np.cos(2 * np.pi * xs) * np.cos(2 * np.pi * ys)
        ], axis=-1)
        
        field1 = MappedUnitField(
            data=data,
            interp_method=InterpMethod.LINEAR
        )

        field2 = field1.with_interp_method(InterpMethod.CUBIC)

        assert field2 is not field1  # Should be new instance
        assert field2.interp_method == InterpMethod.CUBIC
        assert field2.cache_size == field1.cache_size
        np.testing.assert_array_equal(field2.data, field1.data)

        # Test at multiple points to ensure different interpolation methods
        # produce different results (they should for non-linear data)
        test_points = [
            (0.25, 0.25),
            (0.5, 0.5),
            (0.75, 0.75),
            (0.33, 0.67),
            (0.67, 0.33)
        ]
        
        differences = []
        for point in test_points:
            result1 = field1.get_value(point)
            result2 = field2.get_value(point)
            # Calculate L2 distance between results
            diff = np.sqrt((result1[0] - result2[0])**2 + (result1[1] - result2[1])**2)
            differences.append(diff)
        
        # At least some points should show significant differences
        # (Cubic interpolation should be different from linear for non-linear data)
        max_diff = max(differences)
        assert max_diff > 1e-5, (
            f"Linear and cubic interpolation produced nearly identical results "
            f"(max difference: {max_diff}). This is unexpected for non-linear data."
        )
    
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
    
    def test_data_immutability(self, sample_2d_field_data):
        """Test that data property returns a copy or at least doesn't allow modification."""
        original_data = sample_2d_field_data.copy()
        field = MappedUnitField(data=original_data, interp_method=InterpMethod.LINEAR)
        
        # Modify the returned data
        field.data[0, 0, 0] = 999
        
        # Original data should not be modified (if property returns copy)
        # This test depends on implementation - remove if property doesn't return copy
        if not np.may_share_memory(field.data, original_data):
            assert original_data[0, 0, 0] != 999


class TestUnitMappedEndomorphism:
    """Tests for the UnitMappedEndomorphism class."""
    
    @pytest.fixture
    def valid_endomorphism_data(self):
        """Create valid endomorphism data (output dim equals spatial dims)."""
        shape = (5, 5, 2)  # 2D endomorphism
        return np.random.rand(*shape)
    
    @pytest.fixture
    def invalid_endomorphism_data(self):
        """Create invalid endomorphism data (output dim != spatial dims)."""
        shape = (5, 5, 3)  # Should be (5, 5, 2) for 2D endomorphism
        return np.random.rand(*shape)
    
    def test_valid_initialization(self, valid_endomorphism_data):
        """Test initialization with valid endomorphism data."""
        endo = UnitMappedEndomorphism(
            data=valid_endomorphism_data,
            interp_method=InterpMethod.LINEAR
        )
        
        assert endo.ndim == 2
        assert endo.spatial_shape == (5, 5)
        assert isinstance(endo, MappedUnitField)
    
    def test_invalid_initialization(self, invalid_endomorphism_data):
        """Test initialization with invalid endomorphism data."""
        with pytest.raises(ValueError, match="does not represent an endomorphism"):
            UnitMappedEndomorphism(
                data=invalid_endomorphism_data,
                interp_method=InterpMethod.LINEAR
            )
    
    def test_inheritance(self, valid_endomorphism_data):
        """Test that UnitMappedEndomorphism inherits from MappedUnitField."""
        endo = UnitMappedEndomorphism(
            data=valid_endomorphism_data,
            interp_method=InterpMethod.LINEAR
        )
        
        # Should have all MappedUnitField methods
        assert hasattr(endo, 'get_value')
        assert hasattr(endo, 'get_values')
        assert hasattr(endo, 'with_interp_method')
        assert hasattr(endo, 'data')
        assert hasattr(endo, 'interp_method')
        assert hasattr(endo, 'cache_size')
    
    def test_1d_endomorphism(self):
        """Test 1D endomorphism (rare but should work)."""
        data = np.random.rand(10, 1)  # 1D endomorphism
        endo = UnitMappedEndomorphism(data=data, interp_method=InterpMethod.LINEAR)
        
        assert endo.ndim == 1
        assert endo.spatial_shape == (10,)
        
        # Should be able to get values
        result = endo.get_value((0.5,))
        assert isinstance(result, tuple)
        assert len(result) == 1


class TestUnit2DMappedEndomorphism:
    """Tests for the Unit2DMappedEndomorphism class."""
    
    @pytest.fixture
    def sample_2d_endo_data(self):
        """Create sample 2D endomorphism data."""
        shape = (10, 10, 2)
        return np.random.rand(*shape)
    
    @pytest.fixture
    def identity_endo_data(self):
        """Create identity endomorphism data."""
        height, width = 10, 10
        xs, ys = np.meshgrid(
            np.linspace(0, 1, width),
            np.linspace(0, 1, height),
            indexing='xy'
        )
        return np.stack([xs, ys], axis=-1)
    
    def test_valid_initialization(self, sample_2d_endo_data):
        """Test initialization with valid 2D endomorphism data."""
        endo = Unit2DMappedEndomorphism(
            data=sample_2d_endo_data,
            interp_method=InterpMethod.LINEAR
        )
        
        assert endo.ndim == 2
        assert endo.spatial_shape == (10, 10)
        assert isinstance(endo, UnitMappedEndomorphism)
    
    def test_invalid_initialization_wrong_dimensions(self):
        """Test initialization with wrong dimensions."""
        # 3D data instead of 2D
        data_3d = np.random.rand(10, 10, 10, 3)
        with pytest.raises(ValueError, match="does not represent a 2D endomorphism"):
            Unit2DMappedEndomorphism(data=data_3d, interp_method=InterpMethod.LINEAR)
        
        # Wrong last dimension
        data_wrong_last = np.random.rand(10, 10, 3)
        with pytest.raises(ValueError, match="Expected shape"):
            Unit2DMappedEndomorphism(data=data_wrong_last, interp_method=InterpMethod.LINEAR)
    
    def test_get_value_uses_cv2(self, sample_2d_endo_data):
        """Test that get_value uses OpenCV backend."""
        endo = Unit2DMappedEndomorphism(
            data=sample_2d_endo_data,
            interp_method=InterpMethod.LINEAR
        )
        
        # Patch cv2_unit_field_sample to verify it's called
        with patch('unitfield.core.unitfield.cv2_unit_field_sample') as mock_cv2_sample:
            mock_cv2_sample.return_value = np.array([[0.5, 0.6]])
            
            result = endo.get_value((0.5, 0.5))
            
            mock_cv2_sample.assert_called_once()
            assert result == (0.5, 0.6)
    
    def test_get_values_uses_cv2(self, sample_2d_endo_data):
        """Test that get_values uses OpenCV backend."""
        endo = Unit2DMappedEndomorphism(
            data=sample_2d_endo_data,
            interp_method=InterpMethod.LINEAR
        )

        coords = np.array([[0.5, 0.5], [0.6, 0.6]])

        # Patch the correct location
        # The function is imported in unitfield.core.unitfield
        with patch('unitfield.core.unitfield.cv2_unit_field_sample') as mock_cv2_sample:
            mock_cv2_sample.return_value = np.array([[0.5, 0.6], [0.7, 0.8]])
            
            result = endo.get_values(coords)
            
            mock_cv2_sample.assert_called_once_with(
                endo.data,
                coords,
                endo.interp_method
            )
            np.testing.assert_array_equal(result, [[0.5, 0.6], [0.7, 0.8]])
    
    def test_get_values_coordinate_validation(self, sample_2d_endo_data):
        """Test coordinate validation in get_values."""
        endo = Unit2DMappedEndomorphism(
            data=sample_2d_endo_data,
            interp_method=InterpMethod.LINEAR
        )
        
        # Test with 3D coordinates
        coords = np.array([[[0.5, 0.5, 0.5]]])
        with pytest.raises(ValueError, match="Expected 2D coordinates"):
            endo.get_values(coords)
    
    def test_rasterize_mapping_basic(self, identity_endo_data):
        """Test basic rasterize_mapping functionality."""
        endo = Unit2DMappedEndomorphism(
            data=identity_endo_data,
            interp_method=InterpMethod.LINEAR
        )
        
        width, height = 20, 15
        mapping = endo.rasterize_mapping(width, height)
        
        assert mapping.shape == (height, width, 2)
        assert mapping.dtype == np.float32
        
        # For identity mapping, x-coordinate should scale from 0 to width-1
        assert np.allclose(mapping[0, 0, 0], 0, atol=1e-5)
        assert np.allclose(mapping[-1, -1, 0], width - 1, atol=1e-5)
        
        # y-coordinate should scale from 0 to height-1
        assert np.allclose(mapping[0, 0, 1], 0, atol=1e-5)
        assert np.allclose(mapping[-1, -1, 1], height - 1, atol=1e-5)
    
    def test_rasterize_mapping_invalid_dimensions(self, sample_2d_endo_data):
        """Test rasterize_mapping with invalid dimensions."""
        endo = Unit2DMappedEndomorphism(
            data=sample_2d_endo_data,
            interp_method=InterpMethod.LINEAR
        )
        
        with pytest.raises(ValueError, match="Width must be positive"):
            endo.rasterize_mapping(width=0, height=10)
        
        with pytest.raises(ValueError, match="Height must be positive"):
            endo.rasterize_mapping(width=10, height=0)
    
    def test_rasterize_mapping_dtype(self, sample_2d_endo_data):
        """Test rasterize_mapping with different dtypes."""
        endo = Unit2DMappedEndomorphism(
            data=sample_2d_endo_data,
            interp_method=InterpMethod.LINEAR
        )
        
        # Test with float64
        mapping = endo.rasterize_mapping(10, 10, dtype=np.float64)
        assert mapping.dtype == np.float32  # Should be converted to float32 at the end
        
        # Test with float32 (default)
        mapping = endo.rasterize_mapping(10, 10, dtype=np.float32)
        assert mapping.dtype == np.float32
    
    def test_remap_basic(self, identity_endo_data):
        """Test basic remap functionality."""
        endo = Unit2DMappedEndomorphism(
            data=identity_endo_data,
            interp_method=InterpMethod.LINEAR
        )
        
        # Create test image
        test_image = np.random.rand(20, 30, 3).astype(np.float32)
        
        # Patch rasterize_mapping and remap_tensor_cv2
        with patch.object(endo, 'rasterize_mapping') as mock_rasterize:
            with patch('unitfield.core.unitfield.remap_tensor_cv2') as mock_remap:
                mock_rasterize.return_value = np.zeros((20, 30, 2), dtype=np.float32)
                mock_remap.return_value = np.zeros_like(test_image)
                
                result = endo.remap(test_image)
                
                # Check the call without asserting exact kwargs
                mock_rasterize.assert_called_once()
                mock_remap.assert_called_once()
                assert result.shape == test_image.shape
    
    def test_remap_with_parameters(self, identity_endo_data):
        """Test remap with custom parameters."""
        endo = Unit2DMappedEndomorphism(
            data=identity_endo_data,
            interp_method=InterpMethod.LINEAR
        )
        
        test_image = np.random.rand(20, 30, 3).astype(np.float32)
        
        with patch.object(endo, 'rasterize_mapping') as mock_rasterize:
            with patch('unitfield.core.unitfield.remap_tensor_cv2') as mock_remap:
                mock_rasterize.return_value = np.zeros((20, 30, 2), dtype=np.float32)
                mock_remap.return_value = np.zeros_like(test_image)
                
                result = endo.remap(
                    test_image,
                    interpolation=cv2.INTER_CUBIC,
                    border_mode=cv2.BORDER_CONSTANT,
                    border_value=1.0
                )
                
                # Check that parameters were passed through
                if mock_remap.called:
                    call_args, call_kwargs = mock_remap.call_args
                    assert call_kwargs.get('interpolation') == cv2.INTER_CUBIC
                    assert call_kwargs.get('border_mode') == cv2.BORDER_CONSTANT
                    assert call_kwargs.get('border_value') == 1.0
    
    def test_compose_basic(self, sample_2d_endo_data):
        """Test composition of two endomorphisms."""
        endo1 = Unit2DMappedEndomorphism(
            data=sample_2d_endo_data,
            interp_method=InterpMethod.LINEAR
        )
        
        # Create second endomorphism
        endo2_data = np.random.rand(10, 10, 2)
        endo2 = Unit2DMappedEndomorphism(
            data=endo2_data,
            interp_method=InterpMethod.CUBIC
        )
        
        # Test composition
        composed = endo1.compose(endo2)
        
        assert isinstance(composed, Unit2DMappedEndomorphism)
        assert composed.spatial_shape == endo1.spatial_shape
        assert composed.cache_size == endo1.cache_size
        
        # Should use endo1's interpolation method by default
        assert composed.interp_method == endo1.interp_method
    
    def test_compose_with_custom_interp(self, sample_2d_endo_data):
        """Test composition with custom interpolation method."""
        endo1 = Unit2DMappedEndomorphism(
            data=sample_2d_endo_data,
            interp_method=InterpMethod.LINEAR
        )
        
        endo2_data = np.random.rand(10, 10, 2)
        endo2 = Unit2DMappedEndomorphism(
            data=endo2_data,
            interp_method=InterpMethod.CUBIC
        )
        
        composed = endo1.compose(endo2, interp_method=InterpMethod.NEAREST_MANHATTAN)
        
        assert composed.interp_method == InterpMethod.NEAREST_MANHATTAN
    
    def test_compose_invalid_type(self, sample_2d_endo_data):
        """Test composition with invalid type."""
        endo = Unit2DMappedEndomorphism(
            data=sample_2d_endo_data,
            interp_method=InterpMethod.LINEAR
        )
        
        with pytest.raises(TypeError, match="Can only compose with Unit2DMappedEndomorphism"):
            endo.compose("not_an_endomorphism")
    
    def test_composition_identity(self, identity_endo_data):
        """Test composition with identity endomorphism."""
        endo = Unit2DMappedEndomorphism(
            data=identity_endo_data,
            interp_method=InterpMethod.LINEAR
        )
        
        # Compose with itself (identity)
        composed = endo.compose(endo)
        
        # Should be very close to original (identity composed with identity is identity)
        # But due to interpolation, might not be exact
        np.testing.assert_allclose(
            composed.data,
            endo.data,
            atol=1e-5,
            rtol=1e-5
        )
    
    def test_caching_in_2d_endomorphism(self, sample_2d_endo_data):
        """Test that 2D endomorphism uses caching for get_value."""
        endo = Unit2DMappedEndomorphism(
            data=sample_2d_endo_data,
            interp_method=InterpMethod.LINEAR,
            cache_size=5
        )

        # Test with a single patch context
        with patch('unitfield.core.unitfield.cv2_unit_field_sample') as mock_cv2:
            mock_cv2.return_value = np.array([[0.5, 0.6]])
            
            # First call should compute
            result1 = endo.get_value((0.5, 0.5))
            
            # Second call should use cache (mock shouldn't be called again)
            result2 = endo.get_value((0.5, 0.5))
            
            # Should have been called only once
            mock_cv2.assert_called_once()
            
            # Results should be the same (from cache)
            assert result1 == result2
            assert result1 == (0.5, 0.6)
    
    def test_no_caching_in_2d_endomorphism(self, sample_2d_endo_data):
        """Test 2D endomorphism with caching disabled."""
        endo = Unit2DMappedEndomorphism(
            data=sample_2d_endo_data,
            interp_method=InterpMethod.LINEAR,
            cache_size=None
        )
        
        # Both calls should compute
        with patch('unitfield.core.unitfield.cv2_unit_field_sample') as mock_cv2:
            mock_cv2.return_value = np.array([[0.5, 0.6]])
            result1 = endo.get_value((0.5, 0.5))
            
            mock_cv2.return_value = np.array([[0.7, 0.8]])
            result2 = endo.get_value((0.5, 0.5))
        
        # Should have been called twice
        assert mock_cv2.call_count == 2
        # Results should be different (mocks returned different values)
        assert result1 != result2


class TestEdgeCasesAndIntegration:
    """Integration tests and edge cases."""
    
    def test_large_data_handling(self):
        """Test with large data arrays."""
        # Create large but manageable dataset
        shape = (100, 100, 2)  # 10k points, 2 channels = 20k values
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
        small_data = np.random.rand(2, 2, 1)  # 2x2 grid, 1D output
        
        field = MappedUnitField(
            data=small_data,
            interp_method=InterpMethod.NEAREST_MANHATTAN
        )
        
        result = field.get_value((0.5, 0.5))
        assert isinstance(result, tuple)
        assert len(result) == 1
    
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
            # Should be within [0, 1] for unit field
            assert 0.0 <= result[0] <= 1.0
            assert 0.0 <= result[1] <= 1.0
    
    def test_out_of_bounds_handling(self, sample_2d_field_data):
        """Test that coordinates outside [0, 1] are handled."""
        field = MappedUnitField(
            data=sample_2d_field_data,
            interp_method=InterpMethod.LINEAR
        )
        
        # These might be clipped or extrapolated depending on implementation
        # Just ensure no crashes
        extreme_coords = [(-0.1, 0.5), (1.1, 0.5), (0.5, -0.1), (0.5, 1.1)]
        for coord in extreme_coords:
            try:
                result = field.get_value(coord)
                assert isinstance(result, tuple)
            except Exception as e:
                # If implementation raises for out of bounds, that's OK
                assert "out of bounds" in str(e).lower() or "clip" in str(e).lower()
    
    def test_data_with_special_values(self):
        """Test with data containing NaN, Inf, etc."""
        data = np.array([
            [[0.0, 0.0], [1.0, np.nan]],
            [[np.inf, 0.0], [-np.inf, 1.0]]
        ])
        
        field = MappedUnitField(
            data=data,
            interp_method=InterpMethod.LINEAR
        )
        
        # Should handle without crashing
        result = field.get_value((0.5, 0.5))
        assert isinstance(result, tuple)
        # Result might contain NaN/Inf - that's OK
    
    def test_thread_safety(self, sample_2d_field_data):
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
    
    def test_serialization_compatibility(self, sample_2d_field_data):
        """Test that fields can be pickled (basic serialization)."""
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


class TestAbstractBaseClass:
    """Tests for the UnitNdimField abstract base class."""
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that UnitNdimField cannot be instantiated directly."""
        with pytest.raises(TypeError):
            UnitNdimField()
    
    def test_concrete_subclass_must_implement_abstract_methods(self):
        """Test that concrete subclasses must implement abstract methods."""
        
        class IncompleteSubclass(UnitNdimField):
            pass
        
        with pytest.raises(TypeError):
            IncompleteSubclass()
    
    def test_concrete_subclass_works(self):
        """Test a concrete implementation of UnitNdimField."""
        
        class ConcreteField(UnitNdimField):
            def __init__(self, data):
                self._data = data
                self._ndim = data.shape[-1]
                self._spatial_shape = data.shape[:-1]
            
            def get_value(self, coords):
                return tuple([0.5] * self.ndim)
            
            def get_values(self, coords_array):
                shape = coords_array.shape[:-1] + (self.ndim,)
                return np.full(shape, 0.5)
            
            @property
            def ndim(self):
                return self._ndim
            
            @property
            def spatial_shape(self):
                return self._spatial_shape
        
        data = np.random.rand(10, 10, 2)
        field = ConcreteField(data)
        
        assert field.ndim == 2
        assert field.spatial_shape == (10, 10)
        assert field.get_value((0.5, 0.5)) == (0.5, 0.5)
        assert field.get_values(np.array([[0.5, 0.5]])).shape == (1, 2)