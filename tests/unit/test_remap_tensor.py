"""
Tests for the remap_tensor_cv2 function.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pytest
import numpy as np
import cv2
from unittest.mock import patch

from unitfield.core.unitfield import remap_tensor_cv2


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
