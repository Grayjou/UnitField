"""
Integration tests for UnitField.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pytest
import numpy as np
import cv2

from unitfield.core.enums import InterpMethod
from unitfield.core.unitfield import (
    MappedUnitField,
    Unit2DMappedEndomorphism,
    remap_tensor_cv2
)


class TestEndToEndWorkflow:
    """End-to-end integration tests."""
    
    def test_simple_image_warp(self):
        """Test a simple image warping workflow."""
        # Create identity field
        height, width = 50, 50
        xs, ys = np.meshgrid(
            np.linspace(0, 1, width),
            np.linspace(0, 1, height),
            indexing='xy'
        )
        identity = np.stack([xs, ys], axis=-1)
        
        # Create endomorphism
        endo = Unit2DMappedEndomorphism(
            data=identity,
            interp_method=InterpMethod.LINEAR
        )
        
        # Create test image
        image = np.random.rand(50, 50, 3).astype(np.float32)
        
        # Apply transformation
        result = endo.remap(image)
        
        # For identity, result should be very similar to input
        assert result.shape == image.shape
        np.testing.assert_allclose(result, image, rtol=0.1, atol=0.1)
    
    def test_composition_workflow(self):
        """Test composing multiple transformations."""
        # Create two transformations
        h, w = 20, 20
        xs, ys = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h), indexing='xy')
        
        # First transformation: slight shift
        data1 = np.stack([
            np.clip(xs + 0.1, 0, 1),
            np.clip(ys + 0.1, 0, 1)
        ], axis=-1)
        
        # Second transformation: slight rotation effect
        center_x, center_y = 0.5, 0.5
        dx = xs - center_x
        dy = ys - center_y
        angle = 0.1  # Small rotation
        data2 = np.stack([
            center_x + dx * np.cos(angle) - dy * np.sin(angle),
            center_y + dx * np.sin(angle) + dy * np.cos(angle)
        ], axis=-1)
        
        # Create endomorphisms
        endo1 = Unit2DMappedEndomorphism(data=data1)
        endo2 = Unit2DMappedEndomorphism(data=data2)
        
        # Compose
        composed = endo1.compose(endo2)
        
        # Should be able to query composed field
        result = composed.get_value((0.5, 0.5))
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_batch_processing(self):
        """Test processing large batches of coordinates."""
        # Create field
        data = np.random.rand(10, 10, 2)
        field = MappedUnitField(data=data, interp_method=InterpMethod.LINEAR)
        
        # Create large batch of coordinates
        batch_size = 10000
        coords = np.random.rand(batch_size, 2)
        
        # Process batch
        results = field.get_values(coords)
        
        assert results.shape == (batch_size, 2)
        assert np.all(np.isfinite(results))
    
    def test_switching_interpolation_methods(self):
        """Test switching between different interpolation methods."""
        # Create field
        xs, ys = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10), indexing='ij')
        data = np.stack([xs, ys], axis=-1)
        
        # Test coordinate
        test_coord = (0.333, 0.667)
        
        # Create field with each method and compare
        results = {}
        for method in [InterpMethod.NEAREST_MANHATTAN, InterpMethod.LINEAR, InterpMethod.CUBIC]:
            field = MappedUnitField(data=data, interp_method=method)
            results[method] = field.get_value(test_coord)
        
        # All should return valid results
        for method, result in results.items():
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert all(0 <= v <= 1 for v in result)
    
    def test_multidimensional_field(self):
        """Test with higher dimensional fields."""
        # Create 3D field
        data_3d = np.random.rand(5, 5, 5, 3)
        field_3d = MappedUnitField(data=data_3d, interp_method=InterpMethod.LINEAR)
        
        # Query single point
        result = field_3d.get_value((0.5, 0.5, 0.5))
        assert isinstance(result, tuple)
        assert len(result) == 3
        
        # Query multiple points
        coords = np.random.rand(10, 3)
        results = field_3d.get_values(coords)
        assert results.shape == (10, 3)


class TestBackendConsistency:
    """Test consistency between NumPy and OpenCV backends."""
    
    def test_2d_field_consistency(self):
        """Test that 2D fields give consistent results between backends."""
        # Create 2D field
        h, w = 20, 20
        # Note: Use ij indexing to match NumPy array indexing convention
        xs, ys = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h), indexing='ij')
        data = np.stack([xs * 0.8 + 0.1, ys * 0.8 + 0.1], axis=-1)
        
        # Create with MappedUnitField (NumPy backend)
        field_np = MappedUnitField(data=data, interp_method=InterpMethod.LINEAR)
        
        # Create with Unit2DMappedEndomorphism (OpenCV backend)
        field_cv = Unit2DMappedEndomorphism(data=data, interp_method=InterpMethod.LINEAR)
        
        # Test several points
        test_coords = [
            (0.25, 0.25),
            (0.5, 0.5),
            (0.75, 0.75),
        ]
        
        for coord in test_coords:
            result_np = field_np.get_value(coord)
            result_cv = field_cv.get_value(coord)
            
            # Should be very close
            np.testing.assert_allclose(result_np, result_cv, rtol=0.01, atol=0.01)


class TestRealWorldScenarios:
    """Tests for real-world usage scenarios."""
    
    def test_image_distortion_correction(self):
        """Test correcting lens distortion."""
        # Create barrel distortion field
        h, w = 64, 64
        y, x = np.ogrid[:h, :w]
        center_x, center_y = w / 2, h / 2
        
        # Calculate distance from center
        dx = (x - center_x) / center_x
        dy = (y - center_y) / center_y
        r = np.sqrt(dx**2 + dy**2)
        
        # Apply inverse barrel distortion
        k = 0.2  # Distortion coefficient
        distortion = 1 / (1 + k * r**2)
        new_x = center_x + dx * distortion * center_x
        new_y = center_y + dy * distortion * center_y
        
        # Normalize to unit space
        unit_x = new_x / (w - 1)
        unit_y = new_y / (h - 1)
        distortion_field = np.stack([unit_x, unit_y], axis=-1).astype(np.float32)
        
        # Create endomorphism
        endo = Unit2DMappedEndomorphism(data=distortion_field)
        
        # Create test pattern
        test_image = np.zeros((h, w, 3), dtype=np.float32)
        test_image[h//4:3*h//4, w//4:3*w//4] = 1.0  # White square in center
        
        # Apply correction
        corrected = endo.remap(test_image)
        
        assert corrected.shape == test_image.shape
        assert np.all(np.isfinite(corrected))
    
    def test_coordinate_transformation_grid(self):
        """Test creating and querying a coordinate transformation grid."""
        # Create a polar-to-cartesian transformation field
        h, w = 32, 32
        theta = np.linspace(0, 2 * np.pi, w)
        r = np.linspace(0, 1, h)
        theta_grid, r_grid = np.meshgrid(theta, r)
        
        # Convert to cartesian
        x = 0.5 + 0.5 * r_grid * np.cos(theta_grid)
        y = 0.5 + 0.5 * r_grid * np.sin(theta_grid)
        
        # Clip to [0, 1]
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        
        polar_to_cart = np.stack([x, y], axis=-1)
        
        # Create field
        field = MappedUnitField(data=polar_to_cart, interp_method=InterpMethod.LINEAR)
        
        # Query at various points
        test_points = np.random.rand(100, 2)
        results = field.get_values(test_points)
        
        assert results.shape == (100, 2)
        assert np.all(results >= 0)
        assert np.all(results <= 1)
    
    def test_multi_stage_pipeline(self):
        """Test a multi-stage image processing pipeline."""
        # Stage 1: Create base field
        h, w = 30, 30
        xs, ys = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h), indexing='xy')
        base_data = np.stack([xs, ys], axis=-1)
        
        base_field = MappedUnitField(data=base_data, interp_method=InterpMethod.LINEAR)
        
        # Stage 2: Switch interpolation method
        cubic_field = base_field.with_interp_method(InterpMethod.CUBIC)
        
        # Stage 3: Query with different coordinate sets
        coords_1 = np.random.rand(50, 2)
        coords_2 = np.random.rand(50, 2)
        
        results_1 = cubic_field.get_values(coords_1)
        results_2 = cubic_field.get_values(coords_2)
        
        assert results_1.shape == (50, 2)
        assert results_2.shape == (50, 2)
        
        # Results should be different for different coordinates
        assert not np.allclose(results_1, results_2)


class TestErrorRecovery:
    """Tests for error handling and recovery."""
    
    def test_graceful_handling_of_edge_cases(self):
        """Test that edge cases are handled gracefully."""
        # Create minimal field
        data = np.array([[[0.0, 0.0]], [[1.0, 1.0]]])
        field = MappedUnitField(data=data, interp_method=InterpMethod.LINEAR)
        
        # Should handle boundary queries
        result = field.get_value((0.0, 0.0))
        assert isinstance(result, tuple)
        
        result = field.get_value((1.0, 1.0))
        assert isinstance(result, tuple)
    
    def test_recovery_after_invalid_query(self):
        """Test that field remains usable after invalid queries."""
        data = np.random.rand(10, 10, 2)
        field = MappedUnitField(data=data, interp_method=InterpMethod.LINEAR)
        
        # Make invalid query
        try:
            field.get_value((0.5, 0.5, 0.5))  # Wrong dimension
        except ValueError:
            pass
        
        # Field should still work
        result = field.get_value((0.5, 0.5))
        assert isinstance(result, tuple)
        assert len(result) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
