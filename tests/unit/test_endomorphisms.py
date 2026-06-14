"""
Tests for endomorphism classes.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


import numpy as np
import pytest

from unitfield.core.enums import InterpMethod
from unitfield.core.unitfield import Unit2DMappedEndomorphism, UnitMappedEndomorphism


@pytest.fixture
def valid_endomorphism_data():
    """Create valid endomorphism data (output dim equals spatial dims)."""
    shape = (5, 5, 2)  # 2D endomorphism
    return np.random.rand(*shape)


@pytest.fixture
def invalid_endomorphism_data():
    """Create invalid endomorphism data (output dim != spatial dims)."""
    shape = (5, 5, 3)  # Should be (5, 5, 2) for 2D endomorphism
    return np.random.rand(*shape)


@pytest.fixture
def identity_endo_data():
    """Create identity endomorphism data."""
    height, width = 10, 10
    xs, ys = np.meshgrid(
        np.linspace(0, 1, width),
        np.linspace(0, 1, height),
        indexing='xy'
    )
    return np.stack([xs, ys], axis=-1)


class TestUnitMappedEndomorphism:
    """Tests for the UnitMappedEndomorphism class."""

    def test_valid_initialization(self, valid_endomorphism_data):
        """Test initialization with valid endomorphism data."""
        endo = UnitMappedEndomorphism(
            data=valid_endomorphism_data,
            interp_method=InterpMethod.LINEAR
        )

        assert endo.ndim == 2
        assert endo.spatial_shape == (5, 5)

    def test_invalid_initialization(self, invalid_endomorphism_data):
        """Test initialization with invalid endomorphism data."""
        with pytest.raises(ValueError, match="does not represent an endomorphism"):
            UnitMappedEndomorphism(
                data=invalid_endomorphism_data,
                interp_method=InterpMethod.LINEAR
            )

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

    def test_valid_initialization(self, valid_endomorphism_data):
        """Test initialization with valid 2D endomorphism data."""
        endo = Unit2DMappedEndomorphism(
            data=valid_endomorphism_data,
            interp_method=InterpMethod.LINEAR
        )

        assert endo.ndim == 2
        assert endo.spatial_shape == (5, 5)

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

    def test_get_value(self, valid_endomorphism_data):
        """Test get_value returns correct shape and type."""
        endo = Unit2DMappedEndomorphism(
            data=valid_endomorphism_data,
            interp_method=InterpMethod.LINEAR
        )
        result = endo.get_value((0.5, 0.5))
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(v, float) for v in result)

    def test_get_values(self, valid_endomorphism_data):
        """Test get_values returns correct shape."""
        endo = Unit2DMappedEndomorphism(
            data=valid_endomorphism_data,
            interp_method=InterpMethod.LINEAR
        )
        coords = np.array([[0.5, 0.5], [0.6, 0.6]])
        result = endo.get_values(coords)
        assert result.shape == (2, 2)


class TestRasterizeMapping:
    """Tests for rasterize_mapping method."""

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

    def test_rasterize_mapping_invalid_dimensions(self, valid_endomorphism_data):
        """Test rasterize_mapping with invalid dimensions."""
        endo = Unit2DMappedEndomorphism(
            data=valid_endomorphism_data,
            interp_method=InterpMethod.LINEAR
        )

        with pytest.raises(ValueError, match="Width must be positive"):
            endo.rasterize_mapping(width=0, height=10)

        with pytest.raises(ValueError, match="Height must be positive"):
            endo.rasterize_mapping(width=10, height=0)


class TestRemapMethod:
    """Tests for remap method."""

    def test_remap_basic(self, identity_endo_data):
        """Test basic remap functionality."""
        endo = Unit2DMappedEndomorphism(
            data=identity_endo_data,
            interp_method=InterpMethod.LINEAR
        )
        test_image = np.random.rand(20, 30, 3).astype(np.float64)
        result = endo.remap(test_image, interpolation=1)
        assert result.shape == test_image.shape

    def test_remap_with_parameters(self, identity_endo_data):
        """Test remap with different interpolation modes."""
        endo = Unit2DMappedEndomorphism(
            data=identity_endo_data,
            interp_method=InterpMethod.LINEAR
        )
        test_image = np.random.rand(20, 30, 3).astype(np.float64)
        r0 = endo.remap(test_image, interpolation=0)
        r1 = endo.remap(test_image, interpolation=1)
        assert r0.shape == test_image.shape
        assert r1.shape == test_image.shape


class TestCompose:
    """Tests for compose method."""

    def test_compose_basic(self, valid_endomorphism_data):
        """Test composition of two endomorphisms."""
        endo1 = Unit2DMappedEndomorphism(
            data=valid_endomorphism_data,
            interp_method=InterpMethod.LINEAR
        )

        endo2_data = np.random.rand(5, 5, 2)
        endo2 = Unit2DMappedEndomorphism(
            data=endo2_data,
            interp_method=InterpMethod.CUBIC
        )

        composed = endo1.compose(endo2)

        assert isinstance(composed, Unit2DMappedEndomorphism)
        assert composed.spatial_shape == endo1.spatial_shape
        assert composed.cache_size == endo1.cache_size
        assert composed.interp_method == endo1.interp_method

    def test_compose_with_custom_interp(self, valid_endomorphism_data):
        """Test composition with custom interpolation method."""
        endo1 = Unit2DMappedEndomorphism(
            data=valid_endomorphism_data,
            interp_method=InterpMethod.LINEAR
        )

        endo2_data = np.random.rand(5, 5, 2)
        endo2 = Unit2DMappedEndomorphism(
            data=endo2_data,
            interp_method=InterpMethod.CUBIC
        )

        composed = endo1.compose(endo2, interp_method=InterpMethod.NEAREST_MANHATTAN)

        assert composed.interp_method == InterpMethod.NEAREST_MANHATTAN

    def test_compose_invalid_type(self, valid_endomorphism_data):
        """Test composition with invalid type."""
        endo = Unit2DMappedEndomorphism(
            data=valid_endomorphism_data,
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

        composed = endo.compose(endo)

        # Should be very close to original (identity composed with identity is identity)
        np.testing.assert_allclose(
            composed.data,
            endo.data,
            atol=1e-5,
            rtol=1e-5
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
