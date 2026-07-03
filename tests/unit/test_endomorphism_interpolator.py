import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from unitfield import (
    InterpMethod,
    Unit1DEndoInterpolator,
    Unit1dEndoInterpolator,
    Unit1DMappedEndomorphism,
    Unit2DEndoInterpolator,
    Unit2DMappedEndomorphism,
    UnitEndomorphismInterpolator,
)


def _identity_2d(height: int, width: int) -> np.ndarray:
    ys, xs = np.meshgrid(
        np.linspace(0.0, 1.0, height),
        np.linspace(0.0, 1.0, width),
        indexing='ij',
    )
    return np.stack([xs, ys], axis=-1)


class TestUnit1DEndoInterpolator:
    def test_interpolates_array_endomorphisms_with_different_sizes(self):
        start = np.array([0.0, 0.25, 1.0])
        end = np.linspace(0.0, 1.0, 5)

        interpolator = Unit1DEndoInterpolator(start, end, output_shape=(5,))
        interpolated = interpolator.interpolate(t=0.5)

        assert isinstance(interpolated, Unit1DMappedEndomorphism)
        assert interpolated.spatial_shape == (5,)
        np.testing.assert_allclose(
            interpolated.data[:, 0],
            [0.0, 0.1875, 0.375, 0.6875, 1.0],
            atol=1e-6,
        )

    def test_lowercase_alias_matches_public_name(self):
        assert Unit1dEndoInterpolator is Unit1DEndoInterpolator

    def test_interpolated_endo_can_sample(self):
        interpolator = Unit1DEndoInterpolator(
            [0.0, 0.25, 1.0],
            [0.0, 0.5, 1.0],
        )

        interpolated = interpolator.interpolate(0.5)

        assert interpolated.get_value((0.5,))[0] == pytest.approx(0.375)


class TestUnit2DEndoInterpolator:
    def test_interpolates_2d_endomorphisms_with_different_sizes(self):
        start = Unit2DMappedEndomorphism(
            _identity_2d(3, 4),
            interp_method=InterpMethod.LINEAR,
        )
        end_data = _identity_2d(5, 6)
        end_data[..., 0] = np.clip(end_data[..., 0] * 0.5, 0.0, 1.0)
        end = Unit2DMappedEndomorphism(end_data, interp_method=InterpMethod.LINEAR)

        interpolator = Unit2DEndoInterpolator(start, end, output_shape=(2, 3))
        interpolated = interpolator.interpolate(0.5)

        assert isinstance(interpolated, Unit2DMappedEndomorphism)
        assert interpolated.spatial_shape == (2, 3)
        np.testing.assert_allclose(interpolated.data[..., 0], [[0.0, 0.375, 0.75]] * 2)
        np.testing.assert_allclose(interpolated.data[..., 1], [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])


class TestUnitEndomorphismInterpolatorValidation:
    def test_rejects_mismatched_dimensions(self):
        with pytest.raises(ValueError, match="Cannot interpolate 1D and 2D"):
            UnitEndomorphismInterpolator([0.0, 1.0], _identity_2d(2, 2))

    def test_rejects_t_outside_unit_interval(self):
        interpolator = Unit1DEndoInterpolator([0.0, 1.0], [1.0, 0.0])

        with pytest.raises(ValueError, match="t must be finite"):
            interpolator.interpolate(1.5)

    def test_rejects_wrong_output_shape_dimension(self):
        with pytest.raises(ValueError, match="Expected output shape with 1 dimensions"):
            Unit1DEndoInterpolator([0.0, 1.0], [1.0, 0.0], output_shape=(2, 2))
