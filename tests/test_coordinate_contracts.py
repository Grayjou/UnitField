"""Regression tests for public coordinate and remapping contracts."""

import numpy as np
import pytest

from unitfield import (
    BorderConfig,
    BorderMode,
    InterpMethod,
    Unit1DMappedEndomorphism,
    Unit2DMappedEndomorphism,
    absolute_to_displacement,
    displacement_to_absolute,
    identity_coordinate_field,
    pixel_to_unit,
    remap_tensor,
    remap_tensor_1d,
    resample_coordinate_field,
    unit_to_pixel,
    xy_to_yx,
    yx_to_xy,
)


def test_rasterize_identity_rectangular_full_field():
    endomorphism = Unit2DMappedEndomorphism(
        identity_coordinate_field(5, 3, dtype=np.float64), InterpMethod.LINEAR
    )

    mapping = endomorphism.rasterize_mapping(width=7, height=4, dtype=np.float64)
    ys, xs = np.meshgrid(np.arange(4), np.arange(7), indexing="ij")
    expected = np.stack((xs, ys), axis=-1)

    np.testing.assert_allclose(mapping, expected, atol=1e-12)


def test_rasterize_affine_field_uses_xy_vectors_and_yx_queries():
    identity = identity_coordinate_field(5, 3, dtype=np.float64)
    field = identity.copy()
    field[..., 0] = 0.15 + 0.7 * identity[..., 0] + 0.1 * identity[..., 1]
    field[..., 1] = -0.2 + 0.25 * identity[..., 0] + 0.8 * identity[..., 1]
    endomorphism = Unit2DMappedEndomorphism(field, InterpMethod.LINEAR)

    mapping = endomorphism.rasterize_mapping(width=7, height=4, dtype=np.float64)
    ys, xs = np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 7), indexing="ij")
    expected = np.stack(
        ((0.15 + 0.7 * xs + 0.1 * ys) * 6, (-0.2 + 0.25 * xs + 0.8 * ys) * 3), axis=-1
    )

    np.testing.assert_allclose(mapping, expected, atol=1e-12)


def test_rasterize_dtype_is_output_conversion_not_grid_precision():
    endomorphism = Unit2DMappedEndomorphism(
        identity_coordinate_field(5, 3, dtype=np.float64), InterpMethod.LINEAR
    )
    mapping = endomorphism.rasterize_mapping(width=5, height=3, dtype=np.int32)
    ys, xs = np.meshgrid(np.arange(3), np.arange(5), indexing="ij")
    np.testing.assert_array_equal(mapping, np.stack((xs, ys), axis=-1))


def test_rasterize_singleton_source_axes_zero_nonidentity_components():
    field = np.empty((2, 3, 2), dtype=np.float64)
    field[..., 0] = 0.75
    field[..., 1] = 0.25
    endomorphism = Unit2DMappedEndomorphism(field, InterpMethod.LINEAR)

    width_singleton = endomorphism.rasterize_mapping(5, 4, source_shape=(4, 1))
    height_singleton = endomorphism.rasterize_mapping(5, 4, source_shape=(1, 5))

    np.testing.assert_array_equal(width_singleton[..., 0], 0.0)
    np.testing.assert_array_equal(height_singleton[..., 1], 0.0)
    np.testing.assert_allclose(width_singleton[..., 1], 0.75)
    np.testing.assert_allclose(height_singleton[..., 0], 3.0)


def test_rasterize_source_shape_is_independent_from_destination_shape():
    field = identity_coordinate_field(3, 2, dtype=np.float64)
    endomorphism = Unit2DMappedEndomorphism(field, InterpMethod.LINEAR)

    mapping = endomorphism.rasterize_mapping(7, 5, source_shape=(3, 11), dtype=np.float64)
    ys, xs = np.meshgrid(np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 7), indexing="ij")
    np.testing.assert_allclose(mapping, np.stack((xs * 10, ys * 2), axis=-1), atol=1e-12)


@pytest.mark.parametrize("shape", [(1, 5), (4, 1), (1, 1)])
def test_rasterize_identity_singleton_axes(shape):
    height, width = shape
    endomorphism = Unit2DMappedEndomorphism(
        identity_coordinate_field(width, height, dtype=np.float64), InterpMethod.LINEAR
    )
    mapping = endomorphism.rasterize_mapping(width, height, dtype=np.float64)
    ys, xs = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    np.testing.assert_array_equal(mapping, np.stack((xs, ys), axis=-1))


@pytest.mark.parametrize("interpolation", range(5))
def test_remap_2d_independent_source_destination_shape(interpolation):
    source = (10 * np.arange(3)[:, None] + np.arange(5)[None, :]).astype(np.float64)
    ys, xs = np.meshgrid(np.linspace(0, 1, 7), np.linspace(0, 1, 9), indexing="ij")
    result = remap_tensor(source, xs, ys, interpolation=interpolation)

    assert result.shape == (7, 9)
    assert result.dtype == source.dtype
    assert np.isfinite(result).all()
    np.testing.assert_allclose(
        result[[0, -1]][:, [0, -1]], source[[0, -1]][:, [0, -1]], atol=1e-12
    )


def test_different_resolution_linear_remap_has_exact_affine_values():
    source = (10 * np.arange(3)[:, None] + np.arange(5)[None, :]).astype(np.float64)
    ys, xs = np.meshgrid(np.linspace(0, 1, 7), np.linspace(0, 1, 9), indexing="ij")
    result = remap_tensor(source, xs, ys, interpolation=InterpMethod.LINEAR)
    expected = 20 * ys + 4 * xs
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_remap_2d_destination_shape_with_channels_noncontiguous_and_array_border():
    source = np.dstack((np.arange(15).reshape(3, 5), np.ones((3, 5))))[:, ::-1]
    map_x = np.full((6, 4), -0.1)
    map_y = np.full((6, 4), 0.5)
    border = np.arange(48, dtype=np.float64).reshape(6, 4, 2)
    result = remap_tensor(source, map_x, map_y, border_config=BorderConfig.from_array(border))

    assert result.shape == (6, 4, 2)
    np.testing.assert_array_equal(result, border)


@pytest.mark.parametrize(
    "border_config",
    [
        BorderConfig.clamp(),
        BorderConfig.constant(0.0),
        BorderConfig(mode=BorderMode.REFLECT),
        BorderConfig(mode=BorderMode.WRAP),
        BorderConfig(mode=BorderMode.REFLECT_101),
    ],
)
def test_remap_2d_independent_shape_supports_scalar_border_modes(border_config):
    source = np.arange(6, dtype=np.float64).reshape(2, 3)
    map_x = np.full((4, 5), 1.2)
    map_y = np.full((4, 5), -0.2)
    result = remap_tensor(source, map_x, map_y, border_config=border_config)
    assert result.shape == (4, 5)
    assert np.isfinite(result).all()


def test_different_resolution_border_and_feather_values_are_exact():
    source = (10 * np.arange(2)[:, None] + np.arange(3)[None, :]).astype(np.float64)
    map_x = np.full((4, 5), -0.25)
    map_y = np.full((4, 5), 0.5)

    constant = remap_tensor(source, map_x, map_y, border_config=BorderConfig.constant(-3.0))
    reflected = remap_tensor(
        source, map_x, map_y, border_config=BorderConfig(mode=BorderMode.REFLECT)
    )
    feathered = remap_tensor(
        np.full((2, 3), 10.0),
        np.full((4, 5), -0.1),
        np.full((4, 5), 0.5),
        border_config=BorderConfig.constant(0.0, feathering_width=0.2),
    )

    np.testing.assert_array_equal(constant, np.full((4, 5), -3.0))
    np.testing.assert_allclose(reflected, np.full((4, 5), 5.5), atol=1e-12)
    np.testing.assert_allclose(feathered, np.full((4, 5), 5.0), atol=1e-12)


@pytest.mark.parametrize("interpolation", range(5))
def test_remap_1d_independent_source_destination_length(interpolation):
    source = np.arange(5, dtype=np.float32)
    result = remap_tensor_1d(source, np.linspace(0, 1, 9), interpolation=interpolation)

    assert result.shape == (9,)
    assert result.dtype == source.dtype
    np.testing.assert_allclose(result[[0, -1]], source[[0, -1]], atol=1e-12)


def test_remap_1d_independent_length_array_border():
    source = np.ones((3, 2))
    border = np.arange(10, dtype=np.float64).reshape(5, 2)
    result = remap_tensor_1d(
        source, np.full(5, -0.1), border_config=BorderConfig.from_array(border)
    )
    np.testing.assert_array_equal(result, border)


def test_remap_nearest_rounds_half_up():
    source = np.array([10, 20, 30], dtype=np.int64)
    result = remap_tensor_1d(source, np.array([0.25, 0.75]), interpolation=0)
    np.testing.assert_array_equal(result, [20, 30])


def test_remap_rejects_nonfinite_coordinates():
    source = np.ones((2, 2))
    with pytest.raises(ValueError, match="finite"):
        remap_tensor(source, np.array([[np.nan]]), np.array([[0.0]]))
    with pytest.raises(ValueError, match="finite"):
        remap_tensor_1d(np.ones(2), np.array([np.inf]))


def test_field_queries_clip_outside_unit_domain():
    endomorphism = Unit1DMappedEndomorphism.from_envelope(np.array([0.0, 1.0]))
    np.testing.assert_allclose(endomorphism.get_values(np.array([[-0.5], [1.5]]))[:, 0], [0.0, 1.0])


def test_endomorphism_remap_can_set_destination_resolution():
    endomorphism_2d = Unit2DMappedEndomorphism(identity_coordinate_field(3, 2), InterpMethod.LINEAR)
    endomorphism_1d = Unit1DMappedEndomorphism.from_envelope(np.linspace(0, 1, 3))

    assert endomorphism_2d.remap(np.ones((2, 3)), output_shape=(4, 5)).shape == (4, 5)
    assert endomorphism_1d.remap(np.ones(3), output_length=5).shape == (5,)


def test_coordinate_conversion_and_displacement_round_trips():
    pixels = np.array([[0.0, 0.0], [2.0, 4.0]])
    unit = pixel_to_unit(pixels, (3, 5))
    np.testing.assert_array_equal(unit, [[0.0, 0.0], [1.0, 1.0]])
    np.testing.assert_array_equal(unit_to_pixel(unit, (3, 5)), pixels)
    np.testing.assert_array_equal(pixel_to_unit(np.array([[7.0, 4.0]]), (1, 5)), [[0.0, 1.0]])
    np.testing.assert_array_equal(xy_to_yx(yx_to_xy(pixels)), pixels)

    absolute = identity_coordinate_field(5, 3) + np.array([0.1, -0.2])
    np.testing.assert_allclose(displacement_to_absolute(absolute_to_displacement(absolute)), absolute)


def test_coordinate_utilities_validate_malformed_nonfinite_and_integer_inputs():
    np.testing.assert_array_equal(pixel_to_unit(np.array([[0, 4]], dtype=np.uint8), (3, 5)), [[0, 1]])
    np.testing.assert_array_equal(unit_to_pixel(np.array([[1, 1]], dtype=np.uint16), (3, 5)), [[2, 4]])

    for coords in (np.array(0.5), np.array([[0.5, 0.5, 0.5]]), np.array([[np.nan, 0.5]])):
        with pytest.raises((TypeError, ValueError)):
            pixel_to_unit(coords, (3, 5))
    with pytest.raises(TypeError):
        unit_to_pixel(np.array([[0.5, 0.5]]), (3.0, 5))
    with pytest.raises(ValueError):
        identity_coordinate_field(0, 3)
    with pytest.raises(TypeError):
        identity_coordinate_field(3, 2, dtype=np.uint8)
    with pytest.raises(ValueError):
        absolute_to_displacement(np.empty((0, 3, 2), dtype=np.float64))
    with pytest.raises(ValueError):
        displacement_to_absolute(np.full((2, 3, 2), np.inf))
    with pytest.raises(TypeError):
        resample_coordinate_field(identity_coordinate_field(3, 2), 4, 3, dtype=np.uint8)


def test_resample_coordinate_field_preserves_affine_field():
    identity = identity_coordinate_field(3, 2, dtype=np.float64)
    field = np.stack((0.2 + 0.6 * identity[..., 0], 0.1 + 0.7 * identity[..., 1]), axis=-1)
    result = resample_coordinate_field(field, 7, 5, dtype=np.float64)
    target = identity_coordinate_field(7, 5, dtype=np.float64)
    expected = np.stack((0.2 + 0.6 * target[..., 0], 0.1 + 0.7 * target[..., 1]), axis=-1)
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_compose_is_other_after_self_for_noncommuting_fields():
    identity = identity_coordinate_field(5, 4, dtype=np.float64)
    translate = identity.copy()
    translate[..., 0] += 0.1
    scale = identity.copy()
    scale[..., 0] *= 0.5
    first = Unit2DMappedEndomorphism(translate, InterpMethod.LINEAR)
    second = Unit2DMappedEndomorphism(scale, InterpMethod.LINEAR)

    composed = first.compose(second)
    coordinate = np.array([[[0.5, 0.5]]])  # query order is (y, x)
    np.testing.assert_allclose(composed.get_values(coordinate), [[[0.3, 0.5]]], atol=1e-12)


def test_composition_clips_out_of_domain_intermediate_coordinates():
    first_data = np.empty((3, 4, 2), dtype=np.float64)
    first_data[..., 0] = 1.2
    first_data[..., 1] = 0.5
    first = Unit2DMappedEndomorphism(first_data, InterpMethod.LINEAR)
    second = Unit2DMappedEndomorphism(identity_coordinate_field(7, 5), InterpMethod.LINEAR)

    composed = first.compose(second)
    np.testing.assert_allclose(composed.data, np.full((3, 4, 2), [1.0, 0.5]), atol=1e-12)

    source = np.arange(12, dtype=np.float64).reshape(3, 4)
    direct = first.remap(source, border_config=BorderConfig.constant(-1.0))
    composed_render = composed.remap(source, border_config=BorderConfig.constant(-1.0))
    np.testing.assert_array_equal(direct, np.full((3, 4), -1.0))
    np.testing.assert_array_equal(composed_render, np.full((3, 4), source[1, -1]))


def test_composition_across_resolutions_has_exact_affine_values():
    first_identity = identity_coordinate_field(4, 3, dtype=np.float64)
    second_identity = identity_coordinate_field(7, 5, dtype=np.float64)
    first = first_identity.copy()
    first[..., 0] = 0.1 + 0.5 * first_identity[..., 0]
    second = second_identity.copy()
    second[..., 0] = 0.2 + 0.5 * second_identity[..., 0]

    composed = Unit2DMappedEndomorphism(first, InterpMethod.LINEAR).compose(
        Unit2DMappedEndomorphism(second, InterpMethod.LINEAR)
    )
    expected = first.copy()
    expected[..., 0] = 0.25 + 0.25 * first_identity[..., 0]
    np.testing.assert_allclose(composed.data, expected, atol=1e-12)


def test_remap_is_deterministic_across_thread_counts():
    rng = np.random.default_rng(1234)
    source = rng.random((37, 29, 3))
    map_y, map_x = np.meshgrid(np.linspace(-0.1, 1.1, 31), np.linspace(-0.1, 1.1, 41), indexing="ij")
    border = BorderConfig.constant(0.25, feathering_width=0.15)
    single = remap_tensor(source, map_x, map_y, border_config=border, num_threads=1)
    parallel = remap_tensor(source, map_x, map_y, border_config=border, num_threads=2)
    np.testing.assert_array_equal(single, parallel)
