import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unitfield import resize_nearest, resize_unit_field_nearest


def test_resize_nearest_2d_upscale_endpoint_aligned():
    image = np.array([[0, 1], [2, 3]], dtype=np.uint8)

    result = resize_nearest(image, width=3, height=3)

    expected = np.array(
        [
            [0, 1, 1],
            [2, 3, 3],
            [2, 3, 3],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == image.dtype
    assert result.flags.c_contiguous


def test_resize_nearest_2d_downscale_endpoint_aligned():
    image = np.arange(16, dtype=np.int16).reshape(4, 4)

    result = resize_nearest(image, width=2, height=2)

    np.testing.assert_array_equal(result, image[[0, 3]][:, [0, 3]])


def test_resize_nearest_3d_preserves_channels():
    image = np.arange(2 * 3 * 2, dtype=np.float32).reshape(2, 3, 2)

    result = resize_nearest(image, width=5, height=4)

    assert result.shape == (4, 5, 2)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result[0, 0], image[0, 0])
    np.testing.assert_array_equal(result[-1, -1], image[-1, -1])


def test_resize_nearest_can_convert_to_default_dtype():
    image = np.arange(4, dtype=np.uint8).reshape(2, 2)

    result = resize_nearest(image, width=2, height=2, preserve_dtype=False)

    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, image.astype(np.float32))


def test_resize_unit_field_nearest_resizes_map_channels():
    ys, xs = np.meshgrid(
        np.linspace(0.0, 1.0, 2, dtype=np.float32),
        np.linspace(0.0, 1.0, 2, dtype=np.float32),
        indexing='ij',
    )
    field = np.stack([xs, ys], axis=-1)

    result = resize_unit_field_nearest(field, width=4, height=3)

    assert result.shape == (3, 4, 2)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result[0, 0], [0.0, 0.0])
    np.testing.assert_array_equal(result[-1, -1], [1.0, 1.0])


def test_resize_nearest_rejects_unsupported_rank():
    with pytest.raises(ValueError, match="image must be 2-D or 3-D"):
        resize_nearest(np.zeros((2, 2, 2, 2)), width=2, height=2)


def test_resize_unit_field_nearest_rejects_non_field_rank():
    with pytest.raises(ValueError, match="field must have shape"):
        resize_unit_field_nearest(np.zeros((2, 2)), width=2, height=2)


def test_resize_nearest_rejects_non_positive_target_dimensions():
    image = np.zeros((2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="width must be positive"):
        resize_nearest(image, width=0, height=2)
    with pytest.raises(ValueError, match="height must be positive"):
        resize_nearest(image, width=2, height=0)
