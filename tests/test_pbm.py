import numpy as np

from ..unitfield import U2DE, BorderConfig, flat_1d_pbm, flat_1d_upbm, upbm_ndim
from ..unitfield.utilities import (
    positional_basematrix2d,
    positional_basematrix_ndim,
    unit_positional_basematrix2d,
)


def test_positional_basematrix2d():
    pbm = positional_basematrix2d(3, 2)
    expected = [[[0, 0], [1, 0], [2, 0]],
                [[0, 1], [1, 1], [2, 1]]]
    assert (pbm == expected).all()

    pbm_square = positional_basematrix2d(2)
    expected_square = [[[0, 0], [1, 0]],
                       [[0, 1], [1, 1]]]
    assert (pbm_square == expected_square).all()

def test_unit_positional_basematrix2d():
    upbm = unit_positional_basematrix2d(3, 2)
    expected = [[[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
                [[0.0, 1.0], [0.5, 1.0], [1.0, 1.0]]]
    assert np.allclose(upbm, expected)

    upbm_square = unit_positional_basematrix2d(2)
    expected_square = [[[0.0, 0.0], [1.0, 0.0]],
                       [[0.0, 1.0], [1.0, 1.0]]]
    assert np.allclose(upbm_square, expected_square)

def test_positional_basematrix_ndim():
    pbm_3d = positional_basematrix_ndim(2, 2, 2)
    expected_3d = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ])
    assert np.array_equal(pbm_3d.reshape(-1, 3), expected_3d)

def test_identity_unit_2d_endomorphism():
    width, height = 4, 4
    pbm = unit_positional_basematrix2d(width, height)
    u2de = U2DE(pbm)
    unit_base = unit_positional_basematrix2d(width, height)
    transformed = u2de.remap(unit_base)
    assert np.allclose(transformed, unit_base)

def test_circular_mask():
    size = 10
    upbm = unit_positional_basematrix2d(size + 1, size + 1)
    base = np.full((size, size), True, dtype=bool)
    def circular_mask(upbm, center=size // 2, radius=0.5):
        cx, cy = center
        dist = np.sqrt((upbm[...,0] - cx)**2 + (upbm[...,1] - cy)**2)
        mask = dist <= radius
        x, y = upbm[...,0], upbm[...,1]
        x = np.where(mask, x, -1)
        y = np.where(mask, y, -1)
        return np.stack((x, y), axis=-1)
    mask = circular_mask(upbm, center=(0.5, 0.5), radius=0.5)
    u2de = U2DE(mask)
    remapped = u2de.remap(base, border_config=BorderConfig.constant(value=0.0))
    assert not remapped[0,0]
    assert not remapped[size - 1, size - 1]
    assert np.sum(remapped) > 0
    assert np.sum(remapped) < size * size * 0.7853

def test_upbm_1d():
    length = 5
    upbm = upbm_ndim(length)
    print("upbm shape:", upbm.shape)
    expected = np.array([[0.0], [0.25], [0.5], [0.75], [1.0]])
    assert np.allclose(upbm, expected)

def test_upbm_shapes():
    length = 4
    upbm_1d = upbm_ndim(length)
    print("upbm_1d shape:", upbm_1d.shape)
    assert upbm_1d.shape == (length, 1)

    upbm_2d = upbm_ndim(length, length)
    print("upbm_2d shape:", upbm_2d.shape)
    assert upbm_2d.shape == (length, length, 2)

    upbm_3d = upbm_ndim(length, length, length)
    print("upbm_3d shape:", upbm_3d.shape)
    assert upbm_3d.shape == (length, length, length, 3)

def test_flat_pbm():
    length = 6
    pbm = flat_1d_pbm(length)
    expected = np.array([0, 1, 2, 3, 4, 5])
    assert pbm.shape == (length,)
    assert np.array_equal(pbm, expected)
def test_flat_upbm():
    length = 6
    upbm = flat_1d_upbm(length)
    expected = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float32)
    assert upbm.shape == (length,)
    assert np.allclose(upbm, expected)
