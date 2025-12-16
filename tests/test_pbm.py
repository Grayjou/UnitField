from ..unitfield.utilities import positional_basematrix2d, unit_positional_basematrix2d, positional_basematrix_ndim
from ..unitfield import U2DE
import numpy as np
import cv2
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
    l = 10
    upbm = unit_positional_basematrix2d(l+1, l+1)
    base = np.full((l,l), True, dtype=bool)
    def circular_mask(upbm, center = l//2, radius = 0.5):
        cx, cy = center
        dist = np.sqrt((upbm[...,0] - cx)**2 + (upbm[...,1] - cy)**2)
        mask = dist <= radius
        x, y = upbm[...,0], upbm[...,1]
        x = np.where(mask, x, -1) # Border value -1
        y = np.where(mask, y, -1)
        return np.stack((x, y), axis=-1)
    mask = circular_mask(upbm, center=(0.5, 0.5), radius=0.5)
    u2de = U2DE(mask)
    remapped = u2de.remap(base, border_mode=cv2.BORDER_CONSTANT, border_value=False)
    assert remapped[0,0] == False
    assert remapped[l-1, l-1] == False
    assert np.sum(remapped) > 0
    assert np.sum(remapped) < l * l*0.7853 # This number is not suspicious at all :D
