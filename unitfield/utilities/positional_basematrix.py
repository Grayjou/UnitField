
import numpy as np

def positional_basematrix2d(width, height = None) -> np.ndarray:
    if height is None:
        try:
            width, height = width
        except TypeError:
            height = width
    return np.stack(np.indices((height, width), dtype=np.int32)[::-1], axis=-1)

pbm_2d = positional_basematrix2d

def unit_positional_basematrix2d(width, height=None) -> np.ndarray:
    if height is None:
        try:
            width, height = width
        except TypeError:
            height = width

    ys, xs = np.indices((height, width), dtype=np.float32)
    if width > 1:
        xs /= (width - 1)
    if height > 1:
        ys /= (height - 1)

    return np.stack((xs, ys), axis=-1)


upbm_2d = unit_positional_basematrix2d

def positional_basematrix_ndim(*dimlens) -> np.ndarray:
    return np.stack(np.indices(dimlens, dtype=np.int32), axis=-1)

pbm_ndim = positional_basematrix_ndim

def unit_positional_basematrix_ndim(*dimlens) -> np.ndarray:
    grids = np.indices(dimlens, dtype=np.float32)
    for i, length in enumerate(dimlens):
        if length > 1:
            grids[i] /= (length - 1)
    return np.stack(grids, axis=-1)

upbm_ndim = unit_positional_basematrix_ndim