# distances.py
import itertools
import numpy as np
from typing import List, Tuple, Literal
from boundednumbers import UnitFloat
from .types import VectorReturnType

def get_hypercube_points(coords, spatial_shape):
    base_indices = []
    fractions = []
    
    for coord, size in zip(coords, spatial_shape):
        p = coord * (size - 1)
        i0 = int(np.floor(p))
        i1 = min(i0 + 1, size - 1)
        
        # Handle edge case when p is exactly at an integer
        if i0 == i1:
            i1 = i0
        
        base_indices.append((i0, i1))
        fractions.append(p - i0)

    corners = []
    ndim = len(spatial_shape)

    for mask in itertools.product((0, 1), repeat=ndim):
        corner = tuple(
            base_indices[d][mask[d]]
            for d in range(ndim)
        )
        corners.append((corner, mask))

    return corners, tuple(fractions)  #


# ------------------------------------------------------------------ #
# Interpolation methods
# ------------------------------------------------------------------ #


def _return_closest_point_as(
    closest_point: np.ndarray,
    return_type: Literal["array", "tuple", "list"] = "tuple"
    ) -> VectorReturnType:
    if return_type == "array":
        return closest_point
    elif return_type == "list":
        return closest_point.tolist()
    else:  # default to tuple
        return tuple(closest_point)
    


def nearest_manhattan_arr(
    coords: Tuple[UnitFloat, ...],
    spatial_shape: Tuple[int, ...],
    data: np.ndarray,

) -> np.ndarray:
    """
    Nearest-cell sampling using Manhattan projection.
    Each axis is quantized independently.
    """
    indices = tuple(
        int(np.round(coord * (size - 1)))  # Use np.round for proper rounding
        for coord, size in zip(coords, spatial_shape)
    )

    mapped = data[indices]  # shape: (ndim,)
    return mapped

def nearest_manhattan(
    coords: Tuple[UnitFloat, ...],
    spatial_shape: Tuple[int, ...],
    data: np.ndarray,
    return_type: Literal["array", "tuple", "list"] = "tuple"
) -> VectorReturnType:
    mapped = nearest_manhattan_arr(coords, spatial_shape, data)
    return _return_closest_point_as(mapped, return_type)

# distances.py, line 45-52 - FIXED VERSION:

def nearest_euclidean_arr(coords, spatial_shape, data, 
):
    corners, _ = get_hypercube_points(coords, spatial_shape)
    
    p = np.array([
        coord * (size - 1)
        for coord, size in zip(coords, spatial_shape)
    ])
    
    best_corner, _ = min(
        corners,
        key=lambda corner_mask: np.linalg.norm(p - np.array(corner_mask[0]))
    )
    
    return data[best_corner]

def nearest_euclidean(
    coords: Tuple[UnitFloat, ...],
    spatial_shape: Tuple[int, ...],
    data: np.ndarray,
    return_type: Literal["array", "tuple", "list"] = "tuple"
) -> VectorReturnType:
    mapped = nearest_euclidean_arr(coords, spatial_shape, data)
    return _return_closest_point_as(mapped, return_type)

def n_linear_arr(coords, spatial_shape, data):
    corners, fractions = get_hypercube_points(coords, spatial_shape)
    
    result = np.zeros(data.shape[-1], dtype=data.dtype)
    
    for corner, mask in corners:
        weight = 1.0
        for d, t in enumerate(fractions):
            if mask[d] == 0:
                weight *= (1 - t)
            else:
                weight *= t
        
        result += weight * data[corner]
    
    return result

def n_linear(
    coords: Tuple[UnitFloat, ...],
    spatial_shape: Tuple[int, ...],
    data: np.ndarray,
    return_type: Literal["array", "tuple", "list"] = "tuple"
) -> VectorReturnType:
    mapped = n_linear_arr(coords, spatial_shape, data)
    return _return_closest_point_as(mapped, return_type)


def n_linear_vectorized_arr(coords, spatial_shape, data):
    """Vectorized version of n-linear interpolation."""
    ndim = len(spatial_shape)
    
    # Calculate indices and fractions
    indices = []
    fractions = []
    for coord, size in zip(coords, spatial_shape):
        p = coord * (size - 1)
        i0 = int(np.floor(p))
        i1 = min(i0 + 1, size - 1)
        if i0 == i1:
            i1 = i0
        indices.append((i0, i1))
        fractions.append(p - i0)
    
    # Initialize result
    result = np.zeros(data.shape[-1], dtype=data.dtype)
    
    # Generate all combinations using itertools.product
    import itertools
    for mask in itertools.product([0, 1], repeat=ndim):
        # Get corner indices
        corner_idx = tuple(indices[d][mask[d]] for d in range(ndim))
        
        # Calculate weight for this corner
        weight = 1.0
        for d in range(ndim):
            if mask[d] == 0:
                weight *= (1 - fractions[d])
            else:
                weight *= fractions[d]
        
        result += weight * data[corner_idx]
    
    return result

def n_linear_vectorized(
    coords: Tuple[UnitFloat, ...],
    spatial_shape: Tuple[int, ...],
    data: np.ndarray,
    return_type: Literal["array", "tuple", "list"] = "tuple"
) -> VectorReturnType:
    mapped = n_linear_vectorized_arr(coords, spatial_shape, data)
    return _return_closest_point_as(mapped, return_type)

