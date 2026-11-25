"""
Interpolation backends for unit field transformations.
"""

from .interp_np import get_numpy_interpolator, np_interp_dict

try:
    from .interp_cv2 import cv2_interp_dict, cv2_unit_field_sample
except ModuleNotFoundError as exc:
    if exc.name != "cv2":
        raise
    _cv2_import_error = exc
    cv2_interp_dict = {}

    def cv2_unit_field_sample(*_args, **_kwargs):
        """Raise a clear error when the optional OpenCV backend is unavailable."""
        raise ImportError("OpenCV support requires installing unitfield[cv2]") from _cv2_import_error

__all__ = [
    'np_interp_dict',
    'cv2_interp_dict',
    'cv2_unit_field_sample',
    'get_numpy_interpolator'
]
