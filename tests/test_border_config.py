"""Unit tests for BorderConfig dataclass."""

import numpy as np
import pytest

from unitfield.core.border_config import BorderConfig
from unitfield.core.enums import BorderMode


class TestBorderMode:
    """BorderMode enum values and semantics."""

    def test_values(self):
        assert BorderMode.CLAMP == 0
        assert BorderMode.CONSTANT == 1
        assert BorderMode.REFLECT == 2
        assert BorderMode.WRAP == 3
        assert BorderMode.REFLECT_101 == 4
        assert BorderMode.ARRAY == 5

    def test_int_enum(self):
        assert isinstance(BorderMode.CLAMP, int)
        assert int(BorderMode.CLAMP) == 0


class TestBorderConfigDefault:
    """BorderConfig default construction."""

    def test_defaults(self):
        bc = BorderConfig()
        assert bc.mode == BorderMode.CLAMP
        assert bc.constant_value is None
        assert bc.array is None
        assert bc.feathering_width == 0.0
        assert bc.feathering_x_undershoot_multiplier == 1.0
        assert bc.feathering_x_overshoot_multiplier == 1.0
        assert bc.feathering_y_undershoot_multiplier == 1.0
        assert bc.feathering_y_overshoot_multiplier == 1.0

    def test_clamp_constructor(self):
        bc = BorderConfig.clamp()
        assert bc.mode == BorderMode.CLAMP
        assert bc.feathering_width == 0.0


class TestBorderConfigConstant:
    """CONSTANT mode."""

    def test_scalar_value(self):
        bc = BorderConfig.constant(0.5)
        assert bc.mode == BorderMode.CONSTANT
        assert bc.constant_value == 0.5

    def test_with_feathering(self):
        bc = BorderConfig.constant(0.5, feathering_width=2.0)
        assert bc.feathering_width == 2.0

    def test_per_channel_value(self):
        arr = np.array([0.1, 0.2, 0.3])
        bc = BorderConfig.constant(arr)
        np.testing.assert_array_equal(bc.constant_value, arr)


class TestBorderConfigArray:
    """ARRAY mode."""

    def test_valid_array(self):
        arr = np.zeros((32, 48, 3), dtype=np.float64)
        bc = BorderConfig.from_array(arr)
        assert bc.mode == BorderMode.ARRAY
        assert bc.array is arr

    def test_missing_array_raises(self):
        with pytest.raises(ValueError, match="ARRAY mode requires"):
            BorderConfig(mode=BorderMode.ARRAY)


class TestBorderConfigValidation:
    """Validation in __post_init__."""

    def test_negative_feathering_width(self):
        with pytest.raises(ValueError, match="feathering_width must be non-negative"):
            BorderConfig(feathering_width=-1.0)

    def test_negative_multiplier_x_under(self):
        with pytest.raises(ValueError, match="feathering multipliers must be non-negative"):
            BorderConfig(feathering_x_undershoot_multiplier=-1.0)

    def test_negative_multiplier_x_over(self):
        with pytest.raises(ValueError, match="feathering multipliers must be non-negative"):
            BorderConfig(feathering_x_overshoot_multiplier=-1.0)

    def test_negative_multiplier_y_under(self):
        with pytest.raises(ValueError, match="feathering multipliers must be non-negative"):
            BorderConfig(feathering_y_undershoot_multiplier=-1.0)

    def test_negative_multiplier_y_over(self):
        with pytest.raises(ValueError, match="feathering multipliers must be non-negative"):
            BorderConfig(feathering_y_overshoot_multiplier=-1.0)


class TestBorderConfigDerivation:
    """with_() and dataclasses.replace."""

    def test_with_changes_mode(self):
        bc = BorderConfig.clamp()
        bc2 = bc.with_(mode=BorderMode.REFLECT)
        assert bc2.mode == BorderMode.REFLECT
        assert bc.mode == BorderMode.CLAMP  # original unchanged

    def test_with_feathering(self):
        bc = BorderConfig.clamp()
        bc2 = bc.with_(feathering_width=5.0)
        assert bc2.feathering_width == 5.0


class TestBorderConfigSerialisation:
    """to_dict / from_dict round-trip."""

    def test_to_dict(self):
        bc = BorderConfig.constant(0.5, feathering_width=2.0)
        d = bc.to_dict()
        assert d["mode"] == BorderMode.CONSTANT
        assert d["constant_value"] == 0.5
        assert d["feathering_width"] == 2.0

    def test_from_dict(self):
        d = {"mode": BorderMode.REFLECT, "feathering_width": 3.0}
        bc = BorderConfig.from_dict(d)
        assert bc.mode == BorderMode.REFLECT
        assert bc.feathering_width == 3.0

    def test_from_dict_none(self):
        bc = BorderConfig.from_dict(None)
        assert bc.mode == BorderMode.CLAMP

    def test_from_dict_already_borderconfig(self):
        orig = BorderConfig.constant(0.9)
        bc = BorderConfig.from_dict(orig)
        assert bc is orig
