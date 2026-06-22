"""Unit tests for BorderConfig dataclass."""

import json

import numpy as np
import pytest

from unitfield import DistanceMode, RemapBorderConfig
from unitfield.core.border_config import BorderConfig
from unitfield.core.enums import BorderMode, FeatheringMode


class TestBorderMode:
    """BorderMode enum values and semantics."""

    def test_values(self):
        assert BorderMode.CLAMP == 0
        assert BorderMode.CONSTANT == 1
        assert BorderMode.REFLECT == 2
        assert BorderMode.WRAP == 3
        assert BorderMode.REFLECT_101 == 4
        assert BorderMode.ARRAY == 5


class TestFeatheringMode:
    """FeatheringMode enum values and semantics."""

    def test_values(self):
        assert FeatheringMode.BEYOND_BORDER == 0
        assert FeatheringMode.TO_BORDER == 1

    def test_int_enum(self):
        assert isinstance(BorderMode.CLAMP, int)
        assert int(BorderMode.CLAMP) == 0


class TestDistanceMode:
    """DistanceMode enum values and coercion."""

    def test_values(self):
        assert DistanceMode.MAX_NORM == 1
        assert DistanceMode.MANHATTAN == 2
        assert DistanceMode.SCALED_MANHATTAN == 3
        assert DistanceMode.ALPHA_MAX == 4
        assert DistanceMode.ALPHA_MAX_SIMPLE == 5
        assert DistanceMode.TAYLOR == 6
        assert DistanceMode.EUCLIDEAN == 7
        assert DistanceMode.WEIGHTED_MINMAX == 8

    def test_string_coercion(self):
        bc = BorderConfig(distance_mode="alpha_max")
        assert bc.distance_mode == DistanceMode.ALPHA_MAX

    def test_int_coercion(self):
        bc = BorderConfig(distance_mode=7)
        assert bc.distance_mode == DistanceMode.EUCLIDEAN

    def test_bad_string_raises(self):
        with pytest.raises(ValueError, match="Unknown distance mode"):
            BorderConfig(distance_mode="not_a_distance")


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
        assert bc.feathering_mode == FeatheringMode.BEYOND_BORDER
        assert bc.preserve_non_feather_dims is False

    def test_remap_border_config_alias(self):
        assert RemapBorderConfig is BorderConfig

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

    def test_with_feathering_mode(self):
        bc = BorderConfig.constant(0.5, feathering_mode=FeatheringMode.TO_BORDER)
        assert bc.feathering_mode == FeatheringMode.TO_BORDER

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

    def test_negative_multiplier_cancels_feathering(self):
        """Negative multipliers cancel feathering (pure border)."""
        bc = BorderConfig(feathering_x_undershoot_multiplier=-1.0)
        assert bc.feathering_x_undershoot_multiplier == -1.0

    def test_negative_multiplier_all_four(self):
        bc = BorderConfig(
            feathering_x_undershoot_multiplier=-2.0,
            feathering_x_overshoot_multiplier=-1.5,
            feathering_y_undershoot_multiplier=-3.0,
            feathering_y_overshoot_multiplier=-1.0,
        )
        assert bc.feathering_x_undershoot_multiplier == -2.0
        assert bc.feathering_x_overshoot_multiplier == -1.5
        assert bc.feathering_y_undershoot_multiplier == -3.0
        assert bc.feathering_y_overshoot_multiplier == -1.0

    def test_feather_channels_alias(self):
        bc = BorderConfig(feather_channels=[0, 0, 0, 1])
        np.testing.assert_array_equal(
            bc.feather_dims,
            np.array([False, False, False, True]),
        )

    def test_feather_dims_and_channels_same_after_bool_coercion(self):
        bc = BorderConfig(
            feather_dims=[False, False, False, True],
            feather_channels=[0, 0, 0, 2],
        )
        np.testing.assert_array_equal(
            bc.feather_dims,
            np.array([False, False, False, True]),
        )

    def test_feather_dims_and_channels_conflict_raises(self):
        with pytest.raises(ValueError, match="feather_dims and feather_channels"):
            BorderConfig(feather_dims=[False, True], feather_channels=[True, False])


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

    def test_with_feather_channels_alias(self):
        bc = BorderConfig.clamp()
        bc2 = bc.with_(feather_channels=[False, True])
        np.testing.assert_array_equal(bc2.feather_dims, np.array([False, True]))

    def test_with_feather_channels_alias_replaces_existing_mask(self):
        bc = BorderConfig(feather_dims=[True, False])
        bc2 = bc.with_(feather_channels=[False, True])
        np.testing.assert_array_equal(bc2.feather_dims, np.array([False, True]))


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

    def test_from_dict_accepts_feather_channels_alias(self):
        bc = BorderConfig.from_dict({"feather_channels": [0, 0, 0, 1]})
        np.testing.assert_array_equal(
            bc.feather_dims,
            np.array([False, False, False, True]),
        )

    def test_round_trip_uses_canonical_feather_dims(self):
        orig = BorderConfig(distance_mode="l2", feather_channels=[0, 0, 0, 1])
        d = orig.to_dict()
        assert "feather_dims" in d
        assert "feather_channels" not in d
        bc = BorderConfig.from_dict(d)
        assert bc.distance_mode == DistanceMode.EUCLIDEAN
        np.testing.assert_array_equal(bc.feather_dims, orig.feather_dims)

    def test_json_dict_is_json_serializable(self):
        bc = RemapBorderConfig(
            mode=BorderMode.CONSTANT,
            constant_value=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            feathering_width=0.025,
            feather_dims=[False, False, False, True],
            feathering_y_undershoot_multiplier=-1.0,
            feathering_y_overshoot_multiplier=-1.0,
            distance_mode=DistanceMode.EUCLIDEAN,
            preserve_non_feather_dims=True,
        )
        data = bc.to_json_dict()
        json.dumps(data)
        assert data["mode"] == "CONSTANT"
        assert data["feathering_mode"] == "BEYOND_BORDER"
        assert data["distance_mode"] == "EUCLIDEAN"
        assert data["constant_value"] == [0.0, 0.0, 0.0, 0.0]
        assert data["feather_dims"] == [False, False, False, True]

    def test_json_dict_round_trip(self):
        orig = RemapBorderConfig(
            mode=BorderMode.CONSTANT,
            constant_value=np.array([0.1, 0.2], dtype=np.float64),
            array=np.array([[1.0, 2.0]], dtype=np.float64),
            feathering_width=0.5,
            distance_mode=DistanceMode.MAX_NORM,
            preserve_non_feather_dims=True,
            feather_channels=[True, False],
        )
        bc = RemapBorderConfig.from_json_dict(orig.to_json_dict())
        assert bc.mode == orig.mode
        assert bc.feathering_mode == orig.feathering_mode
        assert bc.distance_mode == orig.distance_mode
        assert bc.preserve_non_feather_dims is True
        np.testing.assert_array_equal(bc.constant_value, orig.constant_value)
        np.testing.assert_array_equal(bc.array, orig.array)
        np.testing.assert_array_equal(bc.feather_dims, orig.feather_dims)

    def test_from_dict_none(self):
        bc = BorderConfig.from_dict(None)
        assert bc.mode == BorderMode.CLAMP

    def test_from_dict_already_borderconfig(self):
        orig = BorderConfig.constant(0.9)
        bc = BorderConfig.from_dict(orig)
        assert bc is orig
