"""Unit tests for :mod:`app.pipeline.adaptive.patch` (critic patch sanitisation)."""

from __future__ import annotations

from app.pipeline.adaptive.patch import sanitize_patch

_ALLOWED = (
    "stretch_method",
    "stretch_strength",
    "color_calibration_enabled",
    "camera_defiltered",
    "photometric_calibration_enabled",
)


class TestSanitizePatch:
    def test_drops_disallowed_fields(self) -> None:
        patch = sanitize_patch({"denoise_strength": 0.9}, allowed_fields=_ALLOWED)
        assert patch == {}

    def test_keeps_allowed_bool_field(self) -> None:
        patch = sanitize_patch({"color_calibration_enabled": False}, allowed_fields=_ALLOWED)
        assert patch == {"color_calibration_enabled": False}

    def test_keeps_allowed_float_in_range(self) -> None:
        patch = sanitize_patch({"stretch_strength": 120.0}, allowed_fields=_ALLOWED)
        assert patch == {"stretch_strength": 120.0}

    def test_coerces_int_like_float_value(self) -> None:
        # The critic may return e.g. 120 (int) for a float field.
        patch = sanitize_patch({"stretch_strength": 120}, allowed_fields=_ALLOWED)
        assert patch == {"stretch_strength": 120.0}
        assert isinstance(patch["stretch_strength"], float)

    def test_clamps_below_min(self) -> None:
        patch = sanitize_patch({"stretch_strength": -50.0}, allowed_fields=_ALLOWED)
        assert patch == {"stretch_strength": 0.0}

    def test_drops_value_not_in_choices(self) -> None:
        patch = sanitize_patch({"stretch_method": "not_a_real_method"}, allowed_fields=_ALLOWED)
        assert patch == {}

    def test_keeps_value_in_choices(self) -> None:
        patch = sanitize_patch({"stretch_method": "linear"}, allowed_fields=_ALLOWED)
        assert patch == {"stretch_method": "linear"}

    def test_drops_bool_disguised_as_int_field(self) -> None:
        # bool is a subclass of int in Python; must not silently coerce.
        patch = sanitize_patch({"stretch_strength": True}, allowed_fields=_ALLOWED)
        assert patch == {}

    def test_drops_wrong_type_string_for_bool_field(self) -> None:
        patch = sanitize_patch(
            {"camera_defiltered": "yes"}, allowed_fields=_ALLOWED
        )
        assert patch == {}

    def test_empty_patch_returns_empty(self) -> None:
        assert sanitize_patch({}, allowed_fields=_ALLOWED) == {}

    def test_multiple_fields_mixed_validity(self) -> None:
        patch = sanitize_patch(
            {
                "stretch_strength": 500.0,  # will clamp (no declared max -> unchanged)
                "photometric_calibration_enabled": True,
                "not_a_field": 1,
            },
            allowed_fields=_ALLOWED,
        )
        assert patch == {
            "stretch_strength": 500.0,
            "photometric_calibration_enabled": True,
        }
