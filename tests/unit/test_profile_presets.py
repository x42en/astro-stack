"""Unit tests for built-in processing profile presets."""

from __future__ import annotations

import pytest

from app.domain.job import ProfilePreset
from app.domain.profile import (
    ProcessingProfileConfig,
    PRESET_QUALITY,
    PRESET_QUICK,
    PRESET_STANDARD,
    get_preset_config,
)


class TestPresetConfigs:
    """Tests for built-in preset configurations."""

    def test_quick_preset_disables_plate_solving(self) -> None:
        """Quick preset must disable plate solving for speed."""
        assert PRESET_QUICK.plate_solving_enabled is False

    def test_quick_preset_disables_gradient_removal(self) -> None:
        """Quick preset must disable GraXpert gradient removal."""
        assert PRESET_QUICK.gradient_removal_enabled is False

    def test_standard_preset_enables_plate_solving(self) -> None:
        """Standard preset must enable plate solving."""
        assert PRESET_STANDARD.plate_solving_enabled is True

    def test_standard_preset_enables_denoise(self) -> None:
        """Standard preset must enable AI denoising."""
        assert PRESET_STANDARD.denoise_enabled is True

    def test_quality_preset_enables_drizzle(self) -> None:
        """Quality preset must enable drizzle integration."""
        assert PRESET_QUALITY.drizzle_enabled is True

    def test_quality_preset_enables_super_resolution(self) -> None:
        """QUALITY enables super-resolution; the object-type catalogue auto-skips
        it on bright nebulae (M42-class) where it amplifies clipped cores."""
        assert PRESET_QUALITY.super_resolution_enabled is True

    def test_quality_preset_enables_star_separation(self) -> None:
        """QUALITY enables star separation; the object-type catalogue auto-skips
        it on galaxies / clusters where it destroys the subject."""
        assert PRESET_QUALITY.star_separation_enabled is True

    def test_all_presets_default_to_defiltered_camera(self) -> None:
        """Defiltered cameras are the modern astrophoto norm \u2014 sensible default."""
        assert PRESET_QUICK.camera_defiltered is True
        assert PRESET_STANDARD.camera_defiltered is True
        assert PRESET_QUALITY.camera_defiltered is True

    def test_standard_preset_uses_luminance_only_denoise(self) -> None:
        """Luminance-only denoise preserves chrominance (H\u03b1)."""
        assert PRESET_STANDARD.denoise_luminance_only is True
        assert PRESET_QUALITY.denoise_luminance_only is True

    def test_denoise_strength_increases_from_quick_to_quality(self) -> None:
        """Denoise strength should increase from quick → quality."""
        assert PRESET_QUICK.denoise_strength < PRESET_STANDARD.denoise_strength
        assert PRESET_STANDARD.denoise_strength <= PRESET_QUALITY.denoise_strength

    def test_get_preset_config_returns_quick(self) -> None:
        """get_preset_config() returns the correct config for QUICK."""
        config = get_preset_config(ProfilePreset.QUICK)
        assert config.plate_solving_enabled is False

    def test_get_preset_config_raises_for_advanced(self) -> None:
        """get_preset_config() raises ValueError for ADVANCED."""
        with pytest.raises(ValueError, match="ADVANCED"):
            get_preset_config(ProfilePreset.ADVANCED)

    def test_all_presets_default_gradient_removal_ai_model_to_auto(self) -> None:
        """All built-in presets keep ``gradient_removal_ai_model="auto"`` so
        the orchestrator's catalogue resolver picks the right GraXpert
        model per object type at job start."""
        assert PRESET_QUICK.gradient_removal_ai_model == "auto"
        assert PRESET_STANDARD.gradient_removal_ai_model == "auto"
        assert PRESET_QUALITY.gradient_removal_ai_model == "auto"

    def test_all_presets_default_super_resolution_mode_to_auto(self) -> None:
        """Tri-state ``super_resolution_mode`` defaults to ``"auto"`` so the
        catalogue auto-skip on bright nebulae is honoured."""
        assert PRESET_QUICK.super_resolution_mode == "auto"
        assert PRESET_STANDARD.super_resolution_mode == "auto"
        assert PRESET_QUALITY.super_resolution_mode == "auto"

    def test_all_presets_default_star_separation_mode_to_auto(self) -> None:
        """Tri-state ``star_separation_mode`` defaults to ``"auto"`` so the
        catalogue auto-skip on galaxies / clusters is honoured."""
        assert PRESET_QUICK.star_separation_mode == "auto"
        assert PRESET_STANDARD.star_separation_mode == "auto"
        assert PRESET_QUALITY.star_separation_mode == "auto"


class TestProcessingProfileConfig:
    """Tests for ProcessingProfileConfig validation."""

    def test_default_config_is_standard_like(self) -> None:
        """Default config should have sensible defaults matching standard preset."""
        config = ProcessingProfileConfig()
        assert config.rejection_algorithm == "winsorized"
        assert config.denoise_enabled is True
        assert config.sharpen_enabled is True

    def test_config_serialises_to_dict(self) -> None:
        """Config must serialise to a plain dict for JSONB storage."""
        config = ProcessingProfileConfig(drizzle_enabled=True)
        d = config.model_dump()
        assert isinstance(d, dict)
        assert d["drizzle_enabled"] is True

    def test_config_round_trips_via_dict(self) -> None:
        """Config can be reconstructed from its dict representation."""
        original = ProcessingProfileConfig(denoise_strength=0.9, sharpen_radius=4)
        restored = ProcessingProfileConfig(**original.model_dump())
        assert restored.denoise_strength == pytest.approx(0.9)
        assert restored.sharpen_radius == 4
