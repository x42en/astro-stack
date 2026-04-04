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
        """Quality preset must enable super-resolution."""
        assert PRESET_QUALITY.super_resolution_enabled is True

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


class TestProcessingProfileConfig:
    """Tests for ProcessingProfileConfig validation."""

    def test_default_config_is_standard_like(self) -> None:
        """Default config should have sensible defaults matching standard preset."""
        config = ProcessingProfileConfig()
        assert config.rejection_algorithm == "sigma"
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
