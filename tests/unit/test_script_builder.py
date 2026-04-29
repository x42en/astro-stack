"""Unit tests for the Siril SSF script builder."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.domain.profile import ProcessingProfileConfig
from app.pipeline.adapters.siril_script_builder import (
    SirilScriptBuilder,
    _normalization_flag,
    _rejection_type,
)


@pytest.fixture()
def standard_config() -> ProcessingProfileConfig:
    """Return a standard processing profile configuration."""
    return ProcessingProfileConfig()


@pytest.fixture()
def drizzle_config() -> ProcessingProfileConfig:
    """Return a configuration with drizzle enabled."""
    return ProcessingProfileConfig(
        drizzle_enabled=True,
        drizzle_scale=2,
        drizzle_pixfrac=0.7,
    )


@pytest.fixture()
def minimal_frames() -> dict[str, list[Path]]:
    """Return a minimal frame dict with only lights."""
    return {
        "lights": [Path("/inbox/session/lights/light_001.fits")],
        "darks": [],
        "flats": [],
        "bias": [],
    }


@pytest.fixture()
def full_frames() -> dict[str, list[Path]]:
    """Return a full frame dict with all calibration types."""
    return {
        "lights": [Path("/inbox/session/lights/light_001.fits")],
        "darks": [Path("/inbox/session/darks/dark_001.fits")],
        "flats": [Path("/inbox/session/flats/flat_001.fits")],
        "bias": [Path("/inbox/session/bias/bias_001.fits")],
    }


class TestSirilScriptBuilder:
    """Tests for SirilScriptBuilder command generation."""

    def test_requires_command_present(
        self,
        standard_config: ProcessingProfileConfig,
        minimal_frames: dict,
    ) -> None:
        """The first command must be a 'requires' version check."""
        builder = SirilScriptBuilder(standard_config, minimal_frames, Path("/work"))
        commands = builder.build_preprocessing_commands()
        assert any(cmd.startswith("requires") for cmd in commands)

    def test_no_bias_commands_when_no_bias(
        self,
        standard_config: ProcessingProfileConfig,
        minimal_frames: dict,
    ) -> None:
        """No master-bias command when bias frames are not provided."""
        builder = SirilScriptBuilder(standard_config, minimal_frames, Path("/work"))
        commands = builder.build_preprocessing_commands()
        assert not any("master-bias" in cmd for cmd in commands)

    def test_bias_commands_present_with_bias_frames(
        self,
        standard_config: ProcessingProfileConfig,
        full_frames: dict,
    ) -> None:
        """master-bias stack command must appear when bias frames are present."""
        builder = SirilScriptBuilder(standard_config, full_frames, Path("/work"))
        commands = builder.build_preprocessing_commands()
        assert any("master-bias" in cmd for cmd in commands)

    def test_dark_commands_include_bias_correction(
        self,
        standard_config: ProcessingProfileConfig,
        full_frames: dict,
    ) -> None:
        """Dark stacking must include -bias flag when bias frames exist."""
        builder = SirilScriptBuilder(standard_config, full_frames, Path("/work"))
        commands = builder.build_preprocessing_commands()
        dark_cmds = [c for c in commands if "master-dark" in c and "stack" in c]
        assert len(dark_cmds) == 1
        assert "-bias=" in dark_cmds[0]

    def test_stack_command_contains_rejection(
        self,
        standard_config: ProcessingProfileConfig,
        minimal_frames: dict,
    ) -> None:
        """Stack command must include rejection parameters."""
        builder = SirilScriptBuilder(standard_config, minimal_frames, Path("/work"))
        commands = builder.build_preprocessing_commands()
        stack_cmds = [c for c in commands if c.startswith("stack r_pp_light")]
        assert len(stack_cmds) == 1
        assert "rej" in stack_cmds[0]

    def test_drizzle_flag_not_added_even_when_enabled(
        self,
        drizzle_config: ProcessingProfileConfig,
        minimal_frames: dict,
    ) -> None:
        """Drizzle is incompatible with -cfa -debayer and must never appear in the stack command."""
        builder = SirilScriptBuilder(drizzle_config, minimal_frames, Path("/work"))
        commands = builder.build_preprocessing_commands()
        stack_cmds = [c for c in commands if "stack" in c and "r_pp_light" in c]
        assert len(stack_cmds) == 1
        assert "-drizzle" not in stack_cmds[0]

    def test_no_drizzle_flag_when_disabled(
        self,
        standard_config: ProcessingProfileConfig,
        minimal_frames: dict,
    ) -> None:
        """Stack command must NOT include -drizzle when disabled."""
        builder = SirilScriptBuilder(standard_config, minimal_frames, Path("/work"))
        commands = builder.build_preprocessing_commands()
        stack_cmds = [c for c in commands if "r_pp_light" in c]
        assert all("-drizzle" not in c for c in stack_cmds)

    def test_seqapplyreg_follows_register(
        self,
        standard_config: ProcessingProfileConfig,
        minimal_frames: dict,
    ) -> None:
        """seqapplyreg must appear immediately after register -2pass."""
        builder = SirilScriptBuilder(standard_config, minimal_frames, Path("/work"))
        commands = builder.build_preprocessing_commands()
        reg_idx = next(i for i, c in enumerate(commands) if c.startswith("register"))
        assert commands[reg_idx + 1].startswith("seqapplyreg")

    def test_setfindstar_runs_immediately_before_register_when_overridden(
        self,
        standard_config: ProcessingProfileConfig,
        minimal_frames: dict,
    ) -> None:
        """``setfindstar`` must be tuned right before ``register -2pass``
        when ``findstar_override_enabled`` is True.

        Default behaviour (override disabled) leaves Siril's built-in
        detector alone — relaxing it smears nebular chrominance during the
        stack (verified on M42).  Wide-field DSLR rigs that need looser
        values can opt-in via the profile flag.
        """
        cfg = standard_config.model_copy(
            update={
                "findstar_override_enabled": True,
                "findstar_radius": 20,
                "findstar_sigma": 0.5,
                "findstar_roundness": 0.3,
                "findstar_relax": True,
            }
        )
        builder = SirilScriptBuilder(cfg, minimal_frames, Path("/work"))
        commands = builder.build_preprocessing_commands()
        reg_idx = next(i for i, c in enumerate(commands) if c.startswith("register"))
        # The two preceding commands must be the setfindstar tuning pair.
        assert commands[reg_idx - 2] == "setfindstar reset"
        tune = commands[reg_idx - 1]
        assert tune.startswith("setfindstar")
        # Profile-driven values are reflected verbatim in the command.
        assert "-radius=20" in tune
        assert "-sigma=0.50" in tune
        assert "-roundness=0.30" in tune
        assert "-relax=on" in tune

    def test_setfindstar_absent_by_default(
        self,
        standard_config: ProcessingProfileConfig,
        minimal_frames: dict,
    ) -> None:
        """No ``setfindstar`` command should be emitted with the default
        profile — Siril's built-in detector preserves nebular chrominance
        better than any relaxed override."""
        builder = SirilScriptBuilder(standard_config, minimal_frames, Path("/work"))
        commands = builder.build_preprocessing_commands()
        assert not any(c.startswith("setfindstar") for c in commands)

    def test_setfindstar_appears_exactly_once_when_overridden(
        self,
        standard_config: ProcessingProfileConfig,
        minimal_frames: dict,
    ) -> None:
        """Only one tuning pair should be emitted per preprocessing run."""
        cfg = standard_config.model_copy(
            update={"findstar_override_enabled": True}
        )
        builder = SirilScriptBuilder(cfg, minimal_frames, Path("/work"))
        commands = builder.build_preprocessing_commands()
        tunes = [c for c in commands if c.startswith("setfindstar -")]
        resets = [c for c in commands if c == "setfindstar reset"]
        assert len(tunes) == 1
        assert len(resets) == 1

    def test_stack_uses_correct_siril_14_rejection_syntax(
        self,
        standard_config: ProcessingProfileConfig,
        minimal_frames: dict,
    ) -> None:
        """Stack command must use 'rej <type> <low> <high>' Siril 1.4 syntax."""
        builder = SirilScriptBuilder(standard_config, minimal_frames, Path("/work"))
        commands = builder.build_preprocessing_commands()
        stack_cmd = next(c for c in commands if c.startswith("stack r_pp_light"))
        # Must NOT use old pre-1.4 tokens wrej/lrej
        assert "wrej" not in stack_cmd
        assert "lrej" not in stack_cmd
        # Must follow 'rej <type> <low> <high>' pattern
        assert "rej winsorized" in stack_cmd


class TestRejectionType:
    """Tests for _rejection_type() helper."""

    def test_sigma_maps_to_sigma(self) -> None:
        assert _rejection_type("sigma") == "sigma"

    def test_winsorized_maps_to_winsorized(self) -> None:
        assert _rejection_type("winsorized") == "winsorized"

    def test_linear_maps_to_linear(self) -> None:
        assert _rejection_type("linear") == "linear"

    def test_unknown_defaults_to_winsorized(self) -> None:
        assert _rejection_type("unknown") == "winsorized"


class TestNormalizationFlag:
    """Tests for _normalization_flag() helper."""

    def test_addscale(self) -> None:
        assert _normalization_flag("addscale") == "-norm=addscale"

    def test_mulscale(self) -> None:
        assert _normalization_flag("mulscale") == "-norm=mulscale"

    def test_none(self) -> None:
        assert _normalization_flag("none") == "-nonorm"
