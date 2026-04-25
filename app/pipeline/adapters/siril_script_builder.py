"""Dynamic Siril script (SSF) generation based on processing profile.

Builds the list of Siril commands that will be piped to ``siril-cli`` in headless
mode. The output is a sequence of string commands, not a file, so they can be
streamed one by one via the named pipe interface.

Reference for all available commands:
https://siril.readthedocs.io/en/stable/Commands.html

Example:
    >>> builder = SirilScriptBuilder(profile_config, frames, session_id)
    >>> for cmd in builder.build_preprocessing_commands():
    ...     await siril.run_command(cmd)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.domain.profile import ProcessingProfileConfig


class SirilScriptBuilder:
    """Generates Siril command sequences from a processing profile configuration.

    Adapts the command parameters (rejection algorithms, normalisation, drizzle
    settings, etc.) to the profile values, producing a deterministic and
    reproducible script.

    Attributes:
        config: The active :class:`~app.domain.profile.ProcessingProfileConfig`.
        frames: Dict of frame type → list of file paths.
        work_dir: Working directory known to Siril.
    """

    def __init__(
        self,
        config: ProcessingProfileConfig,
        frames: dict[str, list[Path]],
        work_dir: Path,
    ) -> None:
        """Initialise the builder.

        Args:
            config: Processing profile configuration (preset or advanced).
            frames: Frame inventory produced by :class:`~app.infrastructure.storage.file_store.FileStore`.
            work_dir: Siril working directory (passed via ``-d``).
        """
        self.config = config
        self.frames = frames
        self.work_dir = work_dir

    # ── Public interface ──────────────────────────────────────────────────────

    def build_preprocessing_commands(self) -> list[str]:
        """Build the full pre-processing command sequence.

        The sequence covers:
          1. Bias master (if bias frames present)
          2. Dark master
          3. Flat master
          4. Light calibration (dark, flat, debayer)
          5. Registration (star alignment)
          6. Stacking (with drizzle if configured)

        Returns:
            Ordered list of Siril command strings to execute.
        """
        commands: list[str] = [f"requires 1.2.0"]
        has_bias = len(self.frames.get("bias", [])) > 0
        has_darks = len(self.frames.get("darks", [])) > 0
        has_flats = len(self.frames.get("flats", [])) > 0

        # -- Calibration masters --
        if has_bias:
            commands.extend(self._bias_commands())
        if has_darks:
            commands.extend(self._dark_commands(with_bias=has_bias))
        if has_flats:
            commands.extend(self._flat_commands(with_bias=has_bias))

        # -- Light calibration, registration, stacking --
        commands.extend(self._light_commands(has_darks=has_darks, has_flats=has_flats))

        return commands

    def build_postprocessing_commands(self) -> list[str]:
        """Build the post-stacking Siril command sequence (stretch + colour).

        Returns:
            Ordered list of Siril command strings.
        """
        commands: list[str] = []
        commands.extend(self._stretch_commands())
        if self.config.color_calibration_enabled:
            commands.extend(self._color_commands())
        return commands

    # ── Private command builders ──────────────────────────────────────────────

    def _bias_commands(self) -> list[str]:
        """Generate commands to create the master-bias frame.

        Returns:
            List of Siril command strings for bias stacking.
        """
        rej_low = self.config.rejection_low
        rej_high = self.config.rejection_high
        return [
            "cd bias",
            "convert bias -out=../process",
            "cd ../process",
            f"stack bias rej {rej_low} {rej_high} -nonorm -out=master-bias",
            "cd ..",
        ]

    def _dark_commands(self, with_bias: bool) -> list[str]:
        """Generate commands to create the master-dark frame.

        Args:
            with_bias: Whether a master-bias has been created and should be applied.

        Returns:
            List of Siril command strings for dark stacking.
        """
        rej_low = self.config.rejection_low
        rej_high = self.config.rejection_high
        bias_flag = " -bias=../process/master-bias" if with_bias else ""
        return [
            "cd darks",
            "convert dark -out=../process",
            "cd ../process",
            f"stack dark rej {rej_low} {rej_high} -nonorm{bias_flag} -out=master-dark",
            "cd ..",
        ]

    def _flat_commands(self, with_bias: bool) -> list[str]:
        """Generate commands to create the master-flat frame.

        Args:
            with_bias: Whether a master-bias has been created.

        Returns:
            List of Siril command strings for flat stacking.
        """
        rej_low = self.config.rejection_low
        rej_high = self.config.rejection_high
        bias_flag = " -bias=../process/master-bias" if with_bias else ""
        return [
            "cd flats",
            "convert flat -out=../process",
            "cd ../process",
            f"stack flat rej {rej_low} {rej_high} -norm=mul{bias_flag} -out=master-flat",
            "cd ..",
        ]

    def _light_commands(self, has_darks: bool, has_flats: bool) -> list[str]:
        """Generate light calibration, registration and stacking commands.

        Args:
            has_darks: Whether a master-dark is available.
            has_flats: Whether a master-flat is available.

        Returns:
            List of Siril command strings for light processing.
        """
        commands: list[str] = [
            "cd lights",
            "convert light -out=../process",
            "cd ../process",
        ]

        # Calibration
        cal_flags = self._calibration_flags(has_darks, has_flats)
        commands.append(f"calibrate light {cal_flags}")

        # Registration
        commands.append("register pp_light -2pass")

        # Stacking
        commands.append(self._stack_command())

        commands.append("cd ..")
        return commands

    def _calibration_flags(self, has_darks: bool, has_flats: bool) -> str:
        """Build the flag string for the ``calibrate`` command.

        Args:
            has_darks: Whether a master-dark exists.
            has_flats: Whether a master-flat exists.

        Returns:
            Flag string to append to the ``calibrate`` command.
        """
        flags: list[str] = []
        if has_darks:
            flags.append("-dark=master-dark")
        if has_flats:
            flags.append("-flat=master-flat")

        # Input is Bayer CFA FITS (2-D, single channel). Tell Siril to treat it
        # as CFA data and debayer to RGB after calibration. The Bayer pattern
        # is auto-detected from the BAYERPAT FITS header written by raw_conversion.
        flags.append("-cfa -debayer")

        return " ".join(flags)

    def _stack_command(self) -> str:
        """Build the stacking command based on profile settings.

        Returns:
            The complete Siril ``stack`` command string.
        """
        algo = _rejection_algo_flag(self.config.rejection_algorithm)
        rej_low = self.config.rejection_low
        rej_high = self.config.rejection_high
        norm = _normalization_flag(self.config.normalization)

        base = f"stack r_pp_light {algo} {rej_low} {rej_high} {norm} -out=stack_result"

        if self.config.drizzle_enabled:
            scale = self.config.drizzle_scale
            pixfrac = self.config.drizzle_pixfrac
            base += f" -drizzle -scale={scale} -pixfrac={pixfrac}"

        return base

    def _stretch_commands(self) -> list[str]:
        """Generate stretch commands applied to the stacked image.

        Returns:
            List of Siril stretch command strings.
        """
        method = self.config.stretch_method
        if method == "asinh":
            strength = self.config.stretch_strength
            return [f"asinh -stretch {strength} -human"]
        if method == "auto":
            return ["autostretch"]
        # linear — no-op, keep linear data
        return []

    def _color_commands(self) -> list[str]:
        """Generate colour calibration commands.

        Returns:
            List of Siril colour calibration command strings.
        """
        return [
            "pcc",  # Photometric Color Calibration using GAIA
            "scs",  # Synced Colour Stretch (keeps stars neutral)
        ]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _rejection_algo_flag(algorithm: str) -> str:
    """Map a profile rejection algorithm name to a Siril CLI token.

    Args:
        algorithm: One of ``sigma``, ``winsorized``, ``linear``, ``none``.

    Returns:
        Siril-compatible rejection algorithm token.
    """
    mapping: dict[str, str] = {
        "sigma": "rej",
        "winsorized": "wrej",
        "linear": "lrej",
        "none": "rej",
    }
    return mapping.get(algorithm.lower(), "rej")


def _normalization_flag(normalization: str) -> str:
    """Map a profile normalization name to a Siril CLI flag.

    Args:
        normalization: One of ``addscale``, ``mulscale``, ``none``.

    Returns:
        Siril ``-norm=`` flag string.
    """
    mapping: dict[str, str] = {
        "addscale": "-norm=addscale",
        "mulscale": "-norm=mulscale",
        "none": "-nonorm",
    }
    return mapping.get(normalization.lower(), "-norm=addscale")
