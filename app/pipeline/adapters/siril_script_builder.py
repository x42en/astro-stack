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

        The image ``for_stretch.fits`` must exist in the Siril working directory.
        It is loaded, processed in-place, then saved back and closed.

        Returns:
            Ordered list of Siril command strings.
        """
        commands: list[str] = ["load for_stretch"]
        # rmgreen must run on linear (un-stretched) data to correctly remove
        # the double-green artefact introduced by Bayer CFA debayering.
        # Applying it after a non-linear stretch degrades colour accuracy.
        # NOTE: rmgreen is applied regardless of ``camera_defiltered``: the
        # green excess is a CFA debayer artefact, not a sensor-spectrum
        # artefact.  The defiltered/stock-DSLR distinction is handled
        # downstream by the display pipeline (per-channel red BP softening
        # and red/saturation boost), not by Siril.
        if self.config.color_calibration_enabled:
            commands.append("rmgreen")
        commands.extend(self._stretch_commands())
        # NOTE: 'color_calibration' is a GUI-only feature in Siril 1.4.x and
        # is NOT available as a script command.  'rmgreen' (applied above on
        # linear data) is the only non-plate-solving colour correction that
        # can be used in headless scripts.
        commands.append("save for_stretch")
        commands.append("close")
        return commands

    def build_pcc_commands(self) -> list[str]:
        """Build a standalone Siril command sequence to apply ``pcc``.

        Photometric Colour Calibration must run on linear (un-stretched) data
        and requires the FITS to carry valid WCS headers (set by ASTAP).
        It is invoked separately from the stretch script so the orchestrator
        can tolerate failures (e.g. catalogue download error) without
        aborting the whole post-processing chain.

        Returns:
            Ordered list of Siril command strings.
        """
        # ``pcc`` defaults: catalogue=auto (APASS), limit-mag derived from FOV.
        # We keep flags minimal so Siril picks sane defaults from the WCS.
        return [
            "load for_stretch",
            "pcc",
            "save for_stretch",
            "close",
        ]

    # ── Private command builders ──────────────────────────────────────────────

    def _bias_commands(self) -> list[str]:
        """Generate commands to create the master-bias frame.

        Returns:
            List of Siril command strings for bias stacking.
        """
        rej_clause = self._rej_clause()
        return [
            "cd bias",
            "convert bias -out=../process",
            "cd ../process",
            f"stack bias {rej_clause} -nonorm -out=master-bias",
            "cd ..",
        ]

    def _dark_commands(self, with_bias: bool) -> list[str]:
        """Generate commands to create the master-dark frame.

        Args:
            with_bias: Whether a master-bias has been created and should be applied.

        Returns:
            List of Siril command strings for dark stacking.
        """
        rej_clause = self._rej_clause()
        bias_flag = " -bias=../process/master-bias" if with_bias else ""
        return [
            "cd darks",
            "convert dark -out=../process",
            "cd ../process",
            f"stack dark {rej_clause} -nonorm{bias_flag} -out=master-dark",
            "cd ..",
        ]

    def _flat_commands(self, with_bias: bool) -> list[str]:
        """Generate commands to create the master-flat frame.

        Args:
            with_bias: Whether a master-bias has been created.

        Returns:
            List of Siril command strings for flat stacking.
        """
        rej_clause = self._rej_clause()
        bias_flag = " -bias=../process/master-bias" if with_bias else ""
        return [
            "cd flats",
            "convert flat -out=../process",
            "cd ../process",
            f"stack flat {rej_clause} -norm=mul{bias_flag} -out=master-flat",
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

        # Registration — two-step approach compatible with Siril < 1.2.0.
        # Step 1: analyse star patterns and compute per-frame transforms; writes
        # only pp_light.seq metadata (no output frames yet).
        commands.append("register pp_light -2pass")
        # Step 2: apply transforms and write aligned frames to disk.
        # -framing=min crops to the intersection of all frames so there are no
        # zero-padded borders. This prevents addscale normalization from
        # collapsing: cog/max framing inflates the canvas with empty pixels,
        # driving the background estimate to ≈ 0 → all-black stack.
        # Switch to 'register pp_light -noout' once the image is rebuilt with
        # Siril 1.4.x (ppa:lock042/siril on Ubuntu 24.04).
        commands.append("seqapplyreg pp_light -framing=min -interp=lanczos4")

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

    def _rej_clause(self) -> str:
        """Build the ``rej <type> <low> <high>`` clause for stack commands.

        Returns:
            Rejection clause string (no leading/trailing spaces).
        """
        rej_type = _rejection_type(self.config.rejection_algorithm)
        if rej_type == "none":
            return "rej none"
        return f"rej {rej_type} {self.config.rejection_low} {self.config.rejection_high}"

    def _stack_command(self) -> str:
        """Build the stacking command for light frames based on profile settings.

        Drizzle is intentionally disabled: the CFA→RGB debayer during calibration
        is incompatible with drizzle, which requires the raw Bayer mosaic.

        Returns:
            The complete Siril ``stack`` command string.
        """
        rej_clause = self._rej_clause()
        norm = _normalization_flag(self.config.normalization)
        # Note: -weight=wfwhm is intentionally omitted. When using -noout
        # registration, the virtual r_pp_light sequence does not carry the
        # per-frame FWHM measurements from pp_light.seq. Siril then assigns
        # all frames weight = 0, producing an all-black stack. Equal weighting
        # is correct and safe for well-tracked astrophotography sessions.
        return f"stack r_pp_light {rej_clause} {norm} -out=stack_result"

    def _stretch_commands(self) -> list[str]:
        """Generate stretch commands applied to the loaded stacked image.

        Returns:
            List of Siril stretch command strings.
        """
        method = self.config.stretch_method
        if method == "asinh":
            # stretch_strength is the positional 'stretch' argument (1–1000).
            # -human applies a perceptually-uniform intensity mapping.
            # Siril syntax: asinh [-human] stretch [bg]
            #
            # The optional bg parameter is the background level Siril
            # subtracts before applying the asinh formula.  Siril clamps it
            # to [0, 1]; passing a negative "pedestal" is silently ignored,
            # so we omit it entirely and rely on the per-channel display
            # stretch to lift the background in previews/export.
            strength = self.config.stretch_strength
            return [f"asinh -human {strength:.1f}"]
        if method == "auto":
            # autostretch with -linked preserves colour balance across channels.
            return ["autostretch -linked"]
        # linear — no-op, keep linear data
        return []

    def _color_commands(self) -> list[str]:
        """Generate colour calibration commands.

        This method is intentionally left empty: ``rmgreen`` is applied
        before the stretch (in ``build_postprocessing_commands``) so it works
        on linear data.  ``color_calibration`` is a GUI-only command in Siril
        1.4.x and cannot be used in scripts.
        The method is retained for future use (e.g. ``pcc`` / ``spcc`` once
        plate-solving is functional).

        Returns:
            Empty list (commands are inlined in ``build_postprocessing_commands``).
        """
        return []


# ── Helpers ───────────────────────────────────────────────────────────────────


def _rejection_type(algorithm: str) -> str:
    """Map a profile rejection algorithm name to a Siril 1.4 rejection type keyword.

    In Siril 1.4, ``stack`` uses: ``rej <type> sigma_low sigma_high``.
    Valid ``<type>`` values: ``none``, ``percentile``, ``sigma``, ``median``,
    ``winsorized``, ``linear``, ``generalized``, ``mad``.

    Args:
        algorithm: One of ``sigma``, ``winsorized``, ``linear``, ``none``.

    Returns:
        Siril-compatible rejection type keyword.
    """
    mapping: dict[str, str] = {
        "sigma": "sigma",
        "winsorized": "winsorized",
        "linear": "linear",
        "none": "none",
    }
    return mapping.get(algorithm.lower(), "winsorized")


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
