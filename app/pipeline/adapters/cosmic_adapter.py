"""Cosmic Clarity engine adapter, backed by the SetiAstroSuitePro (SASpro) CLI.

setiastro/cosmicclarity (the old standalone script bundle, MIT licence) was
archived upstream in favour of github.com/setiastro/setiastrosuitepro, which
ships the same engines behind a proper headless CLI documented at
https://github.com/setiastro/setiastrosuitepro/wiki/CLI:-Command-Line-Interface
("ideal for batch processing / automation / headless workflows").

Every mode now takes a single input/output file pair — no more shared
``input/``/``output/`` staging directories, so no cross-call locking is
needed; concurrent invocations are independent subprocesses.

Entry point resolution order documented by SASpro:
    1. ``cosmicclarity`` (pip/Poetry installs — what our Docker image uses)
    2. ``setiastrosuitepro cc`` (works for frozen builds too)
We use (1) since the Dockerfile installs SASpro via pip.

Example:
    >>> adapter = CosmicClarityAdapter()
    >>> await adapter.denoise(input_path, output_path, strength=0.8)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from app.core.config import get_settings
from app.core.errors import ErrorCode, PipelineStepException
from app.core.logging import get_logger

logger = get_logger(__name__)


class CosmicClarityAdapter:
    """Adapter for the Cosmic Clarity engines exposed by SASpro's ``cc`` CLI.

    Attributes:
        cli_binary: Name (resolved via ``PATH``) or absolute path of the
                    ``cosmicclarity`` entry point installed by SASpro.
        gpu_device: CUDA device string (e.g. ``"cuda:0"``). A value that does
                    not start with ``"cuda"`` disables GPU acceleration.
    """

    def __init__(
        self,
        cli_binary: Optional[str] = None,
        models_path: Optional[str] = None,
        gpu_device: str = "cuda:0",
    ) -> None:
        settings = get_settings()
        self.cli_binary = cli_binary or settings.cosmic_clarity_cli
        # models_path kept for API compatibility; SASpro manages its own
        # bundled/downloaded model weights and is not pointed at our volume.
        self.models_path = Path(models_path or settings.models_path)
        self.gpu_device = gpu_device
        self._use_gpu = gpu_device.lower().startswith("cuda")

    # ── Public API ────────────────────────────────────────────────────────────

    async def denoise(
        self,
        input_path: Path,
        output_path: Path,
        strength: float = 0.8,
        luminance_only: bool = False,
        aberration_first: bool = False,
        timeout: float = 600.0,
    ) -> None:
        """Apply AI-based noise reduction to a FITS image.

        Args:
            input_path: Path to the input FITS file.
            output_path: Desired output FITS file path.
            strength: Denoise strength in the range 0.0-1.0. Mapped to both
                ``--denoise-luma`` and ``--denoise-color``.
            luminance_only: Process luminance channel only, preserving
                chrominance (``--denoise-mode luminance`` vs ``full``).
            aberration_first: Run ``cosmicclarity correct`` (standalone
                Aberration Remover) as a pre-pass before denoising. Unlike
                ``sharpen`` (which has an inline ``--stellar-correct-mode``),
                ``denoise`` has no built-in correction flag, so this chains
                two CLI calls through a temporary intermediate file.
            timeout: Maximum execution time in seconds.

        Raises:
            PipelineStepException: If the CLI fails or times out.
        """
        source = input_path
        corrected: Optional[Path] = None
        if aberration_first:
            corrected = output_path.parent / f"_{output_path.stem}_corrected.fits"
            await self.correct_aberration(input_path, corrected, timeout=timeout)
            source = corrected

        args = [
            "--denoise-luma", str(strength),
            "--denoise-color", str(strength),
            "--denoise-mode", "luminance" if luminance_only else "full",
        ]
        try:
            await self._run_cc(
                mode="denoise",
                input_path=source,
                output_path=output_path,
                extra_args=args,
                step_name="denoise",
                error_code=ErrorCode.PIPE_COSMIC_DENOISE_FAILED,
                timeout=timeout,
            )
        finally:
            if corrected is not None:
                corrected.unlink(missing_ok=True)
        logger.info("cosmic_denoise_done", output=str(output_path))

    async def correct_aberration(
        self,
        input_path: Path,
        output_path: Path,
        temp_stretch: bool = True,
        target_median: float = 0.25,
        timeout: float = 300.0,
    ) -> None:
        """Run SASpro's standalone Aberration Remover (``cc correct``).

        Corrects colour fringing from chromatic aberration at the optics
        level, independent of sharpening/denoising. Confirmed as a real CLI
        subcommand (v1.21.5.post1): ``sharpen``/``both`` have their own inline
        ``--stellar-correct-mode``, but ``denoise`` does not, hence this
        standalone entry point for chaining ahead of denoise.

        Args:
            input_path: Path to the input FITS file.
            output_path: Desired output FITS file path.
            temp_stretch: Temporarily stretch linear data before AI
                processing then unstretch after (recommended for linear FITS).
            target_median: Target median for the temporary stretch.
            timeout: Maximum execution time in seconds.

        Raises:
            PipelineStepException: If the CLI fails or times out.
        """
        args = ["--target-median", str(target_median)]
        args.append("--temp-stretch" if temp_stretch else "--no-temp-stretch")
        await self._run_cc(
            mode="correct",
            input_path=input_path,
            output_path=output_path,
            extra_args=args,
            step_name="aberration_correction",
            error_code=ErrorCode.PIPE_COSMIC_DENOISE_FAILED,
            timeout=timeout,
        )
        logger.info("cosmic_aberration_correction_done", output=str(output_path))

    async def sharpen(
        self,
        input_path: Path,
        output_path: Path,
        sharpening_mode: str = "Both",
        stellar_amount: float = 0.5,
        nonstellar_amount: float = 0.7,
        nonstellar_strength: float = 3.0,
        aberration_first: bool = False,
        timeout: float = 600.0,
    ) -> None:
        """Apply AI-based sharpening to a FITS image.

        Args:
            input_path: Path to the input FITS file.
            output_path: Desired output FITS file path.
            sharpening_mode: ``"Stellar Only"``, ``"Non-Stellar Only"`` or ``"Both"``.
            stellar_amount: Sharpening amount for point sources (0.0-1.0).
            nonstellar_amount: Sharpening amount for extended objects (0.0-1.0).
            nonstellar_strength: PSF radius hint for the non-stellar model,
                mapped to ``--nonstellar-psf``.
            aberration_first: Run the Aberration Remover before sharpening via
                the inline ``--stellar-correct-mode correct_sharpen`` flag
                (confirmed CLI option, v1.21.5.post1 — no separate CLI call
                needed here, unlike ``denoise``).
            timeout: Maximum execution time in seconds.

        Raises:
            PipelineStepException: If the CLI fails or times out.
        """
        args = [
            "--sharpening-mode", sharpening_mode,
            "--stellar-amount", str(stellar_amount),
            "--nonstellar-amount", str(nonstellar_amount),
            "--nonstellar-psf", str(nonstellar_strength),
        ]
        if aberration_first:
            args.extend(["--stellar-correct-mode", "correct_sharpen"])
        await self._run_cc(
            mode="sharpen",
            input_path=input_path,
            output_path=output_path,
            extra_args=args,
            step_name="sharpen",
            error_code=ErrorCode.PIPE_COSMIC_SHARPEN_FAILED,
            timeout=timeout,
        )
        logger.info("cosmic_sharpen_done", output=str(output_path))

    async def super_resolution(
        self,
        input_path: Path,
        output_path: Path,
        scale: int = 2,
        timeout: float = 900.0,
    ) -> None:
        """Apply AI super-resolution upscaling to a FITS image.

        Args:
            input_path: Path to the input FITS file.
            output_path: Desired output FITS file path.
            scale: Upscaling factor (2, 3 or 4).
            timeout: Maximum execution time in seconds.

        Raises:
            PipelineStepException: If the CLI fails or times out.
        """
        await self._run_cc(
            mode="superres",
            input_path=input_path,
            output_path=output_path,
            extra_args=["--scale", str(scale)],
            step_name="super_resolution",
            error_code=ErrorCode.PIPE_COSMIC_SUPERRES_FAILED,
            timeout=timeout,
        )
        logger.info("cosmic_super_res_done", output=str(output_path))

    async def remove_satellite_trails(
        self,
        input_path: Path,
        output_path: Path,
        mode: str = "full",
        sensitivity: float = 0.10,
        clip_trail: bool = True,
        timeout: float = 600.0,
    ) -> None:
        """Detect and remove satellite/aircraft trails from a FITS image.

        New capability exposed by SASpro's ``cc satellite`` mode; absent from
        the old standalone Cosmic Clarity scripts. Useful outside curated test
        data where trails are common and previously went unhandled.

        Args:
            input_path: Path to the input FITS file.
            output_path: Desired output FITS file path.
            mode: ``"full"`` or ``"luminance"``.
            sensitivity: Detection sensitivity threshold (lower = more aggressive).
            clip_trail: Whether to hard-clip detected trail pixels
                (``--clip-trail``/``--no-clip-trail``).
            timeout: Maximum execution time in seconds.

        Raises:
            PipelineStepException: If the CLI fails or times out.
        """
        args = [
            "--mode", mode,
            "--sensitivity", str(sensitivity),
            "--clip-trail" if clip_trail else "--no-clip-trail",
        ]
        await self._run_cc(
            mode="satellite",
            input_path=input_path,
            output_path=output_path,
            extra_args=args,
            step_name="satellite_removal",
            error_code=ErrorCode.PIPE_COSMIC_SATELLITE_FAILED,
            timeout=timeout,
        )
        logger.info("cosmic_satellite_removal_done", output=str(output_path))

    async def remove_stars(
        self,
        input_path: Path,
        output_path: Path,
        star_removal_mode: str = "unscreen",
        timeout: float = 600.0,
    ) -> None:
        """Remove stars from an image, isolating the nebula component.

        NOTE — confirmed on real install (v1.21.5.post1): ``darkstar`` IS a
        real ``cc`` subcommand (``{sharpen,correct,denoise,both,superres,
        satellite,darkstar}``), verified via ``cosmicclarity darkstar --help``
        on the target GPU server — this was previously an open question
        (the public wiki page didn't list it) and is now resolved.

        Args:
            input_path: Path to the input FITS file.
            output_path: Output path for the star-removed image.
            star_removal_mode: ``"unscreen"`` (default) or ``"additive"``.
            timeout: Maximum execution time in seconds.

        Raises:
            PipelineStepException: If the CLI fails, times out, or does not
                support this mode.
        """
        await self._run_cc(
            mode="darkstar",
            input_path=input_path,
            output_path=output_path,
            extra_args=["--star-removal-mode", star_removal_mode],
            step_name="star_removal",
            error_code=ErrorCode.PIPE_STAR_SEPARATION_FAILED,
            timeout=timeout,
        )
        logger.info("cosmic_star_removal_done", output=str(output_path))

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _run_cc(
        self,
        mode: str,
        input_path: Path,
        output_path: Path,
        extra_args: list[str],
        step_name: str,
        error_code: ErrorCode,
        timeout: float,
    ) -> None:
        """Run one ``cosmicclarity <mode> -i ... -o ...`` invocation.

        Raises:
            PipelineStepException: On failure, timeout, or missing binary.
        """
        import asyncio  # noqa: PLC0415

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.cli_binary, mode,
            "-i", str(input_path),
            "-o", str(output_path),
            "--gpu" if self._use_gpu else "--no-gpu",
            *extra_args,
        ]
        logger.debug("cosmic_running", step=step_name, cmd=" ".join(cmd[:3]))
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except FileNotFoundError as exc:
            raise PipelineStepException(
                ErrorCode.SYS_EXTERNAL_TOOL_MISSING,
                f"cosmicclarity CLI not found on PATH ({self.cli_binary!r}) — "
                "is setiastrosuitepro installed?",
                step_name=step_name,
                retryable=False,
            ) from exc
        except asyncio.TimeoutError as exc:
            raise PipelineStepException(
                error_code,
                f"Cosmic Clarity {step_name} timed out after {timeout}s.",
                step_name=step_name,
                retryable=True,
            ) from exc

        if proc.returncode != 0:
            stderr_text = stderr.decode("utf-8", errors="replace")[:500]
            raise PipelineStepException(
                error_code,
                f"Cosmic Clarity {step_name} failed (exit {proc.returncode}): {stderr_text}",
                step_name=step_name,
                retryable=True,
                details={"returncode": proc.returncode, "stderr": stderr_text},
            )
        if not output_path.exists():
            raise PipelineStepException(
                error_code,
                f"Cosmic Clarity {step_name} produced no output at {output_path}",
                step_name=step_name,
                retryable=False,
            )

    async def _run(
        self,
        cmd: list[str],
        step_name: str,
        error_code: ErrorCode,
        timeout: float,
    ) -> None:
        """Execute a command and raise on non-zero exit code.

        Args:
            cmd: Command tokens for subprocess execution.
            step_name: Pipeline step name used in error messages.
            error_code: Error code to embed in the raised exception.
            timeout: Maximum execution time in seconds.

        Raises:
            PipelineStepException: On failure, timeout, or missing binary.
        """
        logger.debug("cosmic_running", step=step_name, cmd=" ".join(cmd[:4]))
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise PipelineStepException(
                error_code,
                f"Cosmic Clarity {step_name} timed out after {timeout}s.",
                step_name=step_name,
                retryable=True,
            ) from exc

        if proc.returncode != 0:
            stderr_text = stderr.decode("utf-8", errors="replace")[:500]
            raise PipelineStepException(
                error_code,
                f"Cosmic Clarity {step_name} failed (exit {proc.returncode}): {stderr_text}",
                step_name=step_name,
                retryable=True,
                details={"returncode": proc.returncode, "stderr": stderr_text},
            )
