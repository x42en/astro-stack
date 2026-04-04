"""Cosmic Clarity (SetiAstro) adapter for AI denoise, sharpen and super-res.

Wraps the Cosmic Clarity Python scripts (MIT licence, github.com/setiastro/cosmicclarity)
as async subprocess calls. The scripts accept FITS or TIFF input and produce
FITS output. All models run on CUDA.

Script layout (under ``/opt/cosmic-clarity/``)::

    setiastrocosmicclarity_denoise.py
    SetiAstroCosmicClarity.py              ← sharpen
    SetiAstroCosmicClarity_SuperRes.py
    setiastrocosmicclarity_darkstar.py     ← star removal

Example:
    >>> adapter = CosmicClarityAdapter()
    >>> await adapter.denoise(input_path, output_path, strength=0.8)
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Optional

from app.core.config import get_settings
from app.core.errors import ErrorCode, PipelineStepException
from app.core.logging import get_logger

logger = get_logger(__name__)


class CosmicClarityAdapter:
    """Adapter for the Cosmic Clarity AI image-processing suite.

    All methods run the corresponding Cosmic Clarity Python script as an
    async subprocess with CUDA enabled. The ``models_path`` is passed as an
    argument so Cosmic Clarity can locate its ``.pth`` weight files.

    Attributes:
        source_path: Directory containing the Cosmic Clarity ``.py`` scripts.
        models_path: Directory containing ``.pth`` model weight files.
        gpu_device: CUDA device index string (e.g. ``"cuda:0"``).
    """

    def __init__(
        self,
        source_path: Optional[str] = None,
        models_path: Optional[str] = None,
        gpu_device: str = "cuda:0",
    ) -> None:
        """Initialise the Cosmic Clarity adapter.

        Args:
            source_path: Optional script directory; defaults to settings.
            models_path: Optional models directory; defaults to settings.
            gpu_device: CUDA device string.
        """
        settings = get_settings()
        self.source_path = Path(source_path or settings.cosmic_clarity_source_path)
        self.models_path = Path(models_path or settings.models_path)
        self.gpu_device = gpu_device
        # Extract integer device index from "cuda:0" → "0"
        self._device_index = gpu_device.split(":")[-1] if ":" in gpu_device else "0"

    async def denoise(
        self,
        input_path: Path,
        output_path: Path,
        strength: float = 0.8,
        luminance_only: bool = False,
        timeout: float = 600.0,
    ) -> None:
        """Apply AI-based noise reduction to a FITS image.

        Args:
            input_path: Path to the input FITS file.
            output_path: Desired output FITS file path.
            strength: Denoise strength in the range 0.0–1.0.
            luminance_only: Apply noise reduction to luminance channel only.
            timeout: Maximum execution time in seconds.

        Raises:
            PipelineStepException: If the script fails or times out.
        """
        script = self.source_path / "setiastrocosmicclarity_denoise.py"
        self._check_script(script, "denoise")

        cmd = [
            sys.executable,
            str(script),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--denoise_strength",
            str(strength),
            "--models_path",
            str(self.models_path),
            "--gpu",
            self._device_index,
        ]
        if luminance_only:
            cmd.append("--luminance_only")

        await self._run(
            cmd,
            step_name="denoise",
            error_code=ErrorCode.PIPE_COSMIC_DENOISE_FAILED,
            timeout=timeout,
        )
        logger.info("cosmic_denoise_done", output=str(output_path))

    async def sharpen(
        self,
        input_path: Path,
        output_path: Path,
        stellar_amount: float = 0.5,
        nonstellar_amount: float = 0.7,
        radius: int = 2,
        timeout: float = 600.0,
    ) -> None:
        """Apply AI-based deconvolution/sharpening to a FITS image.

        Cosmic Clarity Sharpen applies separate PSF models for stellar and
        non-stellar (nebula/galaxy) components.

        Args:
            input_path: Path to the input FITS file.
            output_path: Desired output FITS file path.
            stellar_amount: Sharpening amount for stars (0.0–1.0).
            nonstellar_amount: Sharpening amount for extended objects (0.0–1.0).
            radius: PSF radius hint (1, 2, 4, or 8).
            timeout: Maximum execution time in seconds.

        Raises:
            PipelineStepException: If the script fails or times out.
        """
        script = self.source_path / "SetiAstroCosmicClarity.py"
        self._check_script(script, "sharpen")

        cmd = [
            sys.executable,
            str(script),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--stellar_amount",
            str(stellar_amount),
            "--nonstellar_amount",
            str(nonstellar_amount),
            "--radius",
            str(radius),
            "--models_path",
            str(self.models_path),
            "--gpu",
            self._device_index,
        ]

        await self._run(
            cmd,
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
        """Apply AI 2×super-resolution upscaling to a FITS image.

        Args:
            input_path: Path to the input FITS file.
            output_path: Desired output FITS file path.
            scale: Upscaling factor (typically 2).
            timeout: Maximum execution time in seconds.

        Raises:
            PipelineStepException: If the script fails or times out.
        """
        script = self.source_path / "SetiAstroCosmicClarity_SuperRes.py"
        self._check_script(script, "super_resolution")

        cmd = [
            sys.executable,
            str(script),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--scale",
            str(scale),
            "--models_path",
            str(self.models_path),
            "--gpu",
            self._device_index,
        ]

        await self._run(
            cmd,
            step_name="super_resolution",
            error_code=ErrorCode.PIPE_COSMIC_SUPERRES_FAILED,
            timeout=timeout,
        )
        logger.info("cosmic_super_res_done", output=str(output_path))

    async def remove_stars(
        self,
        input_path: Path,
        output_path: Path,
        timeout: float = 600.0,
    ) -> None:
        """Remove stars from an image, isolating the nebula component.

        Uses the Cosmic Clarity Dark Star model.

        Args:
            input_path: Path to the input FITS file.
            output_path: Output path for the star-removed (nebula-only) image.
            timeout: Maximum execution time in seconds.

        Raises:
            PipelineStepException: If the script fails or times out.
        """
        script = self.source_path / "setiastrocosmicclarity_darkstar.py"
        self._check_script(script, "star_removal")

        cmd = [
            sys.executable,
            str(script),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--models_path",
            str(self.models_path),
            "--gpu",
            self._device_index,
        ]

        await self._run(
            cmd,
            step_name="star_separation",
            error_code=ErrorCode.PIPE_STAR_SEPARATION_FAILED,
            timeout=timeout,
        )
        logger.info("cosmic_star_removal_done", output=str(output_path))

    # ── Private helpers ───────────────────────────────────────────────────────

    def _check_script(self, script: Path, step_name: str) -> None:
        """Verify that a Cosmic Clarity script file exists.

        Args:
            script: Expected script path.
            step_name: Pipeline step name for error reporting.

        Raises:
            PipelineStepException: If the script file is not found.
        """
        if not script.exists():
            raise PipelineStepException(
                ErrorCode.SYS_EXTERNAL_TOOL_MISSING,
                f"Cosmic Clarity script not found: {script}",
                step_name=step_name,
                retryable=False,
                details={"script": str(script)},
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
