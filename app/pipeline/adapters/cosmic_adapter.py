"""Cosmic Clarity (SetiAstro) adapter for AI denoise, sharpen and super-res.

Wraps the Cosmic Clarity Python scripts (MIT licence, github.com/setiastro/cosmicclarity)
as async subprocess calls. The scripts accept FITS or TIFF input and produce
FITS output. All models run on CUDA.

Script layout (under ``/opt/cosmic-clarity/``)::

    setiastrocosmicclarity_denoise.py
    SetiAstroCosmicClarity.py              ← sharpen
    SetiAstroCosmicClarity_SuperRes.py
    setiastrocosmicclarity_darkstar.py     ← star removal

The scripts use **fixed I/O directories** relative to their own location:

- Input:  ``{source_path}/input/``
- Output: ``{source_path}/output/``

Super-resolution is the exception: it accepts ``--input``, ``--output_dir``,
``--scale`` and ``--model_dir`` CLI flags and writes to an arbitrary directory.

Because the input/output directories are shared, a module-level asyncio lock
serialises concurrent invocations within the same worker process.

Example:
    >>> adapter = CosmicClarityAdapter()
    >>> await adapter.denoise(input_path, output_path, strength=0.8)
"""

from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path
from typing import Optional

from app.core.config import get_settings
from app.core.errors import ErrorCode, PipelineStepException
from app.core.logging import get_logger

logger = get_logger(__name__)

# Serialise all dir-based invocations within this process so concurrent tasks
# do not clobber each other's files in {source_path}/input/ and /output/.
_cc_dir_lock: asyncio.Lock | None = None


def _get_cc_lock() -> asyncio.Lock:
    global _cc_dir_lock
    if _cc_dir_lock is None:
        _cc_dir_lock = asyncio.Lock()
    return _cc_dir_lock


class CosmicClarityAdapter:
    """Adapter for the Cosmic Clarity AI image-processing suite.

    Attributes:
        source_path: Directory containing the Cosmic Clarity ``.py`` scripts
                     and model ``.pth`` files (e.g. ``/opt/cosmic-clarity``).
        gpu_device: CUDA device string (e.g. ``"cuda:0"``).  A value that does
                    not start with ``"cuda"`` disables GPU acceleration.
    """

    def __init__(
        self,
        source_path: Optional[str] = None,
        models_path: Optional[str] = None,
        gpu_device: str = "cuda:0",
    ) -> None:
        settings = get_settings()
        self.source_path = Path(source_path or settings.cosmic_clarity_source_path)
        # models_path kept for API compatibility but not forwarded to scripts
        # (scripts load models from their own directory automatically).
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
        timeout: float = 600.0,
    ) -> None:
        """Apply AI-based noise reduction to a FITS image.

        Args:
            input_path: Path to the input FITS file.
            output_path: Desired output FITS file path.
            strength: Denoise strength in the range 0.0–1.0.
            luminance_only: Process luminance channel only (faster).
            timeout: Maximum execution time in seconds.

        Raises:
            PipelineStepException: If the script fails or times out.
        """
        script = self.source_path / "setiastrocosmicclarity_denoise.py"
        self._check_script(script, "denoise")

        mode = "luminance" if luminance_only else "full"
        cmd = [
            sys.executable, str(script),
            "--denoise_strength", str(strength),
            "--denoise_mode", mode,
            *self._gpu_args(),
        ]
        expected = self._output_dir / f"{input_path.stem}_denoised{input_path.suffix}"
        await self._run_dir_based(
            cmd=cmd,
            input_path=input_path,
            expected_output=expected,
            final_output=output_path,
            step_name="denoise",
            error_code=ErrorCode.PIPE_COSMIC_DENOISE_FAILED,
            timeout=timeout,
        )
        logger.info("cosmic_denoise_done", output=str(output_path))

    async def sharpen(
        self,
        input_path: Path,
        output_path: Path,
        sharpening_mode: str = "Both",
        stellar_amount: float = 0.5,
        nonstellar_amount: float = 0.7,
        nonstellar_strength: float = 3.0,
        timeout: float = 600.0,
    ) -> None:
        """Apply AI-based sharpening to a FITS image.

        Args:
            input_path: Path to the input FITS file.
            output_path: Desired output FITS file path.
            sharpening_mode: ``"Stellar Only"``, ``"Non-Stellar Only"`` or ``"Both"``.
            stellar_amount: Sharpening amount for point sources (0.0–1.0).
            nonstellar_amount: Sharpening amount for extended objects (0.0–1.0).
            nonstellar_strength: PSF radius hint for non-stellar model (1.0–8.0).
                Must be supplied to avoid the script falling back to an interactive
                GUI dialog.  Maps to ``--nonstellar_strength`` on the CLI.
            timeout: Maximum execution time in seconds.

        Raises:
            PipelineStepException: If the script fails or times out.
        """
        script = self.source_path / "SetiAstroCosmicClarity.py"
        self._check_script(script, "sharpen")

        cmd = [
            sys.executable, str(script),
            "--sharpening_mode", sharpening_mode,
            "--stellar_amount", str(stellar_amount),
            "--nonstellar_amount", str(nonstellar_amount),
            "--nonstellar_strength", str(nonstellar_strength),
            *self._gpu_args(),
        ]
        expected = self._output_dir / f"{input_path.stem}_sharpened{input_path.suffix}"
        await self._run_dir_based(
            cmd=cmd,
            input_path=input_path,
            expected_output=expected,
            final_output=output_path,
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

        Super-resolution accepts explicit ``--input`` / ``--output_dir`` /
        ``--scale`` / ``--model_dir`` flags (unlike the other scripts).  The
        output is always saved as ``{stem}_upscaled{scale}x.fit``.

        Args:
            input_path: Path to the input FITS file.
            output_path: Desired output FITS file path.
            scale: Upscaling factor (2, 3 or 4).
            timeout: Maximum execution time in seconds.

        Raises:
            PipelineStepException: If the script fails or times out.
        """
        script = self.source_path / "SetiAstroCosmicClarity_SuperRes.py"
        self._check_script(script, "super_resolution")

        out_dir = output_path.parent
        # Script always writes {stem}_upscaled{scale}x.fit into output_dir
        script_output = out_dir / f"{input_path.stem}_upscaled{scale}x.fit"

        cmd = [
            sys.executable, str(script),
            "--input", str(input_path),
            "--output_dir", str(out_dir),
            "--scale", str(scale),
            "--model_dir", str(self.source_path),
            # super-res has no --disable_gpu; GPU is auto-detected from CUDA availability
        ]
        logger.debug("cosmic_running", step="super_resolution", cmd=" ".join(cmd[:4]))
        await self._run(cmd, "super_resolution", ErrorCode.PIPE_COSMIC_SUPERRES_FAILED, timeout)

        if not script_output.exists():
            raise PipelineStepException(
                ErrorCode.PIPE_COSMIC_SUPERRES_FAILED,
                f"Super-resolution output not found: {script_output}",
                step_name="super_resolution",
                retryable=False,
            )
        if script_output != output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(script_output), output_path)

        logger.info("cosmic_super_res_done", output=str(output_path))

    async def remove_stars(
        self,
        input_path: Path,
        output_path: Path,
        star_removal_mode: str = "unscreen",
        timeout: float = 600.0,
    ) -> None:
        """Remove stars from an image, isolating the nebula component.

        Args:
            input_path: Path to the input FITS file.
            output_path: Output path for the star-removed image.
            star_removal_mode: ``"unscreen"`` (default) or ``"additive"``.
            timeout: Maximum execution time in seconds.

        Raises:
            PipelineStepException: If the script fails or times out.
        """
        script = self.source_path / "setiastrocosmicclarity_darkstar.py"
        self._check_script(script, "star_removal")

        cmd = [
            sys.executable, str(script),
            "--star_removal_mode", star_removal_mode,
            *self._gpu_args(),
        ]
        expected = self._output_dir / f"{input_path.stem}_starless{input_path.suffix}"
        await self._run_dir_based(
            cmd=cmd,
            input_path=input_path,
            expected_output=expected,
            final_output=output_path,
            step_name="star_removal",
            error_code=ErrorCode.PIPE_STAR_SEPARATION_FAILED,
            timeout=timeout,
        )
        logger.info("cosmic_star_removal_done", output=str(output_path))

    # ── Private helpers ───────────────────────────────────────────────────────

    @property
    def _input_dir(self) -> Path:
        return self.source_path / "input"

    @property
    def _output_dir(self) -> Path:
        return self.source_path / "output"

    def _gpu_args(self) -> list[str]:
        """Return ``["--disable_gpu"]`` when not using CUDA, else ``[]``."""
        return [] if self._use_gpu else ["--disable_gpu"]

    async def _run_dir_based(
        self,
        cmd: list[str],
        input_path: Path,
        expected_output: Path,
        final_output: Path,
        step_name: str,
        error_code: ErrorCode,
        timeout: float,
    ) -> None:
        """Orchestrate a dir-based Cosmic Clarity script run.

        Acquires the module-level lock, sets up ``input/`` / ``output/`` dirs,
        runs the script, then moves the result to *final_output*.
        """
        async with _get_cc_lock():
            self._input_dir.mkdir(parents=True, exist_ok=True)
            self._output_dir.mkdir(parents=True, exist_ok=True)

            # Remove any stale files from a previous run
            for f in self._input_dir.iterdir():
                f.unlink()
            for f in self._output_dir.iterdir():
                f.unlink()

            staged = self._input_dir / input_path.name
            shutil.copy2(input_path, staged)
            try:
                logger.debug("cosmic_running", step=step_name, cmd=" ".join(cmd[:4]))
                await self._run(cmd, step_name, error_code, timeout)
            finally:
                staged.unlink(missing_ok=True)

            # Locate and move the output file
            if not expected_output.exists():
                outputs = list(self._output_dir.iterdir())
                if not outputs:
                    raise PipelineStepException(
                        error_code,
                        f"Cosmic Clarity {step_name} produced no output in {self._output_dir}",
                        step_name=step_name,
                        retryable=False,
                    )
                expected_output = outputs[0]

            final_output.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(expected_output), final_output)

    def _check_script(self, script: Path, step_name: str) -> None:
        """Verify that a Cosmic Clarity script file exists.

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
