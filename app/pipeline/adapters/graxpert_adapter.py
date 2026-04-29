"""GraXpert adapter for AI-based gradient extraction and denoising.

Wraps the GraXpert CLI (github.com/Steffenhir/GraXpert) as an async subprocess.
GraXpert uses U-Net models for background extraction and (since 3.x) for
denoising astrophotography images.

Example:
    >>> adapter = GraXpertAdapter()
    >>> await adapter.remove_background(input_path, output_path)
    >>> await adapter.denoise(input_path, output_path, strength=0.5)
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path
from typing import Optional

from app.core.config import get_settings
from app.core.errors import ErrorCode, PipelineStepException
from app.core.logging import get_logger

logger = get_logger(__name__)


class GraXpertAdapter:
    """Adapter for the GraXpert background-extraction & denoise tool.

    Supports AI U-Net background extraction, polynomial fitting, and the new
    AI denoising mode (``-cmd denoising``) introduced in GraXpert 3.x.

    Attributes:
        source_path: Directory containing the GraXpert installation.
        models_path: Directory containing GraXpert AI model files.
        gpu_device: CUDA device string (e.g. ``"cuda:0"``).
    """

    def __init__(
        self,
        source_path: Optional[str] = None,
        models_path: Optional[str] = None,
        gpu_device: str = "cuda:0",
    ) -> None:
        """Initialise the GraXpert adapter.

        Args:
            source_path: Optional path to GraXpert source; defaults to settings.
            models_path: Optional models directory; defaults to settings.
            gpu_device: CUDA device string.
        """
        settings = get_settings() if (source_path is None or models_path is None) else None
        self.source_path = Path(source_path or settings.graxpert_source_path)  # type: ignore[union-attr]
        # GraXpert 3.x stores models under XDG_DATA_HOME/GraXpert/ (capital G+X)
        self.models_path = Path(models_path or settings.models_path) / "GraXpert"  # type: ignore[union-attr]
        self.gpu_device = gpu_device

    async def remove_background(
        self,
        input_path: Path,
        output_path: Path,
        method: str = "ai",
        ai_model: str = "1.0.1",
        timeout: float = 600.0,
        *,
        correction: str = "Subtraction",
        smoothing: float = 1.0,
    ) -> None:
        """Remove the sky gradient and background from a FITS image.

        Args:
            input_path: Path to the stacked FITS file.
            output_path: Desired output FITS path for the corrected image.
            method: Extraction method: ``"ai"`` (U-Net) or ``"polynomial"``.
            ai_model: GraXpert AI model **version** (must match
                ``^\\d+\\.\\d+\\.\\d+$``, e.g. ``"1.0.1"``). For backward
                compatibility, a leading ``"GraXpert-AI-"`` prefix is stripped.
            timeout: Maximum execution time in seconds.
            correction: GraXpert ``-correction`` flag value, ``"Subtraction"``
                (default) or ``"Division"``.  Division preserves per-channel
                signal/background ratios and protects faint chromatic signal.
                Invalid values fall back to ``"Subtraction"`` with a warning.
            smoothing: GraXpert ``-smoothing`` flag, ``[0.0, 1.0]``.  Lower
                values make the background model more locally detailed (less
                likely to absorb diffuse nebulosity).  Out-of-range values
                are clamped with a warning.

        Raises:
            PipelineStepException: If GraXpert cannot be found or fails.
        """
        ai_version = _normalize_ai_version(ai_model)

        # ``correction`` is a string the GraXpert CLI matches case-sensitively;
        # an unknown value would crash the run, so coerce to the default with
        # a warning rather than propagating a malformed profile to subprocess.
        correction_value = correction if correction in ("Subtraction", "Division") else "Subtraction"
        if correction_value != correction:
            logger.warning(
                "graxpert_invalid_correction",
                requested=correction,
                fallback=correction_value,
            )
        clamped_smoothing = _clamp(float(smoothing), 0.0, 1.0, name="smoothing")

        extra_flags: list[str] = []
        if method == "ai":
            extra_flags = ["-ai_version", ai_version]
        # ``-correction`` and ``-smoothing`` are valid for both AI and polynomial
        # background-extraction modes; pass them unconditionally.
        extra_flags += [
            "-correction", correction_value,
            "-smoothing", f"{clamped_smoothing:.3f}",
        ]

        output_stem = f"{input_path.stem}_GraXpertBGE"
        cmd = self._build_cmd(
            cmd_name="background-extraction",
            input_path=input_path,
            output_stem=output_stem,
            extra_flags=extra_flags,
        )

        await self._run_command(
            cmd=cmd,
            input_path=input_path,
            output_stem=output_stem,
            output_path=output_path,
            timeout=timeout,
            error_code=ErrorCode.PIPE_GRADIENT_REMOVAL_FAILED,
            step_name="gradient_removal",
            log_event="graxpert_bge",
            log_extra={
                "method": method,
                "ai_version": ai_version,
                "correction": correction_value,
                "smoothing": clamped_smoothing,
            },
        )

    async def deconvolve(
        self,
        input_path: Path,
        output_path: Path,
        *,
        target: str,
        ai_model: str,
        strength: float = 0.5,
        psfsize: float = 0.3,
        batch_size: int = 4,
        timeout: float = 900.0,
    ) -> None:
        """Run a GraXpert deconvolution pass (object-only or stellar-only).

        Wraps ``graxpert -cli -cmd deconv-obj|deconv-stellar``.  The CLI
        flags ``-strength`` (``[0.0, 1.0]``), ``-psfsize`` (``[0.0, 5.0]``)
        and ``-batch_size`` (``[1, 32]``) are clamped here so a malformed
        profile cannot abort the run.

        Args:
            input_path: Source FITS file.
            output_path: Desired output FITS path.
            target: ``"object"`` (uses ``deconv-obj`` and the
                ``deconvolution-object-ai-models`` directory) or ``"stars"``
                (uses ``deconv-stellar`` and the stellar models).
            ai_model: Deconvolution model version (e.g. ``"1.0.1"``).
            strength: Deconvolution strength in ``[0.0, 1.0]``.
            psfsize: Estimated PSF radius (GraXpert default ``0.3``).
            batch_size: Tiles processed in parallel ``[1, 32]``.
            timeout: Maximum execution time in seconds.

        Raises:
            ValueError: If ``target`` is not ``"object"`` or ``"stars"``.
            PipelineStepException: If GraXpert cannot be found, times out, or fails.
        """
        if target not in ("object", "stars"):
            raise ValueError(
                f"deconvolve target must be 'object' or 'stars', got {target!r}"
            )
        cmd_name = "deconv-obj" if target == "object" else "deconv-stellar"
        ai_version = _normalize_ai_version(ai_model)
        clamped_strength = _clamp(float(strength), 0.0, 1.0, name="strength")
        clamped_psfsize = _clamp(float(psfsize), 0.0, 5.0, name="psfsize")
        clamped_batch = int(_clamp(int(batch_size), 1, 32, name="batch_size"))

        extra_flags = [
            "-ai_version", ai_version,
            "-strength", f"{clamped_strength:.3f}",
            "-psfsize", f"{clamped_psfsize:.3f}",
            "-batch_size", str(clamped_batch),
        ]

        output_stem = f"{input_path.stem}_GraXpertDeconv{target.capitalize()}"
        cmd = self._build_cmd(
            cmd_name=cmd_name,
            input_path=input_path,
            output_stem=output_stem,
            extra_flags=extra_flags,
        )

        await self._run_command(
            cmd=cmd,
            input_path=input_path,
            output_stem=output_stem,
            output_path=output_path,
            timeout=timeout,
            error_code=ErrorCode.PIPE_GRAXPERT_DECONV_FAILED,
            step_name=f"deconv_{target}",
            log_event="graxpert_deconv",
            log_extra={
                "target": target,
                "ai_version": ai_version,
                "strength": clamped_strength,
                "psfsize": clamped_psfsize,
                "batch_size": clamped_batch,
            },
        )

    async def denoise(
        self,
        input_path: Path,
        output_path: Path,
        *,
        ai_model: str = "3.0.2",
        strength: float = 0.5,
        batch_size: int = 4,
        timeout: float = 900.0,
    ) -> None:
        """Denoise a FITS image with GraXpert's AI denoise model.

        Wraps ``graxpert -cli -cmd denoising``. Strength and batch_size are
        clamped to the CLI-accepted ranges with a warning log on truncation
        so a malformed profile cannot abort the run.

        Args:
            input_path: Path to the input FITS file.
            output_path: Desired output FITS path.
            ai_model: GraXpert denoise model version (semver; e.g. ``"3.0.2"``).
                Prefixed legacy values are tolerated via :func:`_normalize_ai_version`.
            strength: Denoise strength in ``[0.0, 1.0]``. Out-of-range values
                are clamped (warning logged).
            batch_size: Number of tiles processed in parallel ``[1, 32]``.
                Higher values are faster but may cause GPU OOM. Clamped if
                outside the range.
            timeout: Maximum execution time in seconds (denoise is heavier
                than BGE so the default is larger).

        Raises:
            PipelineStepException: If GraXpert cannot be found, times out, or fails.
        """
        ai_version = _normalize_ai_version(ai_model)
        clamped_strength = _clamp(float(strength), 0.0, 1.0, name="strength")
        clamped_batch = int(_clamp(int(batch_size), 1, 32, name="batch_size"))

        extra_flags = [
            "-ai_version", ai_version,
            "-strength", f"{clamped_strength:.3f}",
            "-batch_size", str(clamped_batch),
        ]

        output_stem = f"{input_path.stem}_GraXpertDenoise"
        cmd = self._build_cmd(
            cmd_name="denoising",
            input_path=input_path,
            output_stem=output_stem,
            extra_flags=extra_flags,
        )

        await self._run_command(
            cmd=cmd,
            input_path=input_path,
            output_stem=output_stem,
            output_path=output_path,
            timeout=timeout,
            error_code=ErrorCode.PIPE_GRAXPERT_DENOISE_FAILED,
            step_name="denoise",
            log_event="graxpert_denoise",
            log_extra={
                "ai_version": ai_version,
                "strength": clamped_strength,
                "batch_size": clamped_batch,
            },
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _gpu_flag(self) -> str:
        """Return ``'true'`` when a CUDA device is configured, ``'false'`` otherwise.

        GraXpert's ``-gpu`` flag accepts the string literals ``true`` / ``false``
        (not a device index). We enable GPU whenever ``gpu_device`` starts with
        ``'cuda'``; the runtime will use the first visible CUDA device as
        determined by ``CUDA_VISIBLE_DEVICES`` or driver defaults.
        """
        return "true" if self.gpu_device.startswith("cuda") else "false"

    def _build_cmd(
        self,
        *,
        cmd_name: str,
        input_path: Path,
        output_stem: str,
        extra_flags: list[str],
    ) -> list[str]:
        """Build a GraXpert CLI invocation.

        Selects between the installed ``graxpert`` entry-point and the
        ``GraXpert.py`` source script depending on what's available on disk.
        Both forms share the same flag layout.

        Args:
            cmd_name: ``-cmd`` value (e.g. ``"background-extraction"`` or
                ``"denoising"``).
            input_path: Input FITS path (positional argument).
            output_stem: ``-output`` value (basename without extension; GraXpert
                appends ``.fits``/``.xisf`` and writes next to the input).
            extra_flags: Command-specific flags inserted between the common
                prefix and the input path.

        Returns:
            Command token list ready to pass to :func:`asyncio.create_subprocess_exec`.
        """
        graxpert_main = self.source_path / "GraXpert.py"
        if graxpert_main.exists():
            prefix = [sys.executable, str(graxpert_main)]
        else:
            prefix = ["graxpert"]
        # GraXpert 3.x CLI requires -cli before -cmd to enable headless mode;
        # without it the script prints usage help and exits 2.
        return [
            *prefix,
            "-cli",
            "-cmd", cmd_name,
            "-output", output_stem,
            "-gpu", self._gpu_flag(),
            *extra_flags,
            str(input_path),
        ]

    async def _run_command(
        self,
        *,
        cmd: list[str],
        input_path: Path,
        output_stem: str,
        output_path: Path,
        timeout: float,
        error_code: ErrorCode,
        step_name: str,
        log_event: str,
        log_extra: Optional[dict] = None,
    ) -> None:
        """Run a GraXpert subprocess and move its produced file to ``output_path``.

        GraXpert's ``-output`` flag is a *basename without extension* — it always
        writes the result next to the input as
        ``<input_dir>/<output_stem>.<ext>``. We snapshot the directory before
        the run, glob for the new file afterwards, and rename it to the
        requested ``output_path``.

        Args:
            cmd: Full command list (built by :meth:`_build_cmd`).
            input_path: Input FITS path (used to locate the produced file).
            output_stem: ``-output`` basename (used in the post-run glob).
            output_path: Destination path for the produced file.
            timeout: Subprocess timeout in seconds.
            error_code: :class:`ErrorCode` raised on failure.
            step_name: Pipeline step name for the raised exception.
            log_event: Structured logger event base name.
            log_extra: Optional structured log fields.
        """
        # Snapshot existing siblings so we can detect what GraXpert created.
        pre_existing = {p for p in input_path.parent.glob(f"{output_stem}.*")}

        logger.info(
            f"{log_event}_starting",
            input=str(input_path),
            **(log_extra or {}),
        )

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError as exc:
            raise PipelineStepException(
                error_code,
                f"GraXpert timed out after {timeout}s.",
                step_name=step_name,
                retryable=True,
            ) from exc
        except FileNotFoundError as exc:
            raise PipelineStepException(
                ErrorCode.SYS_EXTERNAL_TOOL_MISSING,
                "GraXpert is not installed or not found at the configured path.",
                step_name=step_name,
                retryable=False,
            ) from exc

        if proc.returncode != 0:
            stderr_text = stderr.decode("utf-8", errors="replace")[:500]
            raise PipelineStepException(
                error_code,
                f"GraXpert failed (exit {proc.returncode}): {stderr_text}",
                step_name=step_name,
                retryable=True,
                details={"returncode": proc.returncode, "stderr": stderr_text},
            )

        # Move the file produced next to the input to the requested output path.
        produced_candidates = sorted(
            p
            for p in input_path.parent.glob(f"{output_stem}.*")
            if p.suffix.lower() in {".fits", ".fit", ".xisf"}
            and p not in pre_existing
        )
        if not produced_candidates:
            raise PipelineStepException(
                error_code,
                (
                    f"GraXpert did not produce expected output "
                    f"({output_stem}.fits/.fit/.xisf) in {input_path.parent}"
                ),
                step_name=step_name,
                retryable=False,
            )
        produced = produced_candidates[0]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()
        produced.rename(output_path)

        logger.info(f"{log_event}_done", output=str(output_path))


_AI_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


def _normalize_ai_version(ai_model: str) -> str:
    """Coerce a profile ``ai_model`` value into the bare semver GraXpert expects.

    GraXpert validates ``-ai_version`` against ``^\\d+\\.\\d+\\.\\d+$`` and
    rejects anything else with argparse exit code 2. Older AstroStack profiles
    used the form ``"GraXpert-AI-1.0.0"`` (a model filename, not a version);
    strip such prefixes for backward compatibility.

    Args:
        ai_model: Value from the profile (e.g. ``"1.0.1"`` or
            ``"GraXpert-AI-1.0.0"``).

    Returns:
        A semver string suitable for ``-ai_version``.
    """
    candidate = ai_model.strip()
    for prefix in ("GraXpert-AI-", "GraXpert-"):
        if candidate.startswith(prefix):
            candidate = candidate[len(prefix):]
            break
    for ext in (".pth", ".onnx"):
        if candidate.endswith(ext):
            candidate = candidate[: -len(ext)]
            break
    if not _AI_VERSION_RE.match(candidate):
        logger.warning(
            "graxpert_ai_version_invalid",
            provided=ai_model,
            fallback="1.0.1",
        )
        return "1.0.1"
    return candidate


def _clamp(value: float, low: float, high: float, *, name: str) -> float:
    """Clamp ``value`` into ``[low, high]``, logging a warning on truncation.

    Args:
        value: Caller-provided value.
        low: Inclusive lower bound.
        high: Inclusive upper bound.
        name: Field name used in the warning log message.

    Returns:
        The clamped value.
    """
    if value < low or value > high:
        clamped = max(low, min(high, value))
        logger.warning(
            "graxpert_param_clamped",
            field=name,
            provided=value,
            clamped=clamped,
            range=[low, high],
        )
        return clamped
    return value
