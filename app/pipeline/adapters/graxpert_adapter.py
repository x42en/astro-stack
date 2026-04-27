"""GraXpert adapter for AI-based gradient and background extraction.

Wraps GraXpert CLI (github.com/Steffenhir/GraXpert) as an async subprocess.
GraXpert uses a U-Net model to extract and subtract sky-background gradients
from astrophotography images.

Example:
    >>> adapter = GraXpertAdapter()
    >>> await adapter.remove_background(input_path, output_path)
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
    """Adapter for the GraXpert background gradient removal tool.

    Supports both AI model-based extraction and polynomial background
    fitting, depending on the profile configuration.

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
        settings = get_settings()
        self.source_path = Path(source_path or settings.graxpert_source_path)
        # GraXpert 3.x stores models under XDG_DATA_HOME/GraXpert/ (capital G+X)
        self.models_path = Path(models_path or settings.models_path) / "GraXpert"
        self.gpu_device = gpu_device

    async def remove_background(
        self,
        input_path: Path,
        output_path: Path,
        method: str = "ai",
        ai_model: str = "1.0.0",
        timeout: float = 600.0,
    ) -> None:
        """Remove the sky gradient and background from a FITS image.

        Args:
            input_path: Path to the stacked FITS file.
            output_path: Desired output FITS path for the background-corrected image.
            method: Extraction method: ``"ai"`` (U-Net) or ``"polynomial"``.
            ai_model: GraXpert AI model **version** (must match
                ``^\d+\.\d+\.\d+$``, e.g. ``"1.0.0"``). For backward
                compatibility, a leading ``"GraXpert-AI-"`` prefix is stripped.
            timeout: Maximum execution time in seconds.

        Raises:
            PipelineStepException: If GraXpert cannot be found or fails.
        """
        # GraXpert validates -ai_version against ^\d+\.\d+\.\d+$ and rejects
        # anything else with argparse exit 2.  Strip the legacy prefix if a
        # stored profile still uses the old "GraXpert-AI-1.0.0" form.
        ai_version = _normalize_ai_version(ai_model)

        # GraXpert's -output flag is a *basename without extension* — it always
        # writes the result next to the input as ``<input_dir>/<output>.fits``.
        # We pass a unique stem and move the produced file to the requested
        # output_path after the run.
        output_stem = f"{input_path.stem}_GraXpertBGE"
        # GraXpert writes ``<output_stem>.<ext>`` next to the input. The
        # extension is derived from GraXpert's own format detection (``.fits``
        # or ``.xisf``) and does NOT necessarily match the input suffix — a
        # ``.fit`` input typically yields a ``.fits`` output. We glob for the
        # produced file *after* the run rather than guessing its name here.

        # Snapshot existing siblings so we can detect what GraXpert created.
        pre_existing = {
            p for p in input_path.parent.glob(f"{output_stem}.*")
        }

        # GraXpert can be invoked as a Python module or as an installed CLI entry-point
        graxpert_main = self.source_path / "GraXpert.py"
        if not graxpert_main.exists():
            cmd = self._build_command_installed(
                input_path=input_path,
                output_stem=output_stem,
                method=method,
                ai_version=ai_version,
            )
        else:
            cmd = self._build_command_script(
                script=graxpert_main,
                input_path=input_path,
                output_stem=output_stem,
                method=method,
                ai_version=ai_version,
            )

        logger.info("graxpert_starting", input=str(input_path), method=method)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise PipelineStepException(
                ErrorCode.PIPE_GRADIENT_REMOVAL_FAILED,
                f"GraXpert timed out after {timeout}s.",
                step_name="gradient_removal",
                retryable=True,
            ) from exc
        except FileNotFoundError as exc:
            raise PipelineStepException(
                ErrorCode.SYS_EXTERNAL_TOOL_MISSING,
                "GraXpert is not installed or not found at the configured path.",
                step_name="gradient_removal",
                retryable=False,
            ) from exc

        if proc.returncode != 0:
            stderr_text = stderr.decode("utf-8", errors="replace")[:500]
            raise PipelineStepException(
                ErrorCode.PIPE_GRADIENT_REMOVAL_FAILED,
                f"GraXpert failed (exit {proc.returncode}): {stderr_text}",
                step_name="gradient_removal",
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
                ErrorCode.PIPE_GRADIENT_REMOVAL_FAILED,
                (
                    f"GraXpert did not produce expected output "
                    f"({output_stem}.fits/.fit/.xisf) in {input_path.parent}"
                ),
                step_name="gradient_removal",
                retryable=False,
            )
        produced = produced_candidates[0]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()
        produced.rename(output_path)

        logger.info("graxpert_done", output=str(output_path))

    # ── Private helpers ───────────────────────────────────────────────────────

    def _gpu_flag(self) -> str:
        """Return ``'true'`` when a CUDA device is configured, ``'false'`` otherwise.

        GraXpert's ``-gpu`` flag accepts the string literals ``true`` / ``false``
        (not a device index).  We enable GPU whenever ``gpu_device`` starts with
        ``'cuda'``; the runtime will use the first visible CUDA device as
        determined by ``CUDA_VISIBLE_DEVICES`` or driver defaults.
        """
        return "true" if self.gpu_device.startswith("cuda") else "false"

    def _build_command_script(
        self,
        script: Path,
        input_path: Path,
        output_stem: str,
        method: str,
        ai_version: str,
    ) -> list[str]:
        """Build the command to invoke GraXpert as a Python script.

        Args:
            script: Path to ``GraXpert.py``.
            input_path: Input FITS path.
            output_stem: Output basename without extension (GraXpert appends
                ``.fits``/``.xisf`` and writes next to the input).
            method: Extraction method (``ai`` or ``polynomial``).
            ai_version: AI model version string (semver, e.g. ``"1.0.0"``).

        Returns:
            Command token list.
        """
        # GraXpert 3.x CLI requires -cli before -cmd to enable headless mode.
        # Without it the script prints usage help and exits 2.
        # Algorithm selection: polynomial/Splines is the default when -ai_version
        # is omitted; AI mode requires -ai_version <semver>.
        cmd = [
            sys.executable,
            str(script),
            "-cli",
            "-cmd", "background-extraction",
            "-output", output_stem,
            "-gpu", self._gpu_flag(),
        ]
        if method == "ai":
            cmd += ["-ai_version", ai_version]
        return cmd + [str(input_path)]

    def _build_command_installed(
        self,
        input_path: Path,
        output_stem: str,
        method: str,
        ai_version: str,
    ) -> list[str]:
        """Build the command when GraXpert is installed as a package.

        Args:
            input_path: Input FITS path.
            output_stem: Output basename without extension (GraXpert appends
                the extension and writes next to the input).
            method: Extraction method.
            ai_version: AI model version string (semver, e.g. ``"1.0.0"``).

        Returns:
            Command token list.
        """
        # GraXpert 3.x CLI requires -cli before -cmd to enable headless mode.
        # Without it the entry-point prints usage help and exits 2.
        # Algorithm selection: polynomial/Splines is the default when -ai_version
        # is omitted; AI mode requires -ai_version <semver>.
        cmd = [
            "graxpert",
            "-cli",
            "-cmd", "background-extraction",
            "-output", output_stem,
            "-gpu", self._gpu_flag(),
        ]
        if method == "ai":
            cmd += ["-ai_version", ai_version]
        return cmd + [str(input_path)]


_AI_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


def _normalize_ai_version(ai_model: str) -> str:
    """Coerce a profile ``ai_model`` value into the bare semver GraXpert expects.

    GraXpert validates ``-ai_version`` against ``^\\d+\\.\\d+\\.\\d+$`` and
    rejects anything else with argparse exit code 2. Older AstroStack profiles
    used the form ``"GraXpert-AI-1.0.0"`` (a model filename, not a version);
    strip such prefixes for backward compatibility.

    Args:
        ai_model: Value from the profile (e.g. ``"1.0.0"`` or
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
            fallback="1.0.0",
        )
        return "1.0.0"
    return candidate
