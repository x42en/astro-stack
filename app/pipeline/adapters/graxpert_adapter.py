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
        self.models_path = Path(models_path or settings.models_path) / "graxpert"
        self.gpu_device = gpu_device
        self._device_index = gpu_device.split(":")[-1] if ":" in gpu_device else "0"

    async def remove_background(
        self,
        input_path: Path,
        output_path: Path,
        method: str = "ai",
        ai_model: str = "GraXpert-AI-1.0.0",
        timeout: float = 600.0,
    ) -> None:
        """Remove the sky gradient and background from a FITS image.

        Args:
            input_path: Path to the stacked FITS file.
            output_path: Desired output FITS path for the background-corrected image.
            method: Extraction method: ``"ai"`` (U-Net) or ``"polynomial"``.
            ai_model: Name of the GraXpert AI model to use.
            timeout: Maximum execution time in seconds.

        Raises:
            PipelineStepException: If GraXpert cannot be found or fails.
        """
        # GraXpert can be invoked as a Python module or as an installed CLI entry-point
        graxpert_main = self.source_path / "GraXpert.py"
        if not graxpert_main.exists():
            # Try installed entry-point
            graxpert_main_str = "graxpert"
            cmd = self._build_command_installed(
                input_path=input_path,
                output_path=output_path,
                method=method,
                ai_model=ai_model,
            )
        else:
            cmd = self._build_command_script(
                script=graxpert_main,
                input_path=input_path,
                output_path=output_path,
                method=method,
                ai_model=ai_model,
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

        logger.info("graxpert_done", output=str(output_path))

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_command_script(
        self,
        script: Path,
        input_path: Path,
        output_path: Path,
        method: str,
        ai_model: str,
    ) -> list[str]:
        """Build the command to invoke GraXpert as a Python script.

        Args:
            script: Path to ``GraXpert.py``.
            input_path: Input FITS path.
            output_path: Output FITS path.
            method: Extraction method (``ai`` or ``polynomial``).
            ai_model: AI model identifier.

        Returns:
            Command token list.
        """
        cmd = [
            sys.executable,
            str(script),
            "--cli",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--gpu",
            self._device_index,
        ]
        if method == "ai":
            cmd += ["--ai_model", ai_model, "--models_dir", str(self.models_path)]
        else:
            cmd += ["--background_model", "Polynomial"]
        return cmd

    def _build_command_installed(
        self,
        input_path: Path,
        output_path: Path,
        method: str,
        ai_model: str,
    ) -> list[str]:
        """Build the command when GraXpert is installed as a package.

        Args:
            input_path: Input FITS path.
            output_path: Output FITS path.
            method: Extraction method.
            ai_model: AI model identifier.

        Returns:
            Command token list.
        """
        cmd = [
            "graxpert",
            "--cli",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--gpu",
            self._device_index,
        ]
        if method == "ai":
            cmd += ["--ai_model", ai_model, "--models_dir", str(self.models_path)]
        else:
            cmd += ["--background_model", "Polynomial"]
        return cmd
