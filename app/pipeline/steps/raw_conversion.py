"""Raw conversion pipeline step.

Converts DSLR RAW image files (CR2, NEF, ARW, etc.) to FITS format using
``rawpy`` (LibRaw) so that Siril can process them in its native format.
FITS files are passed through unchanged. Mixed sessions convert only the
RAW files.

Example:
    >>> step = RawConversionStep()
    >>> result = await step.execute(context, config)
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from app.core.errors import ErrorCode, PipelineStepException
from app.core.logging import get_logger
from app.infrastructure.storage.file_store import FITS_EXTENSIONS, RAW_DSLR_EXTENSIONS
from app.pipeline.base_step import PipelineContext, PipelineStep, StepResult

logger = get_logger(__name__)

_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="raw_conv")


class RawConversionStep(PipelineStep):
    """Converts DSLR RAW frames to FITS format.

    Scans the session inbox for RAW files and converts each one using
    LibRaw (via the ``rawpy`` Python binding). The converted FITS files
    are written into the session working directory, preserving the
    sub-folder structure (lights/, darks/, etc.).

    This step is skipped if the session input format is ``fits``.
    """

    name = "raw_conversion"
    display_name = "RAW → FITS Conversion"

    async def execute(
        self,
        context: PipelineContext,
        config: dict[str, Any],
    ) -> StepResult:
        """Convert all RAW frames in the session to FITS.

        Args:
            context: Shared pipeline context.
            config: Profile config dict (unused by this step).

        Returns:
            :class:`~app.pipeline.base_step.StepResult` with a count of
            converted files in ``metadata["converted_count"]``.

        Raises:
            PipelineStepException: If a RAW file cannot be decoded.
        """
        input_format = context.metadata.get("input_format", "fits")
        if input_format == "fits":
            logger.info("raw_conversion_skipped", reason="input_format=fits")
            return StepResult(
                success=True, skipped=True, message="Input is FITS, skipping RAW conversion."
            )

        frames: dict[str, list[Path]] = context.metadata.get("frames", {})
        raw_files: list[Path] = [
            f for files in frames.values() for f in files if f.suffix.lower() in RAW_DSLR_EXTENSIONS
        ]

        if not raw_files:
            return StepResult(success=True, skipped=True, message="No RAW files found.")

        logger.info("raw_conversion_starting", count=len(raw_files))
        converted = 0
        loop = asyncio.get_event_loop()

        for raw_path in raw_files:
            if context.cancelled:
                break
            fits_path = _target_fits_path(raw_path, context.work_dir)
            fits_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                await loop.run_in_executor(
                    _EXECUTOR,
                    _convert_raw_to_fits,
                    raw_path,
                    fits_path,
                )
                converted += 1
                logger.debug("raw_converted", src=str(raw_path), dst=str(fits_path))
            except Exception as exc:  # noqa: BLE001
                raise PipelineStepException(
                    ErrorCode.PIPE_RAW_CONVERSION_FAILED,
                    f"Failed to convert RAW file {raw_path.name}: {exc}",
                    step_name=self.name,
                    retryable=False,
                    details={"file": str(raw_path)},
                ) from exc

        # Update frame paths in context to point to converted FITS files
        _remap_frames_to_fits(frames, context.work_dir)

        logger.info("raw_conversion_done", converted=converted)
        return StepResult(
            success=True,
            metadata={"converted_count": converted},
            message=f"Converted {converted} RAW files to FITS.",
        )


# ── Private helpers ───────────────────────────────────────────────────────────


def _target_fits_path(raw_path: Path, work_dir: Path) -> Path:
    """Return the FITS output path for a given RAW source file.

    Preserves the last two path components (type-dir / filename).

    Args:
        raw_path: Source RAW file path.
        work_dir: Session working directory.

    Returns:
        Target FITS path inside the working directory.
    """
    parts = raw_path.parts
    if len(parts) >= 2:
        rel = Path(parts[-2]) / raw_path.stem
    else:
        rel = Path(raw_path.stem)
    return work_dir / rel.with_suffix(".fits")


def _convert_raw_to_fits(raw_path: Path, fits_path: Path) -> None:
    """Synchronous RAW → FITS conversion using rawpy and astropy.

    Runs in a thread pool to avoid blocking the event loop.

    Args:
        raw_path: Source DSLR RAW file.
        fits_path: Destination FITS file path.

    Raises:
        ImportError: If ``rawpy`` or ``astropy`` is not installed.
        Exception: Any rawpy or I/O error.
    """
    import rawpy  # noqa: PLC0415 — lazy import to avoid startup cost
    from astropy.io import fits  # noqa: PLC0415

    with rawpy.imread(str(raw_path)) as raw:
        # Read raw Bayer CFA data directly — do NOT debayer.
        # Siril will calibrate (dark/flat) at the CFA level, then debayer,
        # which is the correct astrophotography workflow.
        bayer = raw.raw_image_visible.copy()          # uint16, shape (H, W)
        bayer_pattern = raw.color_desc.decode("ascii")  # e.g. "RGGB"

    data = bayer.astype(np.float32)

    hdu = fits.PrimaryHDU(data=data)
    hdu.header["ORIGINAL"] = str(raw_path.name)
    # Standard Bayer headers that Siril reads to auto-detect the CFA pattern.
    hdu.header["BAYERPAT"] = bayer_pattern
    hdu.header["XBAYROFF"] = 0
    hdu.header["YBAYROFF"] = 0
    hdu.writeto(str(fits_path), overwrite=True)


def _remap_frames_to_fits(
    frames: dict[str, list[Path]],
    work_dir: Path,
) -> None:
    """Replace RAW file paths in the frames dict with their FITS equivalents.

    Modifies ``frames`` in-place, replacing any RAW paths with the expected
    FITS paths in the working directory.

    Args:
        frames: Frame inventory dict to update.
        work_dir: Session working directory.
    """
    for frame_type, paths in frames.items():
        updated: list[Path] = []
        for p in paths:
            if p.suffix.lower() in RAW_DSLR_EXTENSIONS:
                updated.append(_target_fits_path(p, work_dir))
            else:
                updated.append(p)
        frames[frame_type] = updated
