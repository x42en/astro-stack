"""Final export pipeline step.

Converts the processed FITS image to multiple output formats:
- FITS 32-bit float (scientific archive)
- TIFF 16-bit (manual editing)
- JPEG high-quality preview
- PNG thumbnail (800px, for API previews)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from app.core.logging import get_logger
from app.pipeline.base_step import PipelineContext, PipelineStep, StepResult
from app.core.errors import ErrorCode, PipelineStepException

logger = get_logger(__name__)

_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="export")


class ExportStep(PipelineStep):
    """Exports the final FITS image to TIFF, JPEG, and PNG thumbnail formats."""

    name = "export"
    display_name = "Export (FITS / TIFF / JPEG / Thumbnail)"

    async def execute(
        self,
        context: PipelineContext,
        config: dict[str, Any],
    ) -> StepResult:
        """Export the best available image to all output formats.

        Args:
            context: Pipeline context. Uses ``final_fits_path`` if set,
                otherwise falls through the processing chain.
            config: Profile config dict (unused directly by this step).

        Returns:
            StepResult with ``fits_path``, ``tiff_path``, ``jpeg_path``,
            and ``thumbnail_path`` in metadata.
        """
        source_fits = (
            context.final_fits_path
            or context.superres_path
            or context.sharpened_path
            or context.denoised_path
            or context.background_removed_path
            or context.stacked_fits_path
        )

        if source_fits is None:
            raise PipelineStepException(
                ErrorCode.PIPE_EXPORT_FAILED,
                "No processed FITS image available for export.",
                step_name=self.name,
                retryable=False,
            )

        out_dir = context.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        import asyncio  # noqa: PLC0415

        loop = asyncio.get_event_loop()

        fits_out = out_dir / "final.fits"
        tiff_out = out_dir / "final.tiff"
        jpeg_out = out_dir / "preview.jpg"
        thumb_out = out_dir / "thumbnail.png"

        try:
            # Copy FITS as-is
            await loop.run_in_executor(_EXECUTOR, _copy_fits, source_fits, fits_out)
            # Export raster formats
            await loop.run_in_executor(
                _EXECUTOR, _export_raster, source_fits, tiff_out, jpeg_out, thumb_out
            )
        except Exception as exc:  # noqa: BLE001
            raise PipelineStepException(
                ErrorCode.PIPE_EXPORT_FAILED,
                f"Export failed: {exc}",
                step_name=self.name,
                retryable=False,
            ) from exc

        context.final_fits_path = fits_out
        logger.info(
            "export_done",
            fits=str(fits_out),
            tiff=str(tiff_out),
            jpeg=str(jpeg_out),
        )

        return StepResult(
            success=True,
            metadata={
                "fits_path": str(fits_out),
                "tiff_path": str(tiff_out),
                "jpeg_path": str(jpeg_out),
                "thumbnail_path": str(thumb_out),
            },
            message="Export complete.",
        )


# ── Private helpers ───────────────────────────────────────────────────────────


def _copy_fits(src: Path, dst: Path) -> None:
    """Copy a FITS file to the output directory.

    Args:
        src: Source FITS path.
        dst: Destination FITS path.
    """
    import shutil  # noqa: PLC0415

    shutil.copy2(str(src), str(dst))


def _export_raster(
    fits_path: Path,
    tiff_path: Path,
    jpeg_path: Path,
    thumb_path: Path,
) -> None:
    """Convert a FITS image to TIFF, JPEG, and thumbnail.

    Applies auto-stretch to map the 32-bit float FITS data to 8-bit/16-bit
    display ranges. Uses astropy + Pillow.

    Args:
        fits_path: Source FITS file.
        tiff_path: Output 16-bit TIFF path.
        jpeg_path: Output JPEG path (quality 95).
        thumb_path: Output PNG thumbnail (800px wide).
    """
    import numpy as np  # noqa: PLC0415
    from astropy.io import fits  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    with fits.open(str(fits_path)) as hdul:
        data = hdul[0].data

    if data is None:
        raise ValueError(f"FITS file {fits_path} has no image data.")

    # Handle 3-channel (CxHxW) or 2D (HxW) arrays
    if data.ndim == 3:
        # Convert (C, H, W) → (H, W, C) for Pillow
        arr = np.moveaxis(data, 0, -1)
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
    else:
        arr = data

    # Sanitise non-finite values before any arithmetic (NaN/inf in FITS → bad casts)
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)

    # Auto-stretch: clip to 0.01–99.99th percentile
    lo, hi = np.percentile(arr, (0.01, 99.99))
    if hi == lo:
        hi = lo + 1.0
    arr_norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)

    # 16-bit TIFF — PIL.fromarray does not support uint16 RGB;
    # use tifffile when available (installed via Cosmic Clarity requirements), else uint8.
    arr_16 = (arr_norm * 65535).astype(np.uint16)
    try:
        import tifffile as _tifffile  # noqa: PLC0415
        _photometric = "rgb" if arr_16.ndim == 3 and arr_16.shape[2] == 3 else "minisblack"
        _tifffile.imwrite(str(tiff_path), arr_16, photometric=_photometric, compression="deflate")
    except ImportError:
        # Fallback: 8-bit TIFF (PIL uint16 RGB is unsupported)
        arr_fb = (arr_norm * 255).astype(np.uint8)
        img_fb = (
            Image.fromarray(arr_fb)
            if arr_fb.ndim == 2
            else Image.fromarray(arr_fb, mode="RGB" if arr_fb.shape[2] == 3 else "L")
        )
        img_fb.save(str(tiff_path), format="TIFF", compression="tiff_deflate")

    # JPEG (8-bit)
    arr_8 = (arr_norm * 255).astype(np.uint8)
    img_8 = Image.fromarray(arr_8)
    if img_8.mode == "I;16":
        img_8 = img_8.convert("L")
    img_8.save(str(jpeg_path), format="JPEG", quality=95)

    # Thumbnail (800px wide)
    thumb = img_8.copy()
    w, h = thumb.size
    if w > 800:
        new_h = int(h * 800 / w)
        thumb = thumb.resize((800, new_h), Image.LANCZOS)
    thumb.save(str(thumb_path), format="PNG")
