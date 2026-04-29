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
from app.pipeline.utils.preview import save_step_preview

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
            # Profile config dict — already injected by the orchestrator into
            # context.metadata; falls back to the raw config the step receives.
            profile_config = context.metadata.get("profile_config") or config
            # Copy FITS as-is, then stamp with provenance HISTORY cards.
            await loop.run_in_executor(
                _EXECUTOR, _copy_fits, source_fits, fits_out, profile_config
            )
            # Export raster formats (TIFF tags + JPEG badge embedded).
            await loop.run_in_executor(
                _EXECUTOR,
                _export_raster,
                source_fits,
                tiff_out,
                jpeg_out,
                thumb_out,
                profile_config,
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

        # Copy the export JPEG into the per-step previews directory. Non-critical.
        preview_url: str | None = None
        try:
            preview_path = context.output_dir / "previews" / "export.jpg"
            await save_step_preview(
                source_fits,
                preview_path,
                camera_defiltered=bool(config.get("camera_defiltered", True)),
            )
            preview_url = f"/api/v1/sessions/{context.session_id}/step-preview/export"
        except Exception:  # noqa: BLE001
            logger.warning("export_preview_failed")

        return StepResult(
            success=True,
            metadata={
                "fits_path": str(fits_out),
                "tiff_path": str(tiff_out),
                "jpeg_path": str(jpeg_out),
                "thumbnail_path": str(thumb_out),
                **({"preview_url": preview_url} if preview_url else {}),
                # Surface adaptive overrides applied at job start so the UI
                # can render a "what the catalogue changed" panel after the
                # job completes (see :class:`AdaptiveOverridesPanel`).
                **(
                    {"adaptive_overrides_applied": context.metadata["adaptive_overrides_applied"]}
                    if isinstance(context.metadata.get("adaptive_overrides_applied"), dict)
                    else {}
                ),
            },
            message="Export complete.",
        )


# ── Private helpers ───────────────────────────────────────────────────────────


def _copy_fits(src: Path, dst: Path, profile_config: dict[str, Any] | None = None) -> None:
    """Copy a FITS file to the output directory and stamp provenance.

    Adds HISTORY cards identifying AstroStack as the producer plus a series
    of COMMENT cards holding a JSON dump of the active profile parameters.
    Falls back to a plain copy if the FITS update path raises (e.g. astropy
    missing or header full).

    Args:
        src: Source FITS path.
        dst: Destination FITS path.
        profile_config: Active processing profile (dict).
    """
    import shutil  # noqa: PLC0415

    shutil.copy2(str(src), str(dst))
    try:
        _stamp_fits_metadata(dst, profile_config)
    except Exception as exc:  # noqa: BLE001
        logger.warning("export_fits_stamp_failed", error=str(exc))


def _stamp_fits_metadata(path: Path, profile_config: dict[str, Any] | None) -> None:
    """Append AstroStack provenance HISTORY/COMMENT cards to a FITS header."""
    import json  # noqa: PLC0415

    from astropy.io import fits  # noqa: PLC0415

    from app.pipeline.utils.display import (  # noqa: PLC0415
        ASTROSTACK_BRAND,
        ASTROSTACK_URL,
        summarize_profile_config,
    )

    with fits.open(str(path), mode="update") as hdul:
        hdr = hdul[0].header
        hdr["CREATOR"] = ("AstroStack", "Pipeline used to produce this file")
        hdr.add_history(f"{ASTROSTACK_BRAND} - {ASTROSTACK_URL}")
        for label, value in summarize_profile_config(profile_config or {}):
            # FITS card payload limited to ~70 chars; keep it short.
            hdr.add_history(f"{label}: {value}"[:70])
        if profile_config:
            try:
                payload = json.dumps(profile_config, separators=(",", ":"), default=str)
            except (TypeError, ValueError):
                payload = ""
            # Split JSON across multiple COMMENT cards (max ~70 chars each).
            for i in range(0, len(payload), 70):
                hdr.add_comment(f"AS_PROFILE: {payload[i:i + 70]}")
        hdul.flush()


def _export_raster(
    fits_path: Path,
    tiff_path: Path,
    jpeg_path: Path,
    thumb_path: Path,
    profile_config: dict[str, Any] | None = None,
) -> None:
    """Convert a FITS image to TIFF, JPEG, and thumbnail.

    Uses :func:`app.pipeline.utils.display.load_fits_display_rgb` to apply the
    same per-channel percentile + asinh midtone stretch as the in-pipeline
    previews. This guarantees the final ``preview.jpg`` matches the per-step
    JPEGs the user has been seeing during processing.

    Args:
        fits_path: Source FITS file.
        tiff_path: Output 16-bit TIFF path.
        jpeg_path: Output JPEG path (quality 95).
        thumb_path: Output PNG thumbnail (800px wide).
    """
    from PIL import Image  # noqa: PLC0415

    from app.pipeline.utils.display import (  # noqa: PLC0415
        ASTROSTACK_BRAND,
        ASTROSTACK_URL,
        apply_hdr_polish,
        load_fits_display_rgb,
        render_metadata_badge,
        summarize_profile_config,
        to_uint8,
        to_uint16,
    )

    camera_defiltered = bool((profile_config or {}).get("camera_defiltered", True))
    arr_norm = load_fits_display_rgb(fits_path, camera_defiltered=camera_defiltered)

    # Apply a gentle HDR-style polish (midtone S-curve + highlight rolloff +
    # mild saturation boost) on the deliverable rasters. The pristine FITS
    # archive copy is left untouched for scientific reuse.
    arr_polished = apply_hdr_polish(arr_norm, camera_defiltered=camera_defiltered)

    # Build provenance strings shared by TIFF tags + JPEG badge.
    summary_pairs = summarize_profile_config(profile_config or {})
    summary_text = "; ".join(f"{k}={v}" for k, v in summary_pairs)
    tiff_description = f"{ASTROSTACK_BRAND} ({ASTROSTACK_URL}). {summary_text}".strip()

    # 16-bit TIFF — PIL.fromarray does not support uint16 RGB;
    # use tifffile when available (installed via Cosmic Clarity requirements), else uint8.
    arr_16 = to_uint16(arr_polished)
    try:
        import tifffile as _tifffile  # noqa: PLC0415
        _photometric = "rgb" if arr_16.ndim == 3 and arr_16.shape[2] == 3 else "minisblack"
        # TIFF tag 305 = Software, 270 = ImageDescription, 315 = Artist.
        extratags = [
            (305, "s", 0, "AstroStack", True),
            (270, "s", 0, tiff_description, True),
            (315, "s", 0, ASTROSTACK_URL, True),
        ]
        _tifffile.imwrite(
            str(tiff_path),
            arr_16,
            photometric=_photometric,
            compression="deflate",
            extratags=extratags,
        )
    except ImportError:
        # Fallback: 8-bit TIFF (PIL uint16 RGB is unsupported)
        arr_fb = to_uint8(arr_polished)
        img_fb = (
            Image.fromarray(arr_fb)
            if arr_fb.ndim == 2
            else Image.fromarray(arr_fb, mode="RGB" if arr_fb.shape[2] == 3 else "L")
        )
        img_fb.save(str(tiff_path), format="TIFF", compression="tiff_deflate")

    # JPEG (8-bit) — also from the polished rendition for consistency, with a
    # composited provenance badge along the bottom edge.
    arr_8 = to_uint8(arr_polished)
    img_8 = (
        Image.fromarray(arr_8, mode="L").convert("RGB")
        if arr_8.ndim == 2
        else Image.fromarray(arr_8, mode="RGB")
    )
    try:
        img_branded = render_metadata_badge(img_8, summary_pairs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("export_jpeg_badge_failed", error=str(exc))
        img_branded = img_8
    img_branded.save(str(jpeg_path), format="JPEG", quality=95)

    # Thumbnail (800px wide) — built from the un-branded 8-bit array so the
    # badge is not duplicated on the small preview.
    thumb = img_8.copy()
    w, h = thumb.size
    if w > 800:
        new_h = int(h * 800 / w)
        thumb = thumb.resize((800, new_h), Image.LANCZOS)
    thumb.save(str(thumb_path), format="PNG")
