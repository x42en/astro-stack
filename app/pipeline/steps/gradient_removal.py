"""GraXpert gradient removal pipeline step."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.logging import get_logger
from app.pipeline.adapters.graxpert_adapter import GraXpertAdapter
from app.pipeline.base_step import PipelineContext, PipelineStep, StepResult
from app.pipeline.utils.preview import save_step_preview

logger = get_logger(__name__)


# ── ai_model selector parsing ────────────────────────────────────────────
#
# The ``gradient_removal_ai_model`` profile field doubles as a mode +
# version selector so the existing UI dropdown can expose all of GraXpert's
# AI capabilities through a single field:
#
#   * ``"1.0.1"``                — Background Extraction (BGE) — default.
#   * ``"deconv-obj-1.0.1"``     — Deconvolution (object-only).
#   * ``"deconv-stars-1.0.0"``   — Deconvolution (stars-only).
#   * ``"deconv-both-1.0.1"``    — Object then stars deconvolution chained
#                                  (default for galaxies / clusters via the
#                                  object-type catalogue).
#
# The bare semver form is preserved for backwards compatibility with
# imported profiles created before deconvolution was wired in.
DECONV_OBJ_PREFIX = "deconv-obj-"
DECONV_STARS_PREFIX = "deconv-stars-"
DECONV_BOTH_PREFIX = "deconv-both-"


def _parse_ai_model(value: str) -> tuple[str, str]:
    """Return ``(mode, version)`` for an ``ai_model`` selector value.

    ``mode`` is one of ``"bge"``, ``"deconv-obj"``, ``"deconv-stars"``, or
    ``"deconv-both"``.  Unknown / malformed values fall back to BGE 1.0.1
    (logged at warning level).
    """
    if not isinstance(value, str) or not value:
        return ("bge", "1.0.1")
    # ``"auto"`` is the catalogue-driven sentinel resolved by the
    # orchestrator's adaptive overrides phase.  If it ever leaks down to
    # this step (e.g. unit tests, partial profile application) we fall
    # back to BGE 1.0.1 — the historical default.
    if value == "auto":
        return ("bge", "1.0.1")
    if value.startswith(DECONV_BOTH_PREFIX):
        return ("deconv-both", value[len(DECONV_BOTH_PREFIX):] or "1.0.1")
    if value.startswith(DECONV_OBJ_PREFIX):
        return ("deconv-obj", value[len(DECONV_OBJ_PREFIX):] or "1.0.1")
    if value.startswith(DECONV_STARS_PREFIX):
        return ("deconv-stars", value[len(DECONV_STARS_PREFIX):] or "1.0.0")
    # Bare semver → BGE.
    return ("bge", value)


class GradientRemovalStep(PipelineStep):
    """Removes sky gradient and background from the stacked image using GraXpert."""

    name = "gradient_removal"
    display_name = "Background Gradient Removal (GraXpert)"

    def __init__(self, adapter: GraXpertAdapter | None = None) -> None:
        """Initialise the step.

        Args:
            adapter: Optional GraXpert adapter; created from settings if not provided.
        """
        self._adapter = adapter or GraXpertAdapter()

    def _is_ai_model_available(self, ai_model: str) -> bool:
        """Check if the GraXpert AI model is available.

        Args:
            ai_model: Name of the AI model (e.g., "1.0.1").

        Returns:
            True if model files exist, False otherwise.
        """
        # GraXpert 3.x stores models under XDG_DATA_HOME/GraXpert/ with
        # subdirectories per command type:
        #   bge-ai-models/, denoise-ai-models/,
        #   deconvolution-object-ai-models/, deconvolution-stars-ai-models/
        # Each subdir contains .onnx model files.
        models_dir = self._adapter.models_path
        if not models_dir.exists():
            return False

        return any(models_dir.rglob("*.onnx"))

    @staticmethod
    def _copy_wcs_headers(source: Path, target: Path) -> None:
        """Copy WCS / plate-solve headers from ``source`` onto ``target``.

        GraXpert rewrites the FITS file from scratch and drops most non-data
        keywords, including the WCS keywords written by ASTAP. This helper
        lifts them from the platesolved input and merges them into the
        GraXpert output so downstream Siril commands (PCC) still see a
        plate-solved image.
        """
        from astropy.io import fits  # noqa: PLC0415

        # Standard WCS keywords + a few SIP/CD-matrix variants commonly written
        # by ASTAP. We deliberately keep the list narrow to avoid clobbering
        # statistics keywords GraXpert may legitimately have updated.
        wcs_keys = {
            "WCSAXES", "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2",
            "CTYPE1", "CTYPE2", "CUNIT1", "CUNIT2",
            "CDELT1", "CDELT2", "CROTA1", "CROTA2",
            "CD1_1", "CD1_2", "CD2_1", "CD2_2",
            "PC1_1", "PC1_2", "PC2_1", "PC2_2",
            "RADESYS", "EQUINOX", "LONPOLE", "LATPOLE",
            "RA", "DEC", "OBJCTRA", "OBJCTDEC",
            "PLTSOLVD",
        }

        with fits.open(source) as src:
            src_hdr = src[0].header
            keys_to_copy = [(k, src_hdr[k], src_hdr.comments[k]) for k in src_hdr if k in wcs_keys]
            # Preserve SIP higher-order distortion terms if any (A_*, B_*, AP_*, BP_*).
            for k in src_hdr:
                if k.startswith(("A_", "B_", "AP_", "BP_")):
                    keys_to_copy.append((k, src_hdr[k], src_hdr.comments[k]))

        if not keys_to_copy:
            logger.warning(
                "gradient_removal_wcs_copy_empty",
                source=str(source),
                source_n_hdu=None,
            )
            return

        with fits.open(target, mode="update") as tgt:
            tgt_hdr = tgt[0].header
            for key, value, comment in keys_to_copy:
                tgt_hdr[key] = (value, comment)
            tgt.flush()
            target_n_hdu = len(tgt)
            target_hdu0_shape = getattr(tgt[0].data, "shape", None)
        logger.info(
            "gradient_removal_wcs_copied",
            count=len(keys_to_copy),
            keys=sorted({k for k, _, _ in keys_to_copy})[:20],
            target_n_hdu=target_n_hdu,
            target_hdu0_shape=target_hdu0_shape,
        )

    async def execute(
        self,
        context: PipelineContext,
        config: dict[str, Any],
    ) -> StepResult:
        """Run GraXpert on the stacked FITS image.

        Args:
            context: Pipeline context with ``stacked_fits_path``.
            config: Profile config dict with ``gradient_removal_*`` fields.

        Returns:
            StepResult with ``background_removed_path`` in metadata.
        """
        if not config.get("gradient_removal_enabled", True):
            # Pass-through: later steps use stacked_fits as input
            if context.stacked_fits_path:
                context.background_removed_path = context.stacked_fits_path
            return StepResult(success=True, skipped=True, message="Gradient removal disabled.")

        if context.stacked_fits_path is None:
            return StepResult(success=True, skipped=True, message="No stacked FITS available.")

        output_path = context.work_dir / "output" / "background_removed.fits"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self._adapter.gpu_device = context.gpu_device

        # Determine method: use AI if model is available, else fallback to polynomial
        requested_method = config.get("gradient_removal_method", "ai")
        ai_model_value = str(config.get("gradient_removal_ai_model", "1.0.1"))
        mode, ai_version = _parse_ai_model(ai_model_value)

        method = requested_method
        if requested_method == "ai" and not self._is_ai_model_available(ai_version):
            logger.warning(
                "graxpert_ai_model_not_found",
                model=ai_version,
                models_dir=str(self._adapter.models_path),
                message="AI model not found, falling back to polynomial method",
            )
            method = "polynomial"
            mode = "bge"

        # Polynomial / BGE path → existing background-extraction adapter.
        if method == "polynomial" or mode == "bge":
            await self._adapter.remove_background(
                input_path=context.stacked_fits_path,
                output_path=output_path,
                method=method,
                ai_model=ai_version,
                correction=str(config.get("gradient_removal_correction", "Subtraction")),
                smoothing=float(config.get("gradient_removal_smoothing", 1.0)),
            )
            effective_method = method
        else:
            # Deconvolution path: object-only, stars-only, or both chained.
            strength = float(config.get("gradient_removal_deconv_strength", 0.5))
            psfsize = float(config.get("gradient_removal_deconv_psfsize", 0.3))
            batch_size = int(config.get("gradient_removal_deconv_batch_size", 4))
            if mode == "deconv-both":
                # First pass: deconvolve the object (nebula / galaxy core).
                intermediate = output_path.with_name("background_removed_obj.fits")
                await self._adapter.deconvolve(
                    input_path=context.stacked_fits_path,
                    output_path=intermediate,
                    target="object",
                    ai_model=ai_version,
                    strength=strength,
                    psfsize=psfsize,
                    batch_size=batch_size,
                )
                # Chain: stellar deconvolution on the object-deconvolved frame.
                await self._adapter.deconvolve(
                    input_path=intermediate,
                    output_path=output_path,
                    target="stars",
                    ai_model="1.0.0",  # only version published by GraXpert at writing.
                    strength=strength,
                    psfsize=psfsize,
                    batch_size=batch_size,
                )
                effective_method = "deconv-both"
            else:
                target = "object" if mode == "deconv-obj" else "stars"
                # Stars-only model is published as 1.0.0; the object model
                # we honour from the user-supplied selector.
                target_version = ai_version if target == "object" else "1.0.0"
                await self._adapter.deconvolve(
                    input_path=context.stacked_fits_path,
                    output_path=output_path,
                    target=target,
                    ai_model=target_version,
                    strength=strength,
                    psfsize=psfsize,
                    batch_size=batch_size,
                )
                effective_method = mode

        context.background_removed_path = output_path

        # GraXpert strips the WCS / plate-solve headers from its output. Without
        # them the downstream Siril ``pcc`` (Photometric Colour Calibration)
        # bails out with "This command only works on plate solved images".
        # Copy the astrometric keywords from the platesolved input back onto the
        # GraXpert output so PCC (and any subsequent astrometry-aware step)
        # keeps working. Best-effort: never fail the pipeline on this.
        try:
            self._copy_wcs_headers(context.stacked_fits_path, output_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("gradient_removal_wcs_copy_failed", error=str(exc))

        logger.info("gradient_removal_done", output=str(output_path), method=effective_method)

        # Generate a JPEG preview from the background-removed image. Non-critical.
        preview_url: str | None = None
        try:
            preview_path = context.output_dir / "previews" / "gradient_removal.jpg"
            await save_step_preview(
                output_path,
                preview_path,
                camera_defiltered=bool(config.get("camera_defiltered", True)),
            )
            preview_url = f"/api/v1/sessions/{context.session_id}/step-preview/gradient_removal"
        except Exception:  # noqa: BLE001
            logger.warning("gradient_removal_preview_failed")

        return StepResult(
            success=True,
            metadata={
                "background_removed_path": str(output_path),
                "method": effective_method,
                **({"preview_url": preview_url} if preview_url else {}),
            },
            message=f"Background gradient removed using {effective_method} method.",
        )
