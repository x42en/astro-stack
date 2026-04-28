"""AI denoise pipeline step (Cosmic Clarity or GraXpert engine).

The active engine is selected per-profile via ``denoise_engine``.  Both
engines share the same input/output convention so downstream steps (sharpen,
star-separation) are agnostic to the choice.
"""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger
from app.pipeline.adapters.cosmic_adapter import CosmicClarityAdapter
from app.pipeline.adapters.graxpert_adapter import GraXpertAdapter
from app.pipeline.base_step import PipelineContext, PipelineStep, StepResult
from app.pipeline.utils.preview import save_step_preview

logger = get_logger(__name__)


class DenoiseStep(PipelineStep):
    """Applies AI-based noise reduction using Cosmic Clarity or GraXpert."""

    name = "denoise"
    display_name = "AI Noise Reduction"

    def __init__(
        self,
        adapter: CosmicClarityAdapter | None = None,
        graxpert_adapter: GraXpertAdapter | None = None,
    ) -> None:
        """Initialise the step.

        Args:
            adapter: Optional Cosmic Clarity adapter (default engine).
            graxpert_adapter: Optional GraXpert adapter; only instantiated
                when the active profile selects ``denoise_engine='graxpert'``.
        """
        self._cosmic = adapter or CosmicClarityAdapter()
        self._graxpert = graxpert_adapter  # lazy: built on first GraXpert run

    async def execute(
        self,
        context: PipelineContext,
        config: dict[str, Any],
    ) -> StepResult:
        """Run AI denoise on the current best image using the configured engine.

        Args:
            context: Pipeline context. Uses ``stretched_fits_path`` if set,
                otherwise ``background_removed_path``, otherwise
                ``stacked_fits_path``.
            config: Profile config dict with ``denoise_*`` fields.

        Returns:
            StepResult with ``denoised_path`` in metadata.
        """
        if not config.get("denoise_enabled", True):
            input_path = (
                context.stretched_fits_path
                or context.background_removed_path
                or context.stacked_fits_path
            )
            if input_path:
                context.denoised_path = input_path
            return StepResult(success=True, skipped=True, message="Denoise disabled in profile.")

        input_path = (
            context.stretched_fits_path
            or context.background_removed_path
            or context.stacked_fits_path
        )
        if input_path is None:
            return StepResult(success=True, skipped=True, message="No input FITS for denoise.")

        output_path = context.work_dir / "output" / "denoised.fits"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Engine dispatch ── default to the historical Cosmic Clarity engine
        # so existing profiles (and the migrations data backfill) keep their
        # behaviour unchanged.
        engine = str(config.get("denoise_engine", "cosmic_clarity")).lower()
        strength = float(config.get("denoise_strength", 0.8))

        if engine == "graxpert":
            if self._graxpert is None:
                self._graxpert = GraXpertAdapter()
            self._graxpert.gpu_device = context.gpu_device
            ai_model = str(config.get("denoise_graxpert_ai_model", "3.0.2"))
            batch_size = int(config.get("denoise_graxpert_batch_size", 4))
            logger.info(
                "denoise_engine_selected",
                engine="graxpert",
                strength=strength,
                ai_model=ai_model,
                batch_size=batch_size,
            )
            await self._graxpert.denoise(
                input_path=input_path,
                output_path=output_path,
                strength=strength,
                ai_model=ai_model,
                batch_size=batch_size,
            )
            engine_label = "GraXpert"
        else:
            # ``cosmic_clarity`` (default) and any unknown value fall back here.
            if engine != "cosmic_clarity":
                logger.warning(
                    "denoise_engine_unknown",
                    requested=engine,
                    fallback="cosmic_clarity",
                )
            self._cosmic.gpu_device = context.gpu_device
            luminance_only = bool(config.get("denoise_luminance_only", False))
            logger.info(
                "denoise_engine_selected",
                engine="cosmic_clarity",
                strength=strength,
                luminance_only=luminance_only,
            )
            await self._cosmic.denoise(
                input_path=input_path,
                output_path=output_path,
                strength=strength,
                luminance_only=luminance_only,
            )
            engine_label = "Cosmic Clarity"

        context.denoised_path = output_path
        logger.info("denoise_done", output=str(output_path), engine=engine_label)

        # Generate a JPEG preview from the denoised image. Non-critical.
        preview_url: str | None = None
        try:
            preview_path = context.output_dir / "previews" / "denoise.jpg"
            await save_step_preview(output_path, preview_path)
            preview_url = f"/api/v1/sessions/{context.session_id}/step-preview/denoise"
        except Exception:  # noqa: BLE001
            logger.warning("denoise_preview_failed")

        return StepResult(
            success=True,
            metadata={
                "denoised_path": str(output_path),
                "denoise_engine": engine_label,
                **({"preview_url": preview_url} if preview_url else {}),
            },
            message=f"AI noise reduction complete ({engine_label}).",
        )
