"""Cosmic Clarity AI denoise pipeline step."""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger
from app.pipeline.adapters.cosmic_adapter import CosmicClarityAdapter
from app.pipeline.base_step import PipelineContext, PipelineStep, StepResult
from app.pipeline.utils.preview import save_step_preview

logger = get_logger(__name__)


class DenoiseStep(PipelineStep):
    """Applies AI-based noise reduction using Cosmic Clarity Denoise."""

    name = "denoise"
    display_name = "AI Noise Reduction (Cosmic Clarity)"

    def __init__(self, adapter: CosmicClarityAdapter | None = None) -> None:
        """Initialise the step.

        Args:
            adapter: Optional Cosmic Clarity adapter; created from settings if not provided.
        """
        self._adapter = adapter or CosmicClarityAdapter()

    async def execute(
        self,
        context: PipelineContext,
        config: dict[str, Any],
    ) -> StepResult:
        """Run Cosmic Clarity denoise on the current best image.

        Args:
            context: Pipeline context. Uses ``background_removed_path`` if set,
                otherwise ``stacked_fits_path``.
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

        self._adapter.gpu_device = context.gpu_device

        await self._adapter.denoise(
            input_path=input_path,
            output_path=output_path,
            strength=float(config.get("denoise_strength", 0.8)),
            luminance_only=bool(config.get("denoise_luminance_only", False)),
        )

        context.denoised_path = output_path
        logger.info("denoise_done", output=str(output_path))

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
                **({"preview_url": preview_url} if preview_url else {}),
            },
            message="AI noise reduction complete.",
        )
