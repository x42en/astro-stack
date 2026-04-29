"""Cosmic Clarity AI sharpening pipeline step."""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger
from app.pipeline.adapters.cosmic_adapter import CosmicClarityAdapter
from app.pipeline.base_step import PipelineContext, PipelineStep, StepResult
from app.pipeline.utils.preview import save_step_preview

logger = get_logger(__name__)


class SharpenStep(PipelineStep):
    """Applies AI deconvolution/sharpening using Cosmic Clarity Sharpen."""

    name = "sharpen"
    display_name = "AI Sharpening / Deconvolution (Cosmic Clarity)"

    def __init__(self, adapter: CosmicClarityAdapter | None = None) -> None:
        """Initialise the step.

        Args:
            adapter: Optional Cosmic Clarity adapter.
        """
        self._adapter = adapter or CosmicClarityAdapter()

    async def execute(
        self,
        context: PipelineContext,
        config: dict[str, Any],
    ) -> StepResult:
        """Run Cosmic Clarity sharpen on the denoised (or best available) image.

        Args:
            context: Pipeline context.
            config: Profile config dict with ``sharpen_*`` fields.

        Returns:
            StepResult with ``sharpened_path`` in metadata.
        """
        if not config.get("sharpen_enabled", True):
            input_path = (
                context.denoised_path
                or context.stretched_fits_path
                or context.background_removed_path
            )
            if input_path:
                context.sharpened_path = input_path
            return StepResult(success=True, skipped=True, message="Sharpening disabled in profile.")

        input_path = (
            context.denoised_path
            or context.stretched_fits_path
            or context.background_removed_path
            or context.stacked_fits_path
        )
        if input_path is None:
            return StepResult(success=True, skipped=True, message="No input FITS for sharpening.")

        output_path = context.work_dir / "output" / "sharpened.fits"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self._adapter.gpu_device = context.gpu_device

        await self._adapter.sharpen(
            input_path=input_path,
            output_path=output_path,
            stellar_amount=float(config.get("sharpen_stellar_amount", 0.5)),
            nonstellar_amount=float(config.get("sharpen_nonstellar_amount", 0.7)),
            nonstellar_strength=float(config.get("sharpen_radius", 3.0)),
        )

        context.sharpened_path = output_path
        logger.info("sharpen_done", output=str(output_path))

        # Generate a JPEG preview from the sharpened image. Non-critical.
        preview_url: str | None = None
        try:
            preview_path = context.output_dir / "previews" / "sharpen.jpg"
            await save_step_preview(
                output_path,
                preview_path,
                camera_defiltered=bool(config.get("camera_defiltered", True)),
            )
            preview_url = f"/api/v1/sessions/{context.session_id}/step-preview/sharpen"
        except Exception:  # noqa: BLE001
            logger.warning("sharpen_preview_failed")

        return StepResult(
            success=True,
            metadata={
                "sharpened_path": str(output_path),
                **({"preview_url": preview_url} if preview_url else {}),
            },
            message="AI sharpening complete.",
        )
