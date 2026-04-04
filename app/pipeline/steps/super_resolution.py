"""Cosmic Clarity AI super-resolution pipeline step."""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger
from app.pipeline.adapters.cosmic_adapter import CosmicClarityAdapter
from app.pipeline.base_step import PipelineContext, PipelineStep, StepResult

logger = get_logger(__name__)


class SuperResolutionStep(PipelineStep):
    """Applies AI 2× super-resolution upscaling (optional step)."""

    name = "super_resolution"
    display_name = "AI Super-Resolution 2× (Cosmic Clarity)"

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
        """Run Cosmic Clarity super-resolution on the sharpened image.

        Args:
            context: Pipeline context.
            config: Profile config dict with ``super_resolution_*`` fields.

        Returns:
            StepResult with ``superres_path`` in metadata.
        """
        if not config.get("super_resolution_enabled", False):
            input_path = context.sharpened_path or context.denoised_path
            if input_path:
                context.superres_path = input_path
            return StepResult(success=True, skipped=True, message="Super-resolution disabled.")

        input_path = context.sharpened_path or context.denoised_path or context.stacked_fits_path
        if input_path is None:
            return StepResult(
                success=True, skipped=True, message="No input FITS for super-resolution."
            )

        output_path = context.work_dir / "output" / "superres.fits"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self._adapter.gpu_device = context.gpu_device

        await self._adapter.super_resolution(
            input_path=input_path,
            output_path=output_path,
            scale=int(config.get("super_resolution_scale", 2)),
        )

        context.superres_path = output_path
        logger.info("super_resolution_done", output=str(output_path))

        return StepResult(
            success=True,
            metadata={"superres_path": str(output_path)},
            message="AI super-resolution complete.",
        )
