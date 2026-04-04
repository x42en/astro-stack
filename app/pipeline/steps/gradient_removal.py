"""GraXpert gradient removal pipeline step."""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger
from app.pipeline.adapters.graxpert_adapter import GraXpertAdapter
from app.pipeline.base_step import PipelineContext, PipelineStep, StepResult

logger = get_logger(__name__)


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

        await self._adapter.remove_background(
            input_path=context.stacked_fits_path,
            output_path=output_path,
            method=str(config.get("gradient_removal_method", "ai")),
            ai_model=str(config.get("gradient_removal_ai_model", "GraXpert-AI-1.0.0")),
        )

        context.background_removed_path = output_path
        logger.info("gradient_removal_done", output=str(output_path))

        return StepResult(
            success=True,
            metadata={"background_removed_path": str(output_path)},
            message="Background gradient removed.",
        )
