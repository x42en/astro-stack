"""GraXpert gradient removal pipeline step."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.logging import get_logger
from app.pipeline.adapters.graxpert_adapter import GraXpertAdapter
from app.pipeline.base_step import PipelineContext, PipelineStep, StepResult
from app.pipeline.utils.preview import save_step_preview

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
        ai_model = config.get("gradient_removal_ai_model", "1.0.1")

        method = requested_method
        if requested_method == "ai" and not self._is_ai_model_available(ai_model):
            logger.warning(
                "graxpert_ai_model_not_found",
                model=ai_model,
                models_dir=str(self._adapter.models_path),
                message="AI model not found, falling back to polynomial method",
            )
            method = "polynomial"

        await self._adapter.remove_background(
            input_path=context.stacked_fits_path,
            output_path=output_path,
            method=method,
            ai_model=ai_model,
        )

        context.background_removed_path = output_path
        logger.info("gradient_removal_done", output=str(output_path), method=method)

        # Generate a JPEG preview from the background-removed image. Non-critical.
        preview_url: str | None = None
        try:
            preview_path = context.output_dir / "previews" / "gradient_removal.jpg"
            await save_step_preview(output_path, preview_path)
            preview_url = f"/api/v1/sessions/{context.session_id}/step-preview/gradient_removal"
        except Exception:  # noqa: BLE001
            logger.warning("gradient_removal_preview_failed")

        return StepResult(
            success=True,
            metadata={
                "background_removed_path": str(output_path),
                "method": method,
                **({"preview_url": preview_url} if preview_url else {}),
            },
            message=f"Background gradient removed using {method} method.",
        )
