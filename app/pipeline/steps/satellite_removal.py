"""Satellite/aircraft trail removal pipeline step (SASpro `cc satellite`).

New capability that did not exist in the old standalone Cosmic Clarity script
bundle. Runs last, on the best available final image, since trails are a
compositing-agnostic defect independent of stacking/stretch/denoise choices.
"""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger
from app.pipeline.adapters.cosmic_adapter import CosmicClarityAdapter
from app.pipeline.base_step import PipelineContext, PipelineStep, StepResult
from app.pipeline.utils.preview import save_step_preview

logger = get_logger(__name__)


class SatelliteRemovalStep(PipelineStep):
    """Detects and removes satellite/aircraft trails from the final image."""

    name = "satellite_removal"
    display_name = "Satellite Trail Removal (Cosmic Clarity)"

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
        """Run satellite trail removal on the best available final image.

        Args:
            context: Pipeline context.
            config: Profile config dict with ``satellite_removal_*`` fields.

        Returns:
            StepResult with ``satellite_removed_path`` in metadata.
        """
        input_path = (
            context.final_fits_path
            or context.nebula_only_path
            or context.superres_path
            or context.sharpened_path
            or context.denoised_path
            or context.stretched_fits_path
            or context.background_removed_path
            or context.stacked_fits_path
        )

        if not config.get("satellite_removal_enabled", False):
            if input_path:
                context.final_fits_path = input_path
            return StepResult(
                success=True, skipped=True, message="Satellite trail removal disabled."
            )

        if input_path is None:
            return StepResult(
                success=True, skipped=True, message="No input FITS for satellite removal."
            )

        output_path = context.work_dir / "output" / "satellite_removed.fits"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self._adapter.gpu_device = context.gpu_device

        await self._adapter.remove_satellite_trails(
            input_path=input_path,
            output_path=output_path,
            mode=str(config.get("satellite_removal_mode", "full")),
            sensitivity=float(config.get("satellite_removal_sensitivity", 0.10)),
            clip_trail=bool(config.get("satellite_removal_clip_trail", True)),
        )

        context.satellite_removed_path = output_path
        context.final_fits_path = output_path
        logger.info("satellite_removal_done", output=str(output_path))

        # Generate a JPEG preview. Non-critical.
        preview_url: str | None = None
        try:
            preview_path = context.output_dir / "previews" / "satellite_removal.jpg"
            await save_step_preview(
                output_path,
                preview_path,
                camera_defiltered=bool(config.get("camera_defiltered", True)),
            )
            preview_url = f"/api/v1/sessions/{context.session_id}/step-preview/satellite_removal"
        except Exception:  # noqa: BLE001
            logger.warning("satellite_removal_preview_failed")

        return StepResult(
            success=True,
            metadata={
                "satellite_removed_path": str(output_path),
                "preview_url": preview_url,
            },
            message="Satellite trail removal complete.",
        )
