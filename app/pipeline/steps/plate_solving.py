"""Plate-solving pipeline step using ASTAP."""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger
from app.pipeline.adapters.astap_adapter import AstapAdapter
from app.pipeline.base_step import PipelineContext, PipelineStep, StepResult

logger = get_logger(__name__)


class PlateSolvingStep(PipelineStep):
    """Identifies the sky coordinates of a stacked image using ASTAP.

    On success, updates the pipeline context with the resolved RA/Dec
    and object name, which are later persisted to the session record.
    """

    name = "plate_solving"
    display_name = "Plate Solving (ASTAP)"

    def __init__(self, adapter: AstapAdapter | None = None) -> None:
        """Initialise the step.

        Args:
            adapter: Optional :class:`~app.pipeline.adapters.astap_adapter.AstapAdapter`
                instance; created from settings if not provided.
        """
        self._adapter = adapter or AstapAdapter()

    async def execute(
        self,
        context: PipelineContext,
        config: dict[str, Any],
    ) -> StepResult:
        """Run ASTAP plate solving on the stacked FITS image.

        Args:
            context: Shared pipeline context; must have ``stacked_fits_path`` set.
            config: Profile config dict with ``plate_solving_*`` fields.

        Returns:
            :class:`~app.pipeline.base_step.StepResult` with ``ra``, ``dec``,
            and ``object_name`` in ``metadata``.
        """
        if not config.get("plate_solving_enabled", True):
            return StepResult(
                success=True, skipped=True, message="Plate solving disabled in profile."
            )

        if context.stacked_fits_path is None:
            logger.warning("plate_solving_skipped", reason="no stacked FITS in context")
            return StepResult(success=True, skipped=True, message="No stacked FITS available.")

        result = await self._adapter.solve(
            fits_path=context.stacked_fits_path,
            search_radius_deg=float(config.get("plate_solving_radius_deg", 30.0)),
            speed=str(config.get("plate_solving_speed", "auto")),
        )

        context.metadata.update(result)

        if not result.get("solved", False):
            # ASTAP returned exit 1 — no solution found.  This is not a
            # pipeline error: the job continues without WCS coordinates.
            logger.warning(
                "plate_solving_no_solution",
                fits=str(context.stacked_fits_path),
            )
            return StepResult(
                success=True,
                metadata=result,
                message=(
                    "Plate solving: no solution found "
                    "(catalog missing or insufficient stars) — pipeline continues."
                ),
            )

        logger.info("plate_solving_done", ra=result.get("ra"), dec=result.get("dec"))
        return StepResult(
            success=True,
            metadata=result,
            message=f"Solved: RA={result.get('ra'):.4f}, Dec={result.get('dec'):.4f}"
            if result.get("ra")
            else "Solved (coords unavailable)",
        )
