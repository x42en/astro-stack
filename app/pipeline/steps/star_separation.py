"""Dark Star star-separation pipeline step."""

from __future__ import annotations

import shutil
from typing import Any

from app.core.logging import get_logger
from app.pipeline.adapters.cosmic_adapter import CosmicClarityAdapter
from app.pipeline.base_step import PipelineContext, PipelineStep, StepResult

logger = get_logger(__name__)


class StarSeparationStep(PipelineStep):
    """Removes stars to isolate nebula, then optionally recombines the layers.

    Uses Cosmic Clarity Dark Star to produce a star-free nebula image.
    If recombination is enabled, blends the nebula and original using
    configurable weighting — useful for preserving star colour/shape
    from the sharpened image while benefiting from cleaner nebula rendering.
    """

    name = "star_separation"
    display_name = "Star Separation (Cosmic Clarity Dark Star)"

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
        """Run Dark Star star removal and optional recombination.

        Args:
            context: Pipeline context.
            config: Profile config dict with ``star_separation_*`` fields.

        Returns:
            StepResult with ``nebula_only_path`` and optionally ``final_fits_path``.
        """
        if not config.get("star_separation_enabled", False):
            # Propagate final image from previous step
            final = context.superres_path or context.sharpened_path or context.denoised_path
            if final:
                context.final_fits_path = final
            return StepResult(success=True, skipped=True, message="Star separation disabled.")

        input_path = (
            context.superres_path
            or context.sharpened_path
            or context.denoised_path
            or context.stacked_fits_path
        )
        if input_path is None:
            return StepResult(
                success=True, skipped=True, message="No input FITS for star separation."
            )

        nebula_path = context.work_dir / "output" / "nebula_only.fits"
        nebula_path.parent.mkdir(parents=True, exist_ok=True)

        self._adapter.gpu_device = context.gpu_device

        await self._adapter.remove_stars(
            input_path=input_path,
            output_path=nebula_path,
        )

        context.nebula_only_path = nebula_path

        if config.get("star_separation_recombine", True):
            recombined = await _recombine(
                nebula_path=nebula_path,
                stars_path=input_path,
                output_dir=context.work_dir / "output",
                nebula_weight=float(config.get("star_separation_nebula_weight", 0.8)),
                star_weight=float(config.get("star_separation_star_weight", 0.5)),
            )
            context.final_fits_path = recombined
            logger.info("star_separation_recombined", output=str(recombined))
        else:
            context.final_fits_path = nebula_path

        return StepResult(
            success=True,
            metadata={
                "nebula_only_path": str(nebula_path),
                "final_fits_path": str(context.final_fits_path),
            },
            message="Star separation complete.",
        )


async def _recombine(
    nebula_path: "Path",
    stars_path: "Path",
    output_dir: "Path",
    nebula_weight: float,
    star_weight: float,
) -> "Path":
    """Blend the nebula-only and original (stars) images.

    Uses astropy to read both FITS arrays and compute a weighted sum,
    then writes the result as a new FITS file.

    Args:
        nebula_path: Path to the star-removed (nebula-only) image.
        stars_path: Path to the original image (with stars).
        output_dir: Directory for the recombined output file.
        nebula_weight: Weight applied to the nebula layer (0.0–1.0).
        star_weight: Weight applied to the stars layer (0.0–1.0).

    Returns:
        Path to the recombined FITS file.
    """
    from pathlib import Path  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from astropy.io import fits  # noqa: PLC0415

    with fits.open(str(nebula_path)) as hdul:
        nebula_data = hdul[0].data.astype(np.float32)
        header = hdul[0].header.copy()

    with fits.open(str(stars_path)) as hdul:
        stars_data = hdul[0].data.astype(np.float32)

    combined = np.clip(nebula_data * nebula_weight + stars_data * star_weight, 0, None)
    combined = (combined / combined.max() * 65535).astype(np.float32)

    output_path = output_dir / "recombined.fits"
    hdu = fits.PrimaryHDU(data=combined, header=header)
    hdu.writeto(str(output_path), overwrite=True)
    return output_path
