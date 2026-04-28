"""Siril stretch and colour calibration pipeline step."""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger
from app.domain.profile import ProcessingProfileConfig
from app.domain.ws_event import LogEvent, LogLevel, LogSource
from app.infrastructure.queue.events_bus import EventBus
from app.pipeline.adapters.siril_adapter import SirilAdapter, SirilEventType
from app.pipeline.adapters.siril_script_builder import SirilScriptBuilder
from app.pipeline.base_step import PipelineContext, PipelineStep, StepResult
from app.pipeline.utils.object_type import ObjectType, resolve_object_type
from app.pipeline.utils.preview import save_step_preview

logger = get_logger(__name__)


def _fits_stats(path: Any) -> dict[str, Any] | None:
    """Return min/median/p997/max of a FITS file for diagnostic logging."""
    try:
        from pathlib import Path  # noqa: PLC0415

        import numpy as np  # noqa: PLC0415
        from astropy.io import fits as _fits  # noqa: PLC0415

        p = Path(path)
        if not p.exists():
            return None
        with _fits.open(str(p)) as hdul:
            for hdu in hdul:
                if hdu.data is None:
                    continue
                arr = np.asarray(hdu.data, dtype=np.float32)
                finite = arr[np.isfinite(arr)]
                if finite.size == 0:
                    return {"shape": list(arr.shape), "empty": True}
                return {
                    "shape": list(arr.shape),
                    "min": float(np.min(finite)),
                    "median": float(np.median(finite)),
                    "p997": float(np.percentile(finite, 99.7)),
                    "max": float(np.max(finite)),
                }
        return None
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}


# Per-object-type override of the asinh stretch.  The defaults in
# ``PRESET_STANDARD`` (asinh 150) are tuned for emission nebulae (Hα-rich,
# high surface brightness).  Galaxies and clusters have a much lower mean
# brightness and that aggressive stretch saturates the image to ~1.0,
# producing an all-black preview after the percentile clip.  Drop the
# strength selectively when the catalogue identifies the target.
_ADAPTIVE_STRETCH_BY_TYPE: dict[ObjectType, float] = {
    "galaxy": 50.0,
    "cluster": 50.0,
    "supernova": 60.0,
    "planetary": 80.0,
}


class StretchColorStep(PipelineStep):
    """Applies histogram stretch and photometric colour calibration using Siril.

    Operates on the background-removed image (or stacked image if gradient
    removal was skipped). Outputs a colour-calibrated, stretched FITS file.

    Attributes:
        event_bus: Redis pub/sub bus for forwarding Siril log events.
    """

    name = "stretch_color"
    display_name = "Stretch & Colour Calibration (Siril)"

    def __init__(self, event_bus: EventBus) -> None:
        """Initialise the step.

        Args:
            event_bus: Event bus for real-time log forwarding.
        """
        self.event_bus = event_bus

    async def execute(
        self,
        context: PipelineContext,
        config: dict[str, Any],
    ) -> StepResult:
        """Run Siril stretch and colour calibration.

        Args:
            context: Pipeline context with ``background_removed_path`` or
                ``stacked_fits_path``.
            config: Profile config dict with ``stretch_*`` and
                ``color_calibration_*`` fields.

        Returns:
            StepResult with ``stretched_fits_path`` in metadata.
        """
        input_path = context.background_removed_path or context.stacked_fits_path
        if input_path is None:
            return StepResult(
                success=True, skipped=True, message="No input FITS available for stretch."
            )

        # Copy input to Siril work dir so it knows the file location
        siril_input = context.work_dir / "output" / "for_stretch.fits"
        if input_path != siril_input:
            import shutil  # noqa: PLC0415

            shutil.copy2(str(input_path), str(siril_input))

        profile_config = ProcessingProfileConfig(**config)

        # ── Adaptive stretch override ────────────────────────────────────
        # Lookup the bundled catalogue from the (free-form) target name and
        # soften the asinh strength when the object is a galaxy / cluster /
        # planetary.  Nebulae and unidentified targets keep the profile
        # values verbatim.
        # Try the user-supplied hint first, then fall back to the
        # best-effort name returned by the plate-solver.
        object_name_hint = (
            context.metadata.get("object_name_hint")
            or context.metadata.get("object_name")
            or None
        )
        object_type = resolve_object_type(object_name_hint)
        logger.info(
            "stretch_color_object_lookup",
            object_name_hint=object_name_hint,
            resolved_type=object_type,
        )
        if object_type is not None:
            context.metadata["object_type"] = object_type
        adaptive_strength = (
            _ADAPTIVE_STRETCH_BY_TYPE.get(object_type)
            if object_type is not None
            else None
        )
        if (
            adaptive_strength is not None
            and profile_config.stretch_method == "asinh"
            and adaptive_strength < profile_config.stretch_strength
        ):
            original_strength = profile_config.stretch_strength
            profile_config = profile_config.model_copy(
                update={"stretch_strength": adaptive_strength}
            )
            context.metadata["adaptive_stretch_applied"] = {
                "object_type": object_type,
                "stretch_strength": adaptive_strength,
                "original_stretch_strength": original_strength,
            }
            logger.info(
                "stretch_color_adaptive_override",
                object_type=object_type,
                object_name=object_name_hint,
                stretch_strength=adaptive_strength,
                original_stretch_strength=original_strength,
            )
            await self.event_bus.publish_job_event(
                context.job_id,
                LogEvent(
                    job_id=context.job_id,
                    session_id=context.session_id,
                    level=LogLevel.INFO,
                    source=LogSource.SYSTEM,
                    message=(
                        f"Adaptive stretch: {object_type} detected "
                        f"({object_name_hint!r}) — asinh "
                        f"{original_strength:.0f} → {adaptive_strength:.0f}"
                    ),
                ),
            )

        builder = SirilScriptBuilder(
            config=profile_config,
            frames={},
            work_dir=context.work_dir / "output",
        )

        # Photometric Colour Calibration (PCC) — runs only if plate-solving
        # produced WCS headers in the FITS. Tolerated as best-effort: a PCC
        # failure (missing catalogue, no internet, low star count) must NOT
        # break the post-processing chain.
        wcs_solved = bool(context.metadata.get("solved", False))
        if wcs_solved and profile_config.photometric_calibration_enabled:
            try:
                async with SirilAdapter(
                    work_dir=context.work_dir / "output",
                    pipe_dir=context.work_dir / "pipes_pcc",
                ) as siril:
                    for command in builder.build_pcc_commands():
                        if context.cancelled:
                            break
                        await siril.run_command(command, timeout=180.0)
                # PCC saves back to for_stretch.fit; promote to .fits so the
                # stretch script that follows reloads the calibrated image.
                import os  # noqa: PLC0415

                pcc_out = context.work_dir / "output" / "for_stretch.fit"
                if pcc_out.exists():
                    os.replace(str(pcc_out), str(siril_input))
                logger.info("siril_pcc_done")
            except Exception as exc:  # noqa: BLE001
                logger.warning("siril_pcc_failed", error=str(exc))

        commands = builder.build_postprocessing_commands()

        if not commands:
            context.stretched_fits_path = input_path
            return StepResult(
                success=True, skipped=True, message="No stretch commands for this profile."
            )

        logger.info(
            "stretch_color_input_stats",
            stats=_fits_stats(siril_input),
            commands=commands,
        )

        async with SirilAdapter(
            work_dir=context.work_dir / "output",
            pipe_dir=context.work_dir / "pipes_stretch",
        ) as siril:
            for command in commands:
                if context.cancelled:
                    break

                events = await siril.run_command(command, timeout=300.0)
                for event in events:
                    if event.event_type == SirilEventType.LOG:
                        ws_log = LogEvent(
                            job_id=context.job_id,
                            session_id=context.session_id,
                            level=LogLevel.INFO,
                            source=LogSource.SIRIL,
                            message=event.message,
                        )
                        await self.event_bus.publish_job_event(context.job_id, ws_log)

        # Siril's `save` command writes <name>.fit (without the trailing 's').
        # Rename the output to .fits so subsequent steps can consume it correctly.
        import os  # noqa: PLC0415

        siril_out_fit = context.work_dir / "output" / "for_stretch.fit"
        if siril_out_fit.exists():
            os.replace(str(siril_out_fit), str(siril_input))

        stretched_path = context.work_dir / "output" / "for_stretch.fits"
        context.stretched_fits_path = stretched_path
        logger.info(
            "stretch_color_done",
            output=str(stretched_path),
            output_stats=_fits_stats(stretched_path),
        )

        # Generate a JPEG preview from the stretched image. Non-critical.
        preview_url: str | None = None
        try:
            preview_path = context.output_dir / "previews" / "stretch_color.jpg"
            await save_step_preview(stretched_path, preview_path)
            preview_url = f"/api/v1/sessions/{context.session_id}/step-preview/stretch_color"
        except Exception:  # noqa: BLE001
            logger.warning("stretch_color_preview_failed")

        return StepResult(
            success=True,
            metadata={
                "stretched_fits_path": str(stretched_path),
                **({"preview_url": preview_url} if preview_url else {}),
            },
            message="Stretch and colour calibration complete.",
        )
