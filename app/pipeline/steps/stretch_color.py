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
from app.pipeline.utils.preview import save_step_preview

logger = get_logger(__name__)


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
        if wcs_solved and profile_config.color_calibration_enabled:
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
        logger.info("stretch_color_done", output=str(stretched_path))

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
