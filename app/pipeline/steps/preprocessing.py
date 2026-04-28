"""Siril pre-processing pipeline step.

Executes the full Siril calibration + stacking workflow (bias, dark, flat
masters → light calibration → registration → stacking) using the named-pipe
headless interface.

Progress events from Siril (``progress: xx%``) are forwarded to the event
bus so WebSocket clients receive real-time updates.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from app.core.errors import ErrorCode, PipelineStepException
from app.core.logging import get_logger
from app.domain.profile import ProcessingProfileConfig
from app.infrastructure.queue.events_bus import EventBus
from app.domain.ws_event import LogEvent, LogLevel, LogSource, ProgressEvent
from app.pipeline.adapters.siril_adapter import SirilAdapter, SirilEventType
from app.pipeline.adapters.siril_script_builder import SirilScriptBuilder
from app.pipeline.base_step import PipelineContext, PipelineStep, StepResult
from app.pipeline.utils.preview import save_step_preview

logger = get_logger(__name__)


class PreprocessingStep(PipelineStep):
    """Executes Siril pre-processing: calibration, alignment, and stacking.

    Uses the Siril headless pipe interface to stream progress events back
    through the event bus in real-time.

    Attributes:
        event_bus: Redis pub/sub bus for emitting WebSocket events.
    """

    name = "preprocessing"
    display_name = "Calibration & Stacking (Siril)"

    def __init__(self, event_bus: EventBus) -> None:
        """Initialise the preprocessing step.

        Args:
            event_bus: :class:`~app.infrastructure.queue.events_bus.EventBus`
                instance used to publish progress events.
        """
        self.event_bus = event_bus

    async def execute(
        self,
        context: PipelineContext,
        config: dict[str, Any],
    ) -> StepResult:
        """Run the Siril pre-processing pipeline.

        Sends commands sequentially through the named pipe. Progress
        percentages from Siril are forwarded as :class:`~app.domain.ws_event.ProgressEvent`
        events. Log lines become :class:`~app.domain.ws_event.LogEvent` events.

        Args:
            context: Pipeline context with frame paths and work directory.
            config: Profile config dict; expected to contain
                ``ProcessingProfileConfig`` fields.

        Returns:
            :class:`~app.pipeline.base_step.StepResult` with the stacked FITS path
            in ``metadata["stacked_fits_path"]``.

        Raises:
            PipelineStepException: On Siril startup failure or command error.
        """
        profile_config = ProcessingProfileConfig(**config)
        frames: dict[str, list[Path]] = context.metadata.get("frames", {})

        # Clean the process/ subdirectory before each Siril run so that stale
        # .seq files and intermediate FITS from a prior failed attempt cannot
        # corrupt the new run (affects both job-level retries and new jobs
        # launched against the same session directory).
        process_dir = context.work_dir / "process"
        shutil.rmtree(process_dir, ignore_errors=True)
        process_dir.mkdir(parents=True, exist_ok=True)

        builder = SirilScriptBuilder(
            config=profile_config,
            frames=frames,
            work_dir=context.work_dir,
        )
        commands = builder.build_preprocessing_commands()

        total_commands = len(commands)
        stacked_path = context.work_dir / "process" / "stack_result.fit"

        async with SirilAdapter(
            work_dir=context.work_dir,
            pipe_dir=context.work_dir / "pipes",
            gpu_device=context.gpu_device,
        ) as siril:
            for idx, command in enumerate(commands):
                if context.cancelled:
                    raise PipelineStepException(
                        ErrorCode.JOB_CANCEL_FAILED,
                        "Job was cancelled during preprocessing.",
                        step_name=self.name,
                        retryable=False,
                    )

                logger.debug("siril_sending_command", command=command, index=idx)
                events = await siril.run_command(command, timeout=600.0)

                # Forward Siril events to the WebSocket bus
                for event in events:
                    if event.event_type == SirilEventType.PROGRESS:
                        # Compute overall progress across all commands
                        overall_pct = (idx / total_commands) * 100.0 + (
                            event.percent / total_commands
                        )
                        ws_event = ProgressEvent(
                            job_id=context.job_id,
                            session_id=context.session_id,
                            step=self.name,
                            step_index=1,  # Preprocessing is always step index 1
                            total_steps=9,
                            percent=overall_pct,
                            message=f"[Siril] Command {idx + 1}/{total_commands}: {event.percent:.0f}%",
                        )
                        await self.event_bus.publish_job_event(context.job_id, ws_event)

                    elif event.event_type == SirilEventType.LOG:
                        ws_event_log = LogEvent(
                            job_id=context.job_id,
                            session_id=context.session_id,
                            level=LogLevel.INFO,
                            source=LogSource.SIRIL,
                            message=event.message,
                        )
                        await self.event_bus.publish_job_event(context.job_id, ws_event_log)

        if not stacked_path.exists():
            raise PipelineStepException(
                ErrorCode.PIPE_SIRIL_COMMAND_ERROR,
                f"Siril preprocessing completed but stacked FITS not found at {stacked_path}",
                step_name=self.name,
                retryable=True,
            )

        context.stacked_fits_path = stacked_path
        logger.info("preprocessing_done", stacked_fits=str(stacked_path))

        # Generate a JPEG preview from the stacked image. Non-critical.
        preview_url: str | None = None
        try:
            preview_path = context.output_dir / "previews" / "preprocessing.jpg"
            await save_step_preview(
                stacked_path,
                preview_path,
                camera_defiltered=bool(config.get("camera_defiltered", True)),
            )
            preview_url = f"/api/v1/sessions/{context.session_id}/step-preview/preprocessing"
        except Exception:  # noqa: BLE001
            logger.warning("preprocessing_preview_failed")

        return StepResult(
            success=True,
            metadata={
                "stacked_fits_path": str(stacked_path),
                **({"preview_url": preview_url} if preview_url else {}),
            },
            message=f"Stacked FITS created at {stacked_path.name}",
        )
