"""Pipeline orchestrator — sequences steps, handles retry and resume logic.

The orchestrator is instantiated by the ARQ worker task for each job run.
It reads existing step records to determine which steps were already completed
(enabling partial resumption), then executes pending steps in order.

Each step transition is persisted to the database and broadcast via the
Redis event bus so connected WebSocket clients receive real-time updates.

Example:
    >>> orchestrator = PipelineOrchestrator(
    ...     job_id=job_id,
    ...     session_id=session_id,
    ...     profile_config=config,
    ...     event_bus=bus,
    ...     db_session=session,
    ... )
    >>> await orchestrator.run()
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import AstroStackException, ErrorCode, PipelineStepException
from app.core.logging import get_logger
from app.domain.job import JobStatus, JobStep, PipelineJob, StepStatus
from app.domain.profile import ProcessingProfileConfig
from app.domain.ws_event import (
    CompletedEvent,
    ErrorEvent,
    ProgressEvent,
    StepStatusEvent,
    StepStatusValue,
)
from app.infrastructure.queue.events_bus import EventBus
from app.infrastructure.repositories.job_repo import JobRepository, JobStepRepository
from app.infrastructure.storage.file_store import FileStore
from app.pipeline.base_step import PipelineContext, PipelineStep, StepResult
from app.pipeline.retry import RetryPolicy

logger = get_logger(__name__)


class PipelineOrchestrator:
    """Orchestrates the sequential execution of pipeline steps for a given job.

    Manages step transitions, retry logic, and event publishing via the
    Redis pub/sub bus. Each step result is persisted to the database.

    Attributes:
        job_id: UUID of the pipeline job being executed.
        session_id: UUID of the associated astrophotography session.
        profile_config: The processing profile defining enabled steps and parameters.
        event_bus: Redis pub/sub publisher for real-time WebSocket events.
        db_session: Active async SQLAlchemy session.
        retry_policy: Retry configuration for this run.
        gpu_device: CUDA device assigned to this worker.
    """

    def __init__(
        self,
        job_id: uuid.UUID,
        session_id: uuid.UUID,
        profile_config: ProcessingProfileConfig,
        event_bus: EventBus,
        db_session: AsyncSession,
        retry_policy: RetryPolicy | None = None,
        gpu_device: str = "cuda:0",
    ) -> None:
        """Initialise the orchestrator.

        Args:
            job_id: UUID of the job to execute.
            session_id: UUID of the associated session.
            profile_config: Full processing profile configuration.
            event_bus: Event bus for WebSocket notifications.
            db_session: Database session for step persistence.
            retry_policy: Optional retry configuration; uses defaults if not provided.
            gpu_device: CUDA device string assigned to this worker process.
        """
        self.job_id = job_id
        self.session_id = session_id
        self.profile_config = profile_config
        self.event_bus = event_bus
        self.db_session = db_session
        self.retry_policy = retry_policy or RetryPolicy(max_attempts=profile_config.max_retries)
        self.gpu_device = gpu_device

        self._job_repo = JobRepository(db_session)
        self._step_repo = JobStepRepository(db_session)
        self._file_store = FileStore()

    async def run(self) -> dict[str, Any]:
        """Execute all pipeline steps in sequence, with retry and resume support.

        Loads existing step records to determine which steps have already
        succeeded (allowing partial resumption). Steps with status ``SUCCESS``
        are skipped. Failed or pending steps are (re-)executed.

        Returns:
            Dict of output paths from the final export step.

        Raises:
            AstroStackException: If a non-retryable step fails definitively.
        """
        started_at = datetime.now(tz=timezone.utc)

        # Mark job as running
        await self._job_repo.update_status(self.job_id, JobStatus.RUNNING)

        # Build context
        context = PipelineContext(
            job_id=self.job_id,
            session_id=self.session_id,
            work_dir=self._file_store.session_work_dir(self.session_id),
            output_dir=self._file_store.session_output_dir(self.session_id),
            gpu_device=self.gpu_device,
            metadata={},
        )
        self._file_store.ensure_work_dir(self.session_id)
        self._file_store.ensure_output_dir(self.session_id)

        # Load frames metadata from session
        await self._load_session_metadata(context)

        # Build ordered step list based on profile
        steps = self._build_steps()

        # Load existing step states for resume logic
        existing_steps = {s.step_name: s for s in await self._step_repo.list_by_job(self.job_id)}

        config_dict = self.profile_config.model_dump()
        final_outputs: dict[str, Any] = {}

        for step_index, step in enumerate(steps):
            if context.cancelled:
                break

            existing = existing_steps.get(step.name)

            # Skip successfully completed steps (resume support)
            if existing and existing.status == StepStatus.SUCCESS:
                logger.info(
                    "step_skipped_already_succeeded",
                    step=step.name,
                    job_id=str(self.job_id),
                )
                # Restore context metadata from persisted output
                if existing.output_metadata:
                    context.metadata.update(existing.output_metadata)
                    self._restore_context_paths(context, existing.output_metadata)
                continue

            # Execute step with retry
            result = await self._execute_with_retry(
                step=step,
                step_index=step_index,
                context=context,
                config_dict=config_dict,
            )

            if result.success:
                final_outputs.update(result.metadata)

        # Mark job completed
        duration = (datetime.now(tz=timezone.utc) - started_at).total_seconds()
        await self._job_repo.update(
            self.job_id,
            {
                "status": JobStatus.COMPLETED,
                "completed_at": datetime.now(tz=timezone.utc),
                "output_fits_path": final_outputs.get("fits_path"),
                "output_tiff_path": final_outputs.get("tiff_path"),
                "output_preview_path": final_outputs.get("jpeg_path"),
            },
        )

        completed_event = CompletedEvent(
            job_id=self.job_id,
            session_id=self.session_id,
            duration_seconds=duration,
            outputs=final_outputs,
        )
        await self.event_bus.publish_job_event(self.job_id, completed_event)
        logger.info("pipeline_completed", job_id=str(self.job_id), duration=duration)

        return final_outputs

    async def _execute_with_retry(
        self,
        step: PipelineStep,
        step_index: int,
        context: PipelineContext,
        config_dict: dict[str, Any],
    ) -> StepResult:
        """Execute a single step with retry logic.

        Persists step status transitions to the database and emits WebSocket
        events at each state change.

        Args:
            step: The pipeline step to execute.
            step_index: Zero-based position in the pipeline.
            context: Shared pipeline context.
            config_dict: Profile configuration as a plain dict.

        Returns:
            The final :class:`StepResult` (success or skipped).

        Raises:
            AstroStackException: After all retry attempts are exhausted.
        """
        total_steps = len(self._build_steps())
        last_exception: AstroStackException | None = None

        for attempt in range(1, self.retry_policy.max_attempts + 1):
            # Persist step as running
            step_record = await self._upsert_step(
                name=step.name,
                index=step_index,
                status=StepStatus.RUNNING,
                attempt=attempt,
            )

            # Notify step starting
            await self.event_bus.publish_job_event(
                self.job_id,
                StepStatusEvent(
                    job_id=self.job_id,
                    session_id=self.session_id,
                    step=step.name,
                    step_index=step_index,
                    status=StepStatusValue.STARTING,
                ),
            )
            await self._job_repo.update_status(
                self.job_id, JobStatus.RUNNING, current_step=step.name
            )

            try:
                result = await step.execute(context, config_dict)

                # Step succeeded or was skipped
                status = StepStatus.SKIPPED if result.skipped else StepStatus.SUCCESS
                await self._upsert_step(
                    name=step.name,
                    index=step_index,
                    status=status,
                    attempt=attempt,
                    output_metadata=result.metadata,
                )

                # Announce step completion
                final_status = (
                    StepStatusValue.SKIPPED if result.skipped else StepStatusValue.SUCCESS
                )
                await self.event_bus.publish_job_event(
                    self.job_id,
                    StepStatusEvent(
                        job_id=self.job_id,
                        session_id=self.session_id,
                        step=step.name,
                        step_index=step_index,
                        status=final_status,
                        result=result.metadata,
                    ),
                )

                # Emit per-step 100% progress
                await self.event_bus.publish_job_event(
                    self.job_id,
                    ProgressEvent(
                        job_id=self.job_id,
                        session_id=self.session_id,
                        step=step.name,
                        step_index=step_index,
                        total_steps=total_steps,
                        percent=((step_index + 1) / total_steps) * 100.0,
                        message=result.message or f"{step.display_name} complete.",
                    ),
                )

                return result

            except AstroStackException as exc:
                last_exception = exc
                logger.warning(
                    "step_failed",
                    step=step.name,
                    attempt=attempt,
                    error_code=exc.error_code.value,
                    message=exc.message,
                )

                error_event = ErrorEvent(
                    job_id=self.job_id,
                    session_id=self.session_id,
                    error_code=exc.error_code.value,
                    message=exc.message,
                    step=step.name,
                    retryable=exc.retryable,
                    attempt=attempt,
                    max_attempts=self.retry_policy.max_attempts,
                    details=exc.details,
                )
                await self.event_bus.publish_job_event(self.job_id, error_event)

                if not self.retry_policy.should_retry(exc.error_code, attempt):
                    break

                await self._upsert_step(
                    name=step.name,
                    index=step_index,
                    status=StepStatus.RETRYING,
                    attempt=attempt,
                )
                await self.retry_policy.wait(attempt)

        # All attempts exhausted
        await self._upsert_step(
            name=step.name,
            index=step_index,
            status=StepStatus.FAILED,
            attempt=self.retry_policy.max_attempts,
            error_code=last_exception.error_code.value if last_exception else "UNKNOWN",
            error_message=last_exception.message if last_exception else "Unknown error",
        )
        await self._job_repo.update(
            self.job_id,
            {
                "status": JobStatus.FAILED,
                "completed_at": datetime.now(tz=timezone.utc),
                "error_code": last_exception.error_code.value if last_exception else "UNKNOWN",
                "error_message": last_exception.message if last_exception else "Unknown error",
            },
        )

        await self.event_bus.publish_job_event(
            self.job_id,
            StepStatusEvent(
                job_id=self.job_id,
                session_id=self.session_id,
                step=step.name,
                step_index=step_index,
                status=StepStatusValue.ERROR,
            ),
        )

        if last_exception:
            raise last_exception
        raise AstroStackException(
            ErrorCode.SYS_INTERNAL_ERROR,
            f"Step '{step.name}' failed without a captured exception.",
        )

    def _build_steps(self) -> list[PipelineStep]:
        """Build the ordered list of pipeline steps for this run.

        Imports step classes lazily to avoid circular imports.

        Returns:
            Ordered list of :class:`~app.pipeline.base_step.PipelineStep` instances.
        """
        from app.pipeline.steps.raw_conversion import RawConversionStep  # noqa: PLC0415
        from app.pipeline.steps.preprocessing import PreprocessingStep  # noqa: PLC0415
        from app.pipeline.steps.plate_solving import PlateSolvingStep  # noqa: PLC0415
        from app.pipeline.steps.gradient_removal import GradientRemovalStep  # noqa: PLC0415
        from app.pipeline.steps.stretch_color import StretchColorStep  # noqa: PLC0415
        from app.pipeline.steps.denoise import DenoiseStep  # noqa: PLC0415
        from app.pipeline.steps.sharpen import SharpenStep  # noqa: PLC0415
        from app.pipeline.steps.super_resolution import SuperResolutionStep  # noqa: PLC0415
        from app.pipeline.steps.star_separation import StarSeparationStep  # noqa: PLC0415
        from app.pipeline.steps.export import ExportStep  # noqa: PLC0415
        from app.pipeline.adapters.cosmic_adapter import CosmicClarityAdapter  # noqa: PLC0415
        from app.pipeline.adapters.graxpert_adapter import GraXpertAdapter  # noqa: PLC0415

        cosmic_adapter = CosmicClarityAdapter(gpu_device=self.gpu_device)
        graxpert_adapter = GraXpertAdapter(gpu_device=self.gpu_device)

        return [
            RawConversionStep(),
            PreprocessingStep(event_bus=self.event_bus),
            PlateSolvingStep(),
            GradientRemovalStep(adapter=graxpert_adapter),
            StretchColorStep(event_bus=self.event_bus),
            DenoiseStep(adapter=cosmic_adapter),
            SharpenStep(adapter=cosmic_adapter),
            SuperResolutionStep(adapter=cosmic_adapter),
            StarSeparationStep(adapter=cosmic_adapter),
            ExportStep(),
        ]

    async def _upsert_step(
        self,
        name: str,
        index: int,
        status: StepStatus,
        attempt: int = 1,
        output_metadata: dict[str, Any] | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> JobStep:
        """Create or update a step record in the database.

        Args:
            name: Step identifier.
            index: Step ordinal position.
            status: Target step status.
            attempt: Current attempt number.
            output_metadata: Optional output data to persist.
            error_code: Error code string on failure.
            error_message: Error message text on failure.

        Returns:
            The persisted :class:`~app.domain.job.JobStep` record.
        """
        now = datetime.now(tz=timezone.utc)
        step = JobStep(
            job_id=self.job_id,
            step_name=name,
            step_index=index,
            status=status,
            attempt_count=attempt,
            output_metadata=output_metadata,
            error_code=error_code,
            error_message=error_message,
            started_at=now if status == StepStatus.RUNNING else None,
            completed_at=now
            if status in (StepStatus.SUCCESS, StepStatus.FAILED, StepStatus.SKIPPED)
            else None,
        )
        return await self._step_repo.upsert_step(step)

    async def _load_session_metadata(self, context: PipelineContext) -> None:
        """Load session frame inventory into the pipeline context.

        Reads the inbox path for the session and populates ``context.metadata``
        with frame lists and input format.

        Args:
            context: Pipeline context to populate.
        """
        from sqlmodel import select  # noqa: PLC0415
        from app.domain.session import AstroSession  # noqa: PLC0415

        result = await self.db_session.execute(
            select(AstroSession).where(AstroSession.id == self.session_id)
        )
        session_record = result.first()
        if session_record is None:
            return

        inbox_path = __import__("pathlib").Path(session_record.inbox_path)
        frames = self._file_store.discover_frames(inbox_path)
        input_format = self._file_store.detect_input_format(frames)

        context.metadata["frames"] = frames
        context.metadata["input_format"] = input_format

    @staticmethod
    def _restore_context_paths(
        context: PipelineContext,
        metadata: dict[str, Any],
    ) -> None:
        """Restore file path attributes on the context from persisted step metadata.

        Called when resuming from a previously succeeded step.

        Args:
            context: Pipeline context to update.
            metadata: Step output metadata dict.
        """
        from pathlib import Path  # noqa: PLC0415

        path_mapping = {
            "stacked_fits_path": "stacked_fits_path",
            "background_removed_path": "background_removed_path",
            "denoised_path": "denoised_path",
            "sharpened_path": "sharpened_path",
            "superres_path": "superres_path",
            "nebula_only_path": "nebula_only_path",
            "final_fits_path": "final_fits_path",
        }
        for meta_key, ctx_attr in path_mapping.items():
            if meta_key in metadata and metadata[meta_key]:
                setattr(context, ctx_attr, Path(metadata[meta_key]))
