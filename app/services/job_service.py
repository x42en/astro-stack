"""Job (pipeline execution) business logic service."""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import ConflictException, ErrorCode, NotFoundException
from app.core.logging import get_logger
from app.domain.job import JobRead, JobStatus, JobStepRead, PipelineJob, ProfilePreset, StepStatus
from app.domain.profile import ProcessingProfileConfig, get_preset_config
from app.infrastructure.queue.broker import get_arq_pool
from app.infrastructure.repositories.job_repo import JobRepository, JobStepRepository
from app.infrastructure.repositories.session_repo import SessionRepository
from app.pipeline.orchestrator import PIPELINE_STEP_PLAN

logger = get_logger(__name__)


class JobService:
    """Manages pipeline job lifecycle: creation, enqueueing, retrieval, cancellation.

    Attributes:
        _job_repo: Repository for job records.
        _step_repo: Repository for step records.
        _session_repo: Repository for session records (for FK validation).
    """

    def __init__(self, db_session: AsyncSession) -> None:
        """Initialise the service.

        Args:
            db_session: Active database session.
        """
        self._job_repo = JobRepository(db_session)
        self._step_repo = JobStepRepository(db_session)
        self._session_repo = SessionRepository(db_session)

    async def start_pipeline(
        self,
        session_id: uuid.UUID,
        preset: ProfilePreset = ProfilePreset.STANDARD,
        profile_id: uuid.UUID | None = None,
    ) -> PipelineJob:
        """Create a pipeline job and enqueue it with ARQ.

        Validates that the session exists and is not already being processed.

        Args:
            session_id: UUID of the session to process.
            preset: Which processing preset to apply.
            profile_id: UUID of a saved advanced profile (required if preset is ADVANCED).

        Returns:
            The created :class:`~app.domain.job.PipelineJob` record.

        Raises:
            NotFoundException: If the session does not exist.
            ConflictException: If the session is already being processed.
        """
        session = await self._session_repo.get(session_id)
        if session is None:
            raise NotFoundException(
                ErrorCode.SESS_NOT_FOUND,
                f"Session '{session_id}' not found.",
            )

        active = await self._job_repo.get_active_job_for_session(session_id)
        if active is not None:
            raise ConflictException(
                ErrorCode.SESS_ALREADY_PROCESSING,
                f"Session '{session_id}' is already being processed (job {active.id}).",
                details={"active_job_id": str(active.id)},
            )

        job = PipelineJob(
            session_id=session_id,
            profile_preset=preset,
            profile_id=profile_id,
            status=JobStatus.PENDING,
        )
        # Resolve the effective profile up-front so we can both snapshot it
        # on the job row and pass it to the ARQ task.  Snapshotting at
        # creation time means later edits to the saved profile do not
        # silently rewrite history.
        profile_config = await self._resolve_profile_config(preset, profile_id)
        job.profile_snapshot = profile_config.model_dump()
        created = await self._job_repo.create(job)

        # Enqueue ARQ task
        arq_pool = await get_arq_pool()
        try:
            arq_job = await arq_pool.enqueue_job(
                "run_pipeline",
                str(created.id),
                str(session_id),
                profile_config.model_dump(),
            )
            if arq_job:
                await self._job_repo.update(created.id, {"arq_job_id": arq_job.job_id})
        finally:
            await arq_pool.aclose()

        logger.info("job_enqueued", job_id=str(created.id), preset=preset.value)
        return created

    async def get_job_with_steps(self, job_id: uuid.UUID) -> JobRead:
        """Retrieve a job with its step results, including pending future steps.

        Args:
            job_id: Job UUID.

        Returns:
            :class:`~app.domain.job.JobRead` with all planned steps, using
            ``pending`` status for steps not yet started.

        Raises:
            NotFoundException: If the job does not exist.
        """
        job = await self._job_repo.get(job_id)
        if job is None:
            raise NotFoundException(
                ErrorCode.JOB_NOT_FOUND,
                f"Job '{job_id}' not found.",
            )
        db_steps = {s.step_name: s for s in await self._step_repo.list_by_job(job_id)}

        step_reads = []
        for idx, (step_name, display_name) in enumerate(PIPELINE_STEP_PLAN):
            db_step = db_steps.get(step_name)
            if db_step:
                step_reads.append(JobStepRead(
                    id=db_step.id,
                    step_name=db_step.step_name,
                    display_name=display_name,
                    step_index=db_step.step_index,
                    status=db_step.status,
                    attempt_count=db_step.attempt_count,
                    started_at=db_step.started_at,
                    completed_at=db_step.completed_at,
                    error_code=db_step.error_code,
                    output_metadata=db_step.output_metadata,
                ))
            else:
                step_reads.append(JobStepRead(
                    step_name=step_name,
                    display_name=display_name,
                    step_index=idx,
                    status=StepStatus.PENDING,
                    attempt_count=0,
                    started_at=None,
                    completed_at=None,
                    error_code=None,
                    output_metadata=None,
                ))

        return JobRead(
            id=job.id,
            session_id=job.session_id,
            profile_preset=job.profile_preset,
            status=job.status,
            current_step=job.current_step,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error_code=job.error_code,
            output_fits_path=job.output_fits_path,
            output_tiff_path=job.output_tiff_path,
            output_preview_path=job.output_preview_path,
            profile_snapshot=job.profile_snapshot,
            created_at=job.created_at,
            steps=step_reads,
        )

    async def get_latest_job_for_session(
        self,
        session_id: uuid.UUID,
    ) -> Optional[JobRead]:
        """Return the most recent job for ``session_id`` (any status), or ``None``.

        Used by the UI to recover the rendered preview when the client-side
        session→job mapping has been lost (e.g. after a server restart or
        cleared local storage). The returned :class:`JobRead` includes the
        full step plan so the UI can drive the per-step preview browser.
        """
        jobs = await self._job_repo.list_by_session(session_id, offset=0, limit=1)
        if not jobs:
            return None
        return await self.get_job_with_steps(jobs[0].id)

    async def cancel_job(self, job_id: uuid.UUID) -> PipelineJob:
        """Request cancellation of a running job.

        Sets the ``cancelled`` flag consumed by the orchestrator. Actual
        cancellation happens gracefully at the next step boundary.

        Args:
            job_id: Job UUID to cancel.

        Returns:
            The updated job record.

        Raises:
            NotFoundException: If the job does not exist.
            ConflictException: If the job is already completed or cancelled.
        """
        job = await self._job_repo.get(job_id)
        if job is None:
            raise NotFoundException(ErrorCode.JOB_NOT_FOUND, f"Job '{job_id}' not found.")

        if job.status in (JobStatus.COMPLETED, JobStatus.CANCELLED, JobStatus.FAILED):
            raise ConflictException(
                ErrorCode.JOB_ALREADY_COMPLETED,
                f"Job '{job_id}' is already in terminal state: {job.status}.",
            )

        updated = await self._job_repo.update_status(job_id, JobStatus.CANCELLED)
        logger.info("job_cancel_requested", job_id=str(job_id))
        return updated  # type: ignore[return-value]

    async def _resolve_profile_config(
        self,
        preset: ProfilePreset,
        profile_id: uuid.UUID | None,
    ) -> ProcessingProfileConfig:
        """Load the effective processing profile configuration.

        For built-in presets, returns the hardcoded config. For ADVANCED,
        loads from the database.

        Args:
            preset: The selected preset.
            profile_id: UUID of the saved profile (required for ADVANCED).

        Returns:
            The resolved :class:`~app.domain.profile.ProcessingProfileConfig`.

        Raises:
            NotFoundException: If an ADVANCED profile is requested but not found.
        """
        if preset != ProfilePreset.ADVANCED:
            return get_preset_config(preset)

        if profile_id is None:
            return ProcessingProfileConfig()  # default advanced config

        from app.infrastructure.repositories.profile_repo import ProfileRepository  # noqa: PLC0415
        from app.core.errors import NotFoundException  # noqa: PLC0415

        # Access the underlying session directly
        repo = ProfileRepository(self._job_repo.session)
        profile = await repo.get(profile_id)
        if profile is None:
            raise NotFoundException(
                ErrorCode.PROF_NOT_FOUND,
                f"Advanced profile '{profile_id}' not found.",
            )
        return ProcessingProfileConfig(**profile.config)
