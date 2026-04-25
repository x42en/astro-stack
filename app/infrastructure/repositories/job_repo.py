"""Repository for :class:`~app.domain.job.PipelineJob` and :class:`~app.domain.job.JobStep`.

Provides job-specific query methods needed by the orchestrator and the
API layer to check and update pipeline execution state.
"""

from __future__ import annotations

import uuid
from typing import Optional

from sqlmodel import select

from app.domain.job import JobStatus, JobStep, PipelineJob, StepStatus
from app.infrastructure.repositories.base import BaseRepository


class JobRepository(BaseRepository[PipelineJob]):
    """Async repository for pipeline job records.

    Extends :class:`BaseRepository` with job-specific queries.
    """

    model = PipelineJob

    async def list_by_session(
        self,
        session_id: uuid.UUID,
        offset: int = 0,
        limit: int = 50,
    ) -> list[PipelineJob]:
        """Retrieve all jobs associated with a given session.

        Args:
            session_id: UUID of the parent session.
            offset: Pagination offset.
            limit: Maximum number of results.

        Returns:
            List of jobs ordered by creation time descending.
        """
        stmt = (
            select(PipelineJob)
            .where(PipelineJob.session_id == session_id)
            .order_by(PipelineJob.created_at.desc())  # type: ignore[attr-defined]
            .offset(offset)
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_active_job_for_session(
        self,
        session_id: uuid.UUID,
    ) -> Optional[PipelineJob]:
        """Return the currently running or pending job for a session, if any.

        Args:
            session_id: UUID of the session to check.

        Returns:
            The active job, or ``None`` if none is running.
        """
        stmt = select(PipelineJob).where(
            PipelineJob.session_id == session_id,
            PipelineJob.status.in_([JobStatus.PENDING.value, JobStatus.RUNNING.value]),  # type: ignore[attr-defined]
        )
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def update_status(
        self,
        job_id: uuid.UUID,
        status: JobStatus,
        current_step: Optional[str] = None,
    ) -> Optional[PipelineJob]:
        """Update job status and optionally the current step name.

        Args:
            job_id: UUID of the job to update.
            status: New :class:`~app.domain.job.JobStatus`.
            current_step: Name of the step now executing (optional).

        Returns:
            The updated job, or ``None`` if not found.
        """
        data: dict = {"status": status}
        if current_step is not None:
            data["current_step"] = current_step
        return await self.update(job_id, data)


class JobStepRepository(BaseRepository[JobStep]):
    """Async repository for individual pipeline step result records.

    Used by the orchestrator to persist step transitions and by resume logic
    to determine which steps can be skipped.
    """

    model = JobStep

    async def list_by_job(self, job_id: uuid.UUID) -> list[JobStep]:
        """Retrieve all step records for a job, ordered by step_index.

        Args:
            job_id: UUID of the parent job.

        Returns:
            Ordered list of step records.
        """
        stmt = (
            select(JobStep).where(JobStep.job_id == job_id).order_by(JobStep.step_index)  # type: ignore[attr-defined]
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_by_job_and_name(
        self,
        job_id: uuid.UUID,
        step_name: str,
    ) -> Optional[JobStep]:
        """Retrieve a specific step record by job and step name.

        Args:
            job_id: UUID of the parent job.
            step_name: Unique step identifier string.

        Returns:
            The step record, or ``None`` if not found.
        """
        stmt = select(JobStep).where(
            JobStep.job_id == job_id,
            JobStep.step_name == step_name,
        )
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def upsert_step(self, step: JobStep) -> JobStep:
        """Insert or update a step record.

        If a step with the same ``job_id`` + ``step_name`` already exists,
        it is updated in-place. Otherwise a new record is created.

        Args:
            step: The step record to persist.

        Returns:
            The saved (and refreshed) step record.
        """
        existing = await self.get_by_job_and_name(step.job_id, step.step_name)
        if existing is not None:
            for field in step.model_fields:
                if field not in ("id",):
                    setattr(existing, field, getattr(step, field))
            self.session.add(existing)
            await self.session.commit()
            await self.session.refresh(existing)
            return existing
        return await self.create(step)

    async def update_step_status(
        self,
        job_id: uuid.UUID,
        step_name: str,
        status: StepStatus,
        attempt_count: Optional[int] = None,
    ) -> Optional[JobStep]:
        """Transition a step to a new status.

        Args:
            job_id: UUID of the parent job.
            step_name: Unique step identifier.
            status: Target :class:`~app.domain.job.StepStatus`.
            attempt_count: Updated attempt count (optional).

        Returns:
            The updated step, or ``None`` if not found.
        """
        existing = await self.get_by_job_and_name(job_id, step_name)
        if existing is None:
            return None
        existing.status = status
        if attempt_count is not None:
            existing.attempt_count = attempt_count
        self.session.add(existing)
        await self.session.commit()
        await self.session.refresh(existing)
        return existing
