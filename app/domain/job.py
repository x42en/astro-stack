"""Pipeline job and step domain models.

A :class:`PipelineJob` represents one execution of the processing pipeline
against a session. Each job tracks its current status and is linked to a
set of :class:`JobStep` records that capture the result of every individual
pipeline step. This granularity enables partial resumption after failure.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from sqlalchemy import DateTime, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlmodel import Column, Field, SQLModel


class JobStatus(str, Enum):
    """Lifecycle states of a pipeline job.

    Attributes:
        PENDING: Job queued, not yet started.
        RUNNING: Actively processing.
        COMPLETED: All steps succeeded.
        FAILED: One or more steps failed and max retries were exhausted.
        CANCELLED: Job was manually cancelled.
        PAUSED: Job suspended (reserved for future use).
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class StepStatus(str, Enum):
    """Execution status of a single pipeline step.

    Attributes:
        PENDING: Step not yet started.
        RUNNING: Step currently executing.
        SUCCESS: Step completed successfully.
        FAILED: Step failed (may be retried).
        SKIPPED: Step skipped based on profile configuration.
        RETRYING: Step failed and a retry attempt is scheduled.
    """

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class ProfilePreset(str, Enum):
    """Built-in processing profile presets.

    Attributes:
        QUICK: Fast processing, minimal AI, no plate solving.
        STANDARD: Balanced quality/speed with plate solving and AI denoise.
        QUALITY: Maximum quality, drizzle, full AI suite.
        ADVANCED: Fully custom parameters saved per-user.
    """

    QUICK = "quick"
    STANDARD = "standard"
    QUALITY = "quality"
    ADVANCED = "advanced"


class PipelineJob(SQLModel, table=True):
    """ORM model for a single pipeline execution run.

    Maps to the ``pipeline_jobs`` table in PostgreSQL.

    Attributes:
        id: UUID primary key.
        session_id: Foreign key to :class:`~app.domain.session.AstroSession`.
        profile_preset: Which built-in preset was selected.
        profile_id: UUID of a saved advanced profile (nullable).
        status: Current job lifecycle status.
        current_step: Name of the step currently executing.
        arq_job_id: ARQ task identifier for status polling and cancellation.
        started_at: When processing began.
        completed_at: When processing finished (success or failure).
        error_code: Error code of the definitive failure (if any).
        error_message: Human-readable error description.
        output_fits_path: Path to the final FITS output file.
        output_tiff_path: Path to the final TIFF output file.
        output_preview_path: Path to the JPEG/PNG preview file.
        created_at: Timestamp when the job was enqueued.
    """

    __tablename__ = "pipeline_jobs"  # type: ignore[assignment]

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
    )
    session_id: uuid.UUID = Field(
        sa_column=Column(PG_UUID(as_uuid=True), nullable=False, index=True)
    )
    profile_preset: str = Field(
        default=ProfilePreset.STANDARD,
        sa_column=Column(String(50), nullable=False),
    )
    profile_id: Optional[uuid.UUID] = Field(
        default=None,
        sa_column=Column(PG_UUID(as_uuid=True), nullable=True),
    )

    status: str = Field(
        default=JobStatus.PENDING,
        sa_column=Column(String(50), nullable=False, index=True),
    )
    current_step: Optional[str] = Field(default=None, max_length=100)
    arq_job_id: Optional[str] = Field(default=None, max_length=255)

    started_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )

    error_code: Optional[str] = Field(default=None, max_length=100)
    error_message: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))

    output_fits_path: Optional[str] = Field(default=None, max_length=1024)
    output_tiff_path: Optional[str] = Field(default=None, max_length=1024)
    output_preview_path: Optional[str] = Field(default=None, max_length=1024)

    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(
            DateTime(timezone=True),
            server_default=func.now(),
            nullable=False,
        ),
    )


class JobStep(SQLModel, table=True):
    """ORM model tracking the result of a single pipeline step within a job.

    One :class:`JobStep` record exists per step per job. The orchestrator
    reads these records on job resumption to determine which steps to skip.

    Maps to the ``job_steps`` table in PostgreSQL.

    Attributes:
        id: UUID primary key.
        job_id: Foreign key to :class:`PipelineJob`.
        step_name: Unique identifier for the step (e.g. ``"plate_solving"``).
        step_index: Ordinal position in the pipeline (0-based).
        status: Current step execution status.
        attempt_count: Number of execution attempts made so far.
        started_at: When the most recent attempt began.
        completed_at: When the most recent attempt finished.
        error_code: Error code of the last failure (if any).
        error_message: Human-readable error detail.
        output_metadata: Arbitrary JSON blob (e.g. plate-solve coordinates).
    """

    __tablename__ = "job_steps"  # type: ignore[assignment]

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
    )
    job_id: uuid.UUID = Field(sa_column=Column(PG_UUID(as_uuid=True), nullable=False, index=True))
    step_name: str = Field(max_length=100, index=True)
    step_index: int = Field(ge=0)
    status: str = Field(
        default=StepStatus.PENDING,
        sa_column=Column(String(50), nullable=False, index=True),
    )
    attempt_count: int = Field(default=0, ge=0)

    started_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )

    error_code: Optional[str] = Field(default=None, max_length=100)
    error_message: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    output_metadata: Optional[dict[str, Any]] = Field(
        default=None,
        sa_column=Column(JSONB, nullable=True),
    )


# ── Read/response schemas ─────────────────────────────────────────────────────


class JobStepRead(SQLModel):
    """Read schema for a single pipeline step result.

    Attributes:
        id: Step record UUID (None for steps not yet started).
        step_name: Step identifier.
        display_name: Human-readable step label.
        step_index: Ordinal position.
        status: Execution status.
        attempt_count: Number of attempts made.
        started_at: Start of most recent attempt.
        completed_at: End of most recent attempt.
        error_code: Error code if failed.
        output_metadata: Step-specific result data.
    """

    id: Optional[uuid.UUID] = None
    step_name: str
    display_name: str = ""
    step_index: int
    status: StepStatus
    attempt_count: int
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_code: Optional[str]
    output_metadata: Optional[dict[str, Any]]


class JobRead(SQLModel):
    """Read schema for a pipeline job resource.

    Attributes:
        id: Job UUID.
        session_id: Associated session UUID.
        profile_preset: Which preset was used.
        status: Current job status.
        current_step: Active step name.
        started_at: Processing start time.
        completed_at: Processing end time.
        error_code: Error code of definitive failure.
        output_fits_path: Path to final FITS file.
        output_tiff_path: Path to final TIFF file.
        output_preview_path: Path to preview image.
        created_at: Enqueue timestamp.
        steps: Ordered list of step results.
    """

    id: uuid.UUID
    session_id: uuid.UUID
    profile_preset: ProfilePreset
    status: JobStatus
    current_step: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_code: Optional[str]
    output_fits_path: Optional[str]
    output_tiff_path: Optional[str]
    output_preview_path: Optional[str]
    created_at: datetime
    steps: list[JobStepRead] = Field(default_factory=list)
