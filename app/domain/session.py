"""Session domain model and related enumerations.

A :class:`AstroSession` represents a single astrophotography imaging session
that has been deposited in the inbox directory. It groups calibration frames
(darks, flats, bias) with light frames (brutes) and tracks the overall
lifecycle from detection through to completed processing.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from sqlmodel import Column, Field, SQLModel
from sqlalchemy import DateTime, String, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID


class SessionStatus(str, Enum):
    """Lifecycle states of an astrophotography session.

    Attributes:
        PENDING: Detected in inbox, not yet validated.
        READY: Frame inventory complete, ready to queue.
        PROCESSING: Pipeline actively running.
        COMPLETED: All pipeline steps succeeded.
        FAILED: One or more pipeline steps failed definitively.
        CANCELLED: Processing was manually cancelled.
    """

    PENDING = "pending"
    READY = "ready"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class InputFormat(str, Enum):
    """Source image format detected for this session.

    Attributes:
        FITS: Native astronomical FITS format (.fit, .fits).
        RAW_DSLR: Camera RAW format (.cr2, .nef, .arw, .dng, etc.).
        MIXED: Mix of FITS and RAW files (unusual but tolerated).
    """

    FITS = "fits"
    RAW_DSLR = "raw_dslr"
    MIXED = "mixed"


class AstroSession(SQLModel, table=True):
    """ORM model representing an astrophotography imaging session.

    Maps to the ``astro_sessions`` table in PostgreSQL.

    Attributes:
        id: UUID primary key, generated automatically.
        name: Human-readable name, typically derived from the inbox path.
        inbox_path: Absolute path of the session folder inside ``/inbox/``.
        status: Current lifecycle status (see :class:`SessionStatus`).
        input_format: Auto-detected source file format.
        frame_count_lights: Number of light (science) frames found.
        frame_count_darks: Number of dark calibration frames found.
        frame_count_flats: Number of flat calibration frames found.
        frame_count_bias: Number of bias frames found.
        object_name: Astronomical object name resolved by plate solving.
        ra: Right ascension in decimal degrees (from plate solving).
        dec: Declination in decimal degrees (from plate solving).
        created_at: UTC timestamp of session discovery.
        updated_at: UTC timestamp of last status change.
    """

    __tablename__ = "astro_sessions"  # type: ignore[assignment]

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
    )
    name: str = Field(max_length=255, index=True)
    inbox_path: str = Field(max_length=1024)
    status: SessionStatus = Field(
        default=SessionStatus.PENDING,
        sa_column=Column(String(50), nullable=False, index=True),
    )
    input_format: Optional[InputFormat] = Field(
        default=None,
        sa_column=Column(String(20), nullable=True),
    )

    frame_count_lights: int = Field(default=0, ge=0)
    frame_count_darks: int = Field(default=0, ge=0)
    frame_count_flats: int = Field(default=0, ge=0)
    frame_count_bias: int = Field(default=0, ge=0)

    object_name: Optional[str] = Field(default=None, max_length=255)
    ra: Optional[float] = Field(default=None)
    dec: Optional[float] = Field(default=None)

    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(
            DateTime(timezone=True),
            server_default=func.now(),
            nullable=False,
        ),
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(
            DateTime(timezone=True),
            server_default=func.now(),
            onupdate=func.now(),
            nullable=False,
        ),
    )


class SessionCreate(SQLModel):
    """Schema for creating a session via the REST API (manual creation).

    Attributes:
        name: Human-readable session name.
        inbox_path: Path under ``/inbox/`` for this session.
    """

    name: str = Field(max_length=255)
    inbox_path: str = Field(max_length=1024)


class SessionRead(SQLModel):
    """Read schema returned by the REST API for session resources.

    Attributes:
        id: Session UUID.
        name: Human-readable name.
        inbox_path: Source directory path.
        status: Current lifecycle status.
        input_format: Detected file format.
        frame_count_lights: Number of light frames.
        frame_count_darks: Number of dark frames.
        frame_count_flats: Number of flat frames.
        frame_count_bias: Number of bias frames.
        object_name: Resolved object name (post plate-solve).
        ra: Right ascension.
        dec: Declination.
        created_at: Discovery timestamp.
        updated_at: Last update timestamp.
    """

    id: uuid.UUID
    name: str
    inbox_path: str
    status: SessionStatus
    input_format: Optional[InputFormat]
    frame_count_lights: int
    frame_count_darks: int
    frame_count_flats: int
    frame_count_bias: int
    object_name: Optional[str]
    ra: Optional[float]
    dec: Optional[float]
    created_at: datetime
    updated_at: datetime


class SessionUpdate(SQLModel):
    """Schema for partial session updates.

    Attributes:
        name: New human-readable name.
        status: Target status (restricted transitions enforced by the service).
        object_name: Override resolved object name.
    """

    name: Optional[str] = Field(default=None, max_length=255)
    status: Optional[SessionStatus] = None
    object_name: Optional[str] = Field(default=None, max_length=255)
