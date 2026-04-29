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

from typing import Any

from sqlmodel import Column, Field, SQLModel
from sqlalchemy import DateTime, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID


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


class SessionMode(str, Enum):
    """Session ingestion mode.

    Attributes:
        BATCH: Frames are uploaded en masse, then the full pipeline is run
            on demand. The historical default.
        LIVE: Frames arrive one at a time during acquisition; an incremental
            live-stacking preview is rendered after each frame. The full
            post-processing pipeline can still be triggered later on the
            collected frames.
    """

    BATCH = "batch"
    LIVE = "live"


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
    status: str = Field(
        default=SessionStatus.PENDING,
        sa_column=Column(String(50), nullable=False, index=True),
    )
    input_format: Optional[str] = Field(
        default=None,
        sa_column=Column(String(20), nullable=True),
    )

    frame_count_lights: int = Field(default=0, ge=0)
    frame_count_darks: int = Field(default=0, ge=0)
    frame_count_flats: int = Field(default=0, ge=0)
    frame_count_bias: int = Field(default=0, ge=0)
    frame_count_dark_flats: int = Field(default=0, ge=0)

    # Ingestion mode (batch upload vs. live acquisition with incremental
    # stacking).  See :class:`SessionMode` for semantics.
    mode: str = Field(
        default=SessionMode.BATCH,
        sa_column=Column(String(20), nullable=False, index=True, server_default="batch"),
    )

    # Number of frames ingested by the live stacker.  Distinct from
    # ``frame_count_lights`` so we can distinguish frames already merged in
    # the running stack from frames present on disk but not yet processed.
    live_frame_count: int = Field(default=0, ge=0)

    # Owner of the session (UUID resolved from the JWT ``sub`` claim or from
    # the ``X-Mock-User`` header in mock-auth mode).  Nullable so sessions
    # created before the ownership migration keep working unchanged.
    owner_id: Optional[uuid.UUID] = Field(
        default=None,
        sa_column=Column(PG_UUID(as_uuid=True), nullable=True, index=True),
    )

    object_name: Optional[str] = Field(default=None, max_length=255)
    ra: Optional[float] = Field(default=None)
    dec: Optional[float] = Field(default=None)

    # Capture timestamp of the earliest light frame (extracted from EXIF
    # DateTimeOriginal for RAW files, or DATE-OBS / DATE for FITS).  Falls
    # back to ``None`` when no exploitable header is found.
    acquired_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )

    # Aggregated EXIF / FITS-header capture parameters across the light
    # frames (ISO, exposure_seconds, f_number, focal_length_mm, camera_make,
    # camera_model, lens_model, telescope, filter, temperature_c, plus a
    # frame_count and total_integration_seconds).  See
    # :func:`app.pipeline.utils.exif.extract_capture_metadata` for the full
    # schema.  Stored as JSONB so the UI can render any subset.
    capture_metadata: Optional[dict[str, Any]] = Field(
        default=None,
        sa_column=Column(JSONB, nullable=True),
    )

    # User-supplied target coordinates (J2000, decimal degrees).
    # When set, ASTAP plate-solve uses them as the search centre with a small
    # radius, dramatically improving solve speed and reliability.  These are
    # NOT overwritten by the plate-solve result (which populates ra/dec).
    target_ra: Optional[float] = Field(default=None)
    target_dec: Optional[float] = Field(default=None)

    # ── Public gallery ──────────────────────────────────────────────────
    is_in_gallery: bool = Field(default=False, index=True)
    gallery_published_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )
    gallery_author_name: Optional[str] = Field(default=None, max_length=120)
    gallery_download_count: int = Field(default=0, ge=0)

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
        inbox_path: Optional explicit path under ``/inbox/``. When omitted
            (typical for live sessions) the API allocates ``/inbox/{uuid}``.
        mode: Session ingestion mode (batch / live). Defaults to ``batch``
            so existing clients keep their behaviour.
        object_name: Optional astronomical target name.
        target_ra: Optional user-supplied right ascension hint (J2000
            decimal degrees) — improves plate-solving and is shown in the UI.
        target_dec: Optional user-supplied declination hint (J2000 decimal
            degrees).
        acquired_at: Optional capture timestamp; usually pre-filled from
            ``/prepare`` when starting a live session for tonight's pick.
    """

    name: str = Field(max_length=255)
    inbox_path: Optional[str] = Field(default=None, max_length=1024)
    mode: SessionMode = SessionMode.BATCH
    object_name: Optional[str] = Field(default=None, max_length=255)
    target_ra: Optional[float] = None
    target_dec: Optional[float] = None
    acquired_at: Optional[datetime] = None


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
    mode: SessionMode = SessionMode.BATCH
    live_frame_count: int = 0
    owner_id: Optional[uuid.UUID] = None
    frame_count_lights: int
    frame_count_darks: int
    frame_count_flats: int
    frame_count_bias: int
    frame_count_dark_flats: int = 0
    object_name: Optional[str]
    ra: Optional[float]
    dec: Optional[float]
    target_ra: Optional[float]
    target_dec: Optional[float]
    acquired_at: Optional[datetime]
    capture_metadata: Optional[dict[str, Any]] = None
    is_in_gallery: bool = False
    gallery_published_at: Optional[datetime] = None
    gallery_author_name: Optional[str] = None
    gallery_download_count: int = 0
    created_at: datetime
    updated_at: datetime



class SessionUpdate(SQLModel):
    """Schema for partial session updates.

    Attributes:
        name: New human-readable name.
        status: Target status (restricted transitions enforced by the service).
        object_name: Override resolved object name.
        target_ra: User-supplied target right ascension (J2000, degrees).
        target_dec: User-supplied target declination (J2000, degrees).
    """

    name: Optional[str] = Field(default=None, max_length=255)
    status: Optional[SessionStatus] = None
    object_name: Optional[str] = Field(default=None, max_length=255)
    target_ra: Optional[float] = Field(default=None)
    target_dec: Optional[float] = Field(default=None)
