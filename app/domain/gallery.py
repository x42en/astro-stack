"""Persistence model for public gallery download requests.

Each row records a user-supplied email + chosen format (TIFF or FITS) for
a published gallery session, together with the IP that requested the
download.  The table is purely an audit/log: it is *not* used to gate or
authorise downloads (those are handled by short-lived HMAC tokens).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import Column, DateTime, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlmodel import Field, SQLModel


class GalleryDownload(SQLModel, table=True):
    """Audit row for a single gallery download request."""

    __tablename__ = "astro_gallery_downloads"  # type: ignore[assignment]

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
    )
    session_id: uuid.UUID = Field(
        sa_column=Column(
            PG_UUID(as_uuid=True),
            ForeignKey("astro_sessions.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
    )
    email: str = Field(max_length=320)
    format: str = Field(sa_column=Column(String(10), nullable=False))
    requester_ip: Optional[str] = Field(
        default=None,
        sa_column=Column(String(45), nullable=True),
    )
    requested_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(
            DateTime(timezone=True),
            server_default=func.now(),
            nullable=False,
        ),
    )
