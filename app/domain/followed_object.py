"""Followed object domain model.

A :class:`FollowedObject` represents a catalog object (e.g., galaxy, nebula)
that a user is tracking for observational planning and notifications.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import Optional

from pydantic import field_validator
from sqlalchemy import Boolean, DateTime, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlmodel import Column, Field, SQLModel

from app.core.errors import ErrorCode, ValidationException

_CATALOG_ID_RE = re.compile(r"^[A-Z0-9+\-]+$")


class FollowedObject(SQLModel, table=True):
    """ORM model for user-followed catalog objects.

    Maps to the ``followed_objects`` table in PostgreSQL.

    Attributes:
        id: UUID primary key.
        owner_user_id: ID of the owning user (NOT NULL, indexed).
        catalog_id: Catalog identifier (e.g., "M31", "NGC7000", max 64 chars).
        note: Optional user note about the object (max 500 chars).
        notify_when_visible: Boolean flag for future notification system.
        created_at: Creation timestamp (server-generated).
    """

    __tablename__ = "followed_objects"  # type: ignore[assignment]

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(PG_UUID(as_uuid=True), primary_key=True),
    )
    owner_user_id: uuid.UUID = Field(
        sa_column=Column(PG_UUID(as_uuid=True), nullable=False, index=True),
    )
    catalog_id: str = Field(
        min_length=1,
        max_length=64,
        sa_column=Column(String(64), nullable=False),
    )
    note: Optional[str] = Field(
        default=None,
        max_length=500,
        sa_column=Column(String(500), nullable=True),
    )
    notify_when_visible: bool = Field(
        default=False,
        sa_column=Column(Boolean, nullable=False, server_default="false"),
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime(timezone=True), server_default=func.now()),
    )

    __table_args__ = (
        UniqueConstraint(
            "owner_user_id", "catalog_id", name="uq_followed_owner_catalog"
        ),
    )


# ── Read/write schemas ────────────────────────────────────────────────────────


class FollowedObjectCreate(SQLModel):
    """Create-payload schema for following a catalog object."""

    catalog_id: str = Field(min_length=1, max_length=64)
    note: Optional[str] = Field(default=None, max_length=500)
    notify_when_visible: bool = Field(default=False)

    @field_validator("catalog_id", mode="before")
    @classmethod
    def _v_catalog_id(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValidationException(
                ErrorCode.SITE_VALIDATION_ERROR, "catalog_id must be a string"
            )
        stripped = v.strip().upper().replace(" ", "")
        if not stripped:
            raise ValidationException(
                ErrorCode.SITE_VALIDATION_ERROR, "catalog_id cannot be empty"
            )
        if len(stripped) > 64:
            raise ValidationException(
                ErrorCode.SITE_VALIDATION_ERROR,
                "catalog_id must be at most 64 characters",
            )
        if not _CATALOG_ID_RE.match(stripped):
            raise ValidationException(
                ErrorCode.SITE_VALIDATION_ERROR,
                "catalog_id must contain only alphanumerics, +, and -",
            )
        return stripped

    @field_validator("note", mode="before")
    @classmethod
    def _v_note(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValidationException(
                ErrorCode.SITE_VALIDATION_ERROR, "note must be a string"
            )
        stripped = v.strip()
        if not stripped:
            return None
        if len(stripped) > 500:
            raise ValidationException(
                ErrorCode.SITE_VALIDATION_ERROR,
                "note must be at most 500 characters",
            )
        return stripped


class FollowedObjectRead(SQLModel):
    """Read schema for followed object resources."""

    id: uuid.UUID
    catalog_id: str
    note: Optional[str]
    notify_when_visible: bool
    created_at: datetime
