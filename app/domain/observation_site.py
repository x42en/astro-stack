"""Observation site domain model.

An :class:`ObservationSite` represents a physical location from which
astronomical observations can be conducted. Each site stores geographic
coordinates, elevation, and timezone information for observational planning.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pydantic import field_validator
from sqlalchemy import DateTime, Float, String, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlmodel import Column, Field, SQLModel

from app.core.errors import ErrorCode, ValidationException


class ObservationSite(SQLModel, table=True):
    """ORM model for user-defined observation sites.

    Maps to the ``observation_sites`` table in PostgreSQL.

    Attributes:
        id: UUID primary key.
        owner_user_id: ID of the owning user (NOT NULL, indexed).
        name: Display name of the site (1–120 chars).
        description: Optional longer description (max 500 chars).
        latitude: Geographic latitude in decimal degrees (-90..90).
        longitude: Geographic longitude in decimal degrees (-180..180).
        elevation_m: Elevation in meters above sea level (default 0.0).
        timezone: IANA timezone name (default "UTC", max 64 chars).
        created_at: Creation timestamp (server-generated).
        updated_at: Last modification timestamp (server-generated).
    """

    __tablename__ = "observation_sites"  # type: ignore[assignment]

    id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(PG_UUID(as_uuid=True), primary_key=True),
    )
    owner_user_id: uuid.UUID = Field(
        sa_column=Column(PG_UUID(as_uuid=True), nullable=False, index=True),
    )
    name: str = Field(
        min_length=1,
        max_length=120,
        sa_column=Column(String(120), nullable=False),
    )
    description: Optional[str] = Field(
        default=None,
        max_length=500,
        sa_column=Column(String(500), nullable=True),
    )
    latitude: float = Field(sa_column=Column(Float, nullable=False))
    longitude: float = Field(sa_column=Column(Float, nullable=False))
    elevation_m: float = Field(default=0.0, sa_column=Column(Float, nullable=False))
    timezone: str = Field(
        default="UTC",
        max_length=64,
        sa_column=Column(String(64), nullable=False),
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(DateTime(timezone=True), server_default=func.now()),
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column=Column(
            DateTime(timezone=True),
            server_default=func.now(),
            onupdate=func.now(),
        ),
    )


# ── Read/write schemas ────────────────────────────────────────────────────────


def _validate_latitude(v: float) -> float:
    if not isinstance(v, (int, float)):
        raise ValidationException(ErrorCode.SITE_VALIDATION_ERROR, "latitude must be a number")
    if v < -90.0 or v > 90.0:
        raise ValidationException(
            ErrorCode.SITE_VALIDATION_ERROR,
            "latitude must be between -90 and 90 degrees",
        )
    return float(v)


def _validate_longitude(v: float) -> float:
    if not isinstance(v, (int, float)):
        raise ValidationException(ErrorCode.SITE_VALIDATION_ERROR, "longitude must be a number")
    if v < -180.0 or v > 180.0:
        raise ValidationException(
            ErrorCode.SITE_VALIDATION_ERROR,
            "longitude must be between -180 and 180 degrees",
        )
    return float(v)


def _validate_elevation(v: float) -> float:
    if not isinstance(v, (int, float)):
        raise ValidationException(
            ErrorCode.SITE_VALIDATION_ERROR, "elevation_m must be a number"
        )
    if v < -500.0 or v > 9000.0:
        raise ValidationException(
            ErrorCode.SITE_VALIDATION_ERROR,
            "elevation_m must be between -500 and 9000 meters",
        )
    return float(v)


def _validate_name(v: str) -> str:
    if not isinstance(v, str):
        raise ValidationException(ErrorCode.SITE_VALIDATION_ERROR, "name must be a string")
    stripped = v.strip()
    if not stripped:
        raise ValidationException(ErrorCode.SITE_VALIDATION_ERROR, "name cannot be empty")
    if len(stripped) > 120:
        raise ValidationException(
            ErrorCode.SITE_VALIDATION_ERROR, "name must be at most 120 characters"
        )
    return stripped


def _validate_description(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    if not isinstance(v, str):
        raise ValidationException(ErrorCode.SITE_VALIDATION_ERROR, "description must be a string")
    stripped = v.strip()
    if not stripped:
        return None
    if len(stripped) > 500:
        raise ValidationException(
            ErrorCode.SITE_VALIDATION_ERROR, "description must be at most 500 characters"
        )
    return stripped


def _validate_timezone(v: str) -> str:
    if not isinstance(v, str):
        raise ValidationException(ErrorCode.SITE_VALIDATION_ERROR, "timezone must be a string")
    if not v.strip():
        raise ValidationException(ErrorCode.SITE_VALIDATION_ERROR, "timezone cannot be empty")
    if len(v) > 64:
        raise ValidationException(
            ErrorCode.SITE_VALIDATION_ERROR, "timezone must be at most 64 characters"
        )
    return v


class ObservationSiteCreate(SQLModel):
    """Schema for creating a new observation site via the REST API."""

    name: str = Field(min_length=1, max_length=120)
    description: Optional[str] = Field(default=None, max_length=500)
    latitude: float
    longitude: float
    elevation_m: float = Field(default=0.0)
    timezone: str = Field(default="UTC", max_length=64)

    @field_validator("name", mode="before")
    @classmethod
    def _v_name(cls, v: str) -> str:
        return _validate_name(v)

    @field_validator("description", mode="before")
    @classmethod
    def _v_description(cls, v: Optional[str]) -> Optional[str]:
        return _validate_description(v)

    @field_validator("latitude")
    @classmethod
    def _v_latitude(cls, v: float) -> float:
        return _validate_latitude(v)

    @field_validator("longitude")
    @classmethod
    def _v_longitude(cls, v: float) -> float:
        return _validate_longitude(v)

    @field_validator("elevation_m")
    @classmethod
    def _v_elevation(cls, v: float) -> float:
        return _validate_elevation(v)

    @field_validator("timezone", mode="before")
    @classmethod
    def _v_timezone(cls, v: str) -> str:
        return _validate_timezone(v)


class ObservationSiteUpdate(SQLModel):
    """Partial-update schema for observation sites; all fields optional."""

    name: Optional[str] = Field(default=None, min_length=1, max_length=120)
    description: Optional[str] = Field(default=None, max_length=500)
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    elevation_m: Optional[float] = None
    timezone: Optional[str] = Field(default=None, max_length=64)

    @field_validator("name", mode="before")
    @classmethod
    def _v_name(cls, v: Optional[str]) -> Optional[str]:
        return _validate_name(v) if v is not None else None

    @field_validator("description", mode="before")
    @classmethod
    def _v_description(cls, v: Optional[str]) -> Optional[str]:
        return _validate_description(v)

    @field_validator("latitude")
    @classmethod
    def _v_latitude(cls, v: Optional[float]) -> Optional[float]:
        return _validate_latitude(v) if v is not None else None

    @field_validator("longitude")
    @classmethod
    def _v_longitude(cls, v: Optional[float]) -> Optional[float]:
        return _validate_longitude(v) if v is not None else None

    @field_validator("elevation_m")
    @classmethod
    def _v_elevation(cls, v: Optional[float]) -> Optional[float]:
        return _validate_elevation(v) if v is not None else None

    @field_validator("timezone", mode="before")
    @classmethod
    def _v_timezone(cls, v: Optional[str]) -> Optional[str]:
        return _validate_timezone(v) if v is not None else None


class ObservationSiteRead(SQLModel):
    """Read schema returned by the REST API for observation site resources."""

    id: uuid.UUID
    name: str
    description: Optional[str]
    latitude: float
    longitude: float
    elevation_m: float
    timezone: str
    created_at: datetime
    updated_at: datetime
