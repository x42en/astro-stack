"""REST API endpoints for advanced processing profiles.

Endpoints
---------
* ``GET    /profiles``                 — list profiles visible to the caller
* ``POST   /profiles``                 — create a new profile (owned by caller)
* ``GET    /profiles/{id}``            — fetch one profile
* ``PUT    /profiles/{id}``            — update (owner only)
* ``DELETE /profiles/{id}``            — delete (owner only)
* ``PATCH  /profiles/{id}/share``      — toggle the ``is_shared`` flag (owner)
* ``GET    /profiles/{id}/export``     — download a portable JSON file
* ``POST   /profiles/import``          — import a previously exported file
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.middleware.auth import get_current_user
from app.core.database import get_async_session
from app.core.errors import (
    AuthException,
    ConflictException,
    ErrorCode,
    NotFoundException,
    ValidationException,
)
from app.domain.profile import (
    ProcessingProfile,
    ProcessingProfileConfig,
    ProcessingProfileCreate,
    ProcessingProfileRead,
    ProcessingProfileUpdate,
)
from app.infrastructure.repositories.profile_repo import ProfileRepository

router = APIRouter(prefix="/profiles", tags=["profiles"])


# ── Auxiliary schemas ─────────────────────────────────────────────────────────


class ProfileShareUpdate(BaseModel):
    """Body of ``PATCH /profiles/{id}/share``."""

    is_shared: bool


class ProfileExport(BaseModel):
    """Portable JSON envelope produced by ``GET /profiles/{id}/export``.

    ``kind`` and ``schema_version`` allow the importer to validate the file
    before attempting to deserialise the payload.
    """

    kind: str = Field(default="astrostack-profile")
    schema_version: int = Field(default=1)
    name: str
    description: Optional[str] = None
    config: dict[str, Any]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _current_user_id(user: Optional[dict]) -> Optional[uuid.UUID]:
    """Return the authenticated user's UUID, or ``None`` in anonymous mode."""
    if not user:
        return None
    sub = user.get("sub")
    if sub is None:
        return None
    try:
        return uuid.UUID(str(sub))
    except (ValueError, TypeError):
        return None


def _assert_owner(
    profile: ProcessingProfile,
    user_id: Optional[uuid.UUID],
) -> None:
    """Raise ``AuthException`` (403) if ``user_id`` is not the profile owner.

    In anonymous mode (``user_id is None``) the check is skipped — the deploy
    is single-tenant and every caller is implicitly trusted.
    """
    if user_id is None:
        return
    if profile.owner_user_id is not None and profile.owner_user_id != user_id:
        raise AuthException(
            ErrorCode.AUTH_INSUFFICIENT_PERMISSIONS,
            "You are not the owner of this profile.",
            status_code=403,
        )


def _to_read(
    profile: ProcessingProfile,
    user_id: Optional[uuid.UUID],
) -> ProcessingProfileRead:
    """Convert ORM → response, computing ``is_owner`` from the caller."""
    if user_id is None:
        is_owner = True
    else:
        is_owner = (
            profile.owner_user_id is None or profile.owner_user_id == user_id
        )
    return ProcessingProfileRead(
        id=profile.id,
        owner_user_id=profile.owner_user_id,
        is_shared=profile.is_shared,
        shared_at=profile.shared_at,
        is_owner=is_owner,
        name=profile.name,
        description=profile.description,
        config=profile.config,
        created_at=profile.created_at,
        updated_at=profile.updated_at,
    )


_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _slugify(value: str) -> str:
    """Best-effort slug for the export filename."""
    cleaned = _SLUG_RE.sub("-", value).strip("-")
    return cleaned or "profile"


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get(
    "",
    response_model=list[ProcessingProfileRead],
    summary="List processing profiles",
)
async def list_profiles(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    db: AsyncSession = Depends(get_async_session),
    user: Optional[dict] = Depends(get_current_user),
) -> list[ProcessingProfileRead]:
    """List profiles visible to the caller.

    Returns the caller's own profiles, profiles published via the share flag
    and unowned (system) profiles.  In anonymous mode every profile is
    returned.
    """
    repo = ProfileRepository(db)
    user_id = _current_user_id(user)
    profiles = await repo.list_visible_to(user_id, offset=offset, limit=limit)
    return [_to_read(p, user_id) for p in profiles]


@router.post(
    "",
    response_model=ProcessingProfileRead,
    status_code=201,
    summary="Create a new advanced profile",
)
async def create_profile(
    body: ProcessingProfileCreate,
    db: AsyncSession = Depends(get_async_session),
    user: Optional[dict] = Depends(get_current_user),
) -> ProcessingProfileRead:
    """Create a profile owned by the caller (or unowned in anonymous mode)."""
    repo = ProfileRepository(db)
    owner_id = _current_user_id(user)

    if owner_id is not None:
        existing = await repo.get_by_name_for_user(body.name, owner_id)
        if existing is not None:
            raise ConflictException(
                ErrorCode.PROF_NAME_CONFLICT,
                f"A profile named '{body.name}' already exists for this user.",
            )

    profile = ProcessingProfile(
        owner_user_id=owner_id,
        name=body.name,
        description=body.description,
        config=body.config.model_dump(),
    )
    created = await repo.create(profile)
    return _to_read(created, owner_id)


@router.post(
    "/import",
    response_model=ProcessingProfileRead,
    status_code=201,
    summary="Import a previously exported profile",
)
async def import_profile(
    request: Request,
    db: AsyncSession = Depends(get_async_session),
    user: Optional[dict] = Depends(get_current_user),
    file: Optional[UploadFile] = File(default=None),
) -> ProcessingProfileRead:
    """Import a profile from a multipart ``file`` upload or JSON body.

    The imported profile is always owned by the caller (or unowned in
    anonymous mode) and is never shared automatically.  Name collisions are
    resolved by appending ``" (imported)"`` / ``" (imported N)"``.
    """
    raw: bytes
    if file is not None:
        raw = await file.read()
    else:
        raw = await request.body()

    if not raw:
        raise ValidationException(
            ErrorCode.PROF_VALIDATION_ERROR,
            "Empty request body — provide a JSON profile file.",
        )

    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationException(
            ErrorCode.PROF_VALIDATION_ERROR,
            f"Invalid JSON: {exc}",
        ) from exc

    try:
        envelope = ProfileExport(**data)
    except ValidationError as exc:
        raise ValidationException(
            ErrorCode.PROF_VALIDATION_ERROR,
            f"Invalid profile envelope: {exc.errors()}",
        ) from exc

    if envelope.kind != "astrostack-profile":
        raise ValidationException(
            ErrorCode.PROF_VALIDATION_ERROR,
            f"Unsupported file kind '{envelope.kind}'.",
        )
    if envelope.schema_version != 1:
        raise ValidationException(
            ErrorCode.PROF_VALIDATION_ERROR,
            f"Unsupported schema version {envelope.schema_version}.",
        )

    try:
        config = ProcessingProfileConfig(**envelope.config)
    except ValidationError as exc:
        raise ValidationException(
            ErrorCode.PROF_VALIDATION_ERROR,
            f"Invalid profile configuration: {exc.errors()}",
        ) from exc

    repo = ProfileRepository(db)
    owner_id = _current_user_id(user)
    name = await repo.find_available_name(envelope.name, owner_id)

    profile = ProcessingProfile(
        owner_user_id=owner_id,
        name=name,
        description=envelope.description,
        config=config.model_dump(),
        is_shared=False,
    )
    created = await repo.create(profile)
    return _to_read(created, owner_id)


@router.get(
    "/{profile_id}",
    response_model=ProcessingProfileRead,
    summary="Get a profile by ID",
)
async def get_profile(
    profile_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
    user: Optional[dict] = Depends(get_current_user),
) -> ProcessingProfileRead:
    """Retrieve a profile if it is owned by the caller or shared."""
    repo = ProfileRepository(db)
    profile = await repo.get(profile_id)
    if profile is None:
        raise NotFoundException(
            ErrorCode.PROF_NOT_FOUND,
            f"Profile '{profile_id}' not found.",
        )
    user_id = _current_user_id(user)
    if (
        user_id is not None
        and profile.owner_user_id is not None
        and profile.owner_user_id != user_id
        and not profile.is_shared
    ):
        # Not shared and not owned → hide existence.
        raise NotFoundException(
            ErrorCode.PROF_NOT_FOUND,
            f"Profile '{profile_id}' not found.",
        )
    return _to_read(profile, user_id)


@router.put(
    "/{profile_id}",
    response_model=ProcessingProfileRead,
    summary="Update a profile",
)
async def update_profile(
    profile_id: uuid.UUID,
    body: ProcessingProfileUpdate,
    db: AsyncSession = Depends(get_async_session),
    user: Optional[dict] = Depends(get_current_user),
) -> ProcessingProfileRead:
    """Update a profile.  Owner-only when authentication is enabled."""
    repo = ProfileRepository(db)
    profile = await repo.get(profile_id)
    if profile is None:
        raise NotFoundException(
            ErrorCode.PROF_NOT_FOUND,
            f"Profile '{profile_id}' not found.",
        )
    user_id = _current_user_id(user)
    _assert_owner(profile, user_id)

    update_data: dict = {}
    if body.name is not None:
        update_data["name"] = body.name
    if body.description is not None:
        update_data["description"] = body.description
    if body.config is not None:
        update_data["config"] = body.config.model_dump()

    updated = await repo.update(profile_id, update_data)
    return _to_read(updated, user_id)  # type: ignore[arg-type]


@router.delete(
    "/{profile_id}",
    status_code=204,
    summary="Delete a profile",
)
async def delete_profile(
    profile_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
    user: Optional[dict] = Depends(get_current_user),
) -> None:
    """Delete a profile.  Owner-only when authentication is enabled."""
    repo = ProfileRepository(db)
    profile = await repo.get(profile_id)
    if profile is None:
        raise NotFoundException(
            ErrorCode.PROF_NOT_FOUND,
            f"Profile '{profile_id}' not found.",
        )
    _assert_owner(profile, _current_user_id(user))
    await repo.delete(profile_id)


@router.patch(
    "/{profile_id}/share",
    response_model=ProcessingProfileRead,
    summary="Toggle the share flag on a profile",
)
async def share_profile(
    profile_id: uuid.UUID,
    body: ProfileShareUpdate,
    db: AsyncSession = Depends(get_async_session),
    user: Optional[dict] = Depends(get_current_user),
) -> ProcessingProfileRead:
    """Publish or unpublish a profile.  Owner-only.

    ``shared_at`` is stamped the first time the flag is enabled and is
    preserved even when the profile is later un-shared.
    """
    repo = ProfileRepository(db)
    profile = await repo.get(profile_id)
    if profile is None:
        raise NotFoundException(
            ErrorCode.PROF_NOT_FOUND,
            f"Profile '{profile_id}' not found.",
        )
    user_id = _current_user_id(user)
    _assert_owner(profile, user_id)

    update_data: dict[str, Any] = {"is_shared": body.is_shared}
    if body.is_shared and profile.shared_at is None:
        update_data["shared_at"] = datetime.now(timezone.utc)
    updated = await repo.update(profile_id, update_data)
    return _to_read(updated, user_id)  # type: ignore[arg-type]


@router.get(
    "/{profile_id}/export",
    summary="Export a profile as a portable JSON file",
)
async def export_profile(
    profile_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
    user: Optional[dict] = Depends(get_current_user),
) -> JSONResponse:
    """Return the profile as a downloadable ``.astroprofile.json`` file."""
    repo = ProfileRepository(db)
    profile = await repo.get(profile_id)
    if profile is None:
        raise NotFoundException(
            ErrorCode.PROF_NOT_FOUND,
            f"Profile '{profile_id}' not found.",
        )
    user_id = _current_user_id(user)
    if (
        user_id is not None
        and profile.owner_user_id is not None
        and profile.owner_user_id != user_id
        and not profile.is_shared
    ):
        raise NotFoundException(
            ErrorCode.PROF_NOT_FOUND,
            f"Profile '{profile_id}' not found.",
        )

    payload = ProfileExport(
        name=profile.name,
        description=profile.description,
        config=profile.config,
    ).model_dump()
    filename = f"{_slugify(profile.name)}.astroprofile.json"
    return JSONResponse(
        content=payload,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )
