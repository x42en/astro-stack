"""REST API endpoints for advanced processing profiles."""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.middleware.auth import get_current_user
from app.core.database import get_async_session
from app.core.errors import ConflictException, ErrorCode, NotFoundException
from app.domain.profile import (
    ProcessingProfile,
    ProcessingProfileCreate,
    ProcessingProfileRead,
    ProcessingProfileUpdate,
)
from app.infrastructure.repositories.profile_repo import ProfileRepository

router = APIRouter(prefix="/profiles", tags=["profiles"])


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
    """List all accessible processing profiles for the current user.

    Returns shared profiles (no owner) and profiles owned by the current user.
    When auth is disabled, all profiles are returned.

    Args:
        offset: Pagination offset.
        limit: Maximum results.
        db: Injected database session.
        user: Injected auth user (None if auth disabled).

    Returns:
        List of :class:`~app.domain.profile.ProcessingProfileRead` objects.
    """
    repo = ProfileRepository(db)
    if user:
        user_id = uuid.UUID(str(user.get("sub", uuid.uuid4())))
        profiles = await repo.list_for_user(
            user_id, include_shared=True, offset=offset, limit=limit
        )
    else:
        profiles = await repo.list_all(offset=offset, limit=limit)
    return [_to_read(p) for p in profiles]


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
    """Create and persist a new user-owned advanced processing profile.

    Args:
        body: Profile creation payload.
        db: Injected database session.
        user: Injected auth user.

    Returns:
        The created :class:`~app.domain.profile.ProcessingProfileRead`.
    """
    repo = ProfileRepository(db)
    owner_id = uuid.UUID(str(user["sub"])) if user else None

    # Check uniqueness per user
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
    return _to_read(created)


@router.get(
    "/{profile_id}",
    response_model=ProcessingProfileRead,
    summary="Get a profile by ID",
)
async def get_profile(
    profile_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
    _user: Optional[dict] = Depends(get_current_user),
) -> ProcessingProfileRead:
    """Retrieve a specific processing profile.

    Args:
        profile_id: Profile UUID.
        db: Injected database session.
        _user: Injected auth user.

    Returns:
        :class:`~app.domain.profile.ProcessingProfileRead` for the found profile.
    """
    repo = ProfileRepository(db)
    profile = await repo.get(profile_id)
    if profile is None:
        raise NotFoundException(
            ErrorCode.PROF_NOT_FOUND,
            f"Profile '{profile_id}' not found.",
        )
    return _to_read(profile)


@router.put(
    "/{profile_id}",
    response_model=ProcessingProfileRead,
    summary="Update a profile",
)
async def update_profile(
    profile_id: uuid.UUID,
    body: ProcessingProfileUpdate,
    db: AsyncSession = Depends(get_async_session),
    _user: Optional[dict] = Depends(get_current_user),
) -> ProcessingProfileRead:
    """Replace profile fields with new values.

    Args:
        profile_id: Profile UUID to update.
        body: Fields to update (name, description, config).
        db: Injected database session.
        _user: Injected auth user.

    Returns:
        Updated :class:`~app.domain.profile.ProcessingProfileRead`.
    """
    repo = ProfileRepository(db)
    profile = await repo.get(profile_id)
    if profile is None:
        raise NotFoundException(
            ErrorCode.PROF_NOT_FOUND,
            f"Profile '{profile_id}' not found.",
        )

    update_data: dict = {}
    if body.name is not None:
        update_data["name"] = body.name
    if body.description is not None:
        update_data["description"] = body.description
    if body.config is not None:
        update_data["config"] = body.config.model_dump()

    updated = await repo.update(profile_id, update_data)
    return _to_read(updated)  # type: ignore[arg-type]


@router.delete(
    "/{profile_id}",
    status_code=204,
    summary="Delete a profile",
)
async def delete_profile(
    profile_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
    _user: Optional[dict] = Depends(get_current_user),
) -> None:
    """Delete a processing profile.

    Args:
        profile_id: Profile UUID to delete.
        db: Injected database session.
        _user: Injected auth user.
    """
    repo = ProfileRepository(db)
    deleted = await repo.delete(profile_id)
    if not deleted:
        raise NotFoundException(
            ErrorCode.PROF_NOT_FOUND,
            f"Profile '{profile_id}' not found.",
        )


# ── Helper ─────────────────────────────────────────────────────────────────────


def _to_read(profile: ProcessingProfile) -> ProcessingProfileRead:
    """Convert an ORM profile instance to the read schema.

    Args:
        profile: :class:`~app.domain.profile.ProcessingProfile` ORM instance.

    Returns:
        :class:`~app.domain.profile.ProcessingProfileRead` response schema.
    """
    return ProcessingProfileRead(
        id=profile.id,
        name=profile.name,
        description=profile.description,
        config=profile.config,
        created_at=profile.created_at,
        updated_at=profile.updated_at,
    )
