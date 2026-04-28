"""REST endpoints for the authenticated user's observation sites.

Authentication uses :func:`get_user_id_or_mock` so both production JWT and
the development mock-auth header (``X-Mock-User``) are accepted.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.middleware.auth import get_user_id_or_mock
from app.core.database import get_async_session
from app.core.errors import ErrorCode, NotFoundException
from app.domain.observation_site import (
    ObservationSite,
    ObservationSiteCreate,
    ObservationSiteRead,
    ObservationSiteUpdate,
)
from app.infrastructure.repositories.observation_site_repo import (
    ObservationSiteRepository,
)

router = APIRouter(prefix="/me/sites", tags=["me", "observation-sites"])


def _to_read(site: ObservationSite) -> ObservationSiteRead:
    return ObservationSiteRead(
        id=site.id,
        name=site.name,
        description=site.description,
        latitude=site.latitude,
        longitude=site.longitude,
        elevation_m=site.elevation_m,
        timezone=site.timezone,
        created_at=site.created_at,
        updated_at=site.updated_at,
    )


@router.get("", response_model=list[ObservationSiteRead])
async def list_sites(
    user_id: uuid.UUID = Depends(get_user_id_or_mock),
    session: AsyncSession = Depends(get_async_session),
) -> list[ObservationSiteRead]:
    """Return every observation site owned by the caller."""
    repo = ObservationSiteRepository(session)
    sites = await repo.list_for_user(user_id)
    return [_to_read(s) for s in sites]


@router.post("", response_model=ObservationSiteRead, status_code=status.HTTP_201_CREATED)
async def create_site(
    payload: ObservationSiteCreate,
    user_id: uuid.UUID = Depends(get_user_id_or_mock),
    session: AsyncSession = Depends(get_async_session),
) -> ObservationSiteRead:
    """Create a new observation site for the caller."""
    repo = ObservationSiteRepository(session)
    site = ObservationSite(
        owner_user_id=user_id,
        name=payload.name,
        description=payload.description,
        latitude=payload.latitude,
        longitude=payload.longitude,
        elevation_m=payload.elevation_m,
        timezone=payload.timezone,
    )
    created = await repo.create(site)
    return _to_read(created)


@router.patch("/{site_id}", response_model=ObservationSiteRead)
async def update_site(
    site_id: uuid.UUID,
    payload: ObservationSiteUpdate,
    user_id: uuid.UUID = Depends(get_user_id_or_mock),
    session: AsyncSession = Depends(get_async_session),
) -> ObservationSiteRead:
    """Update an existing observation site (owner-only)."""
    repo = ObservationSiteRepository(session)
    existing = await repo.get_for_user(site_id, user_id)
    if existing is None:
        raise NotFoundException(ErrorCode.SITE_NOT_FOUND, f"Site {site_id} not found")
    data = payload.model_dump(exclude_unset=True)
    updated = await repo.update(site_id, data)
    assert updated is not None  # existence checked above
    return _to_read(updated)


@router.delete("/{site_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_site(
    site_id: uuid.UUID,
    user_id: uuid.UUID = Depends(get_user_id_or_mock),
    session: AsyncSession = Depends(get_async_session),
) -> None:
    """Delete an observation site (owner-only)."""
    repo = ObservationSiteRepository(session)
    existing = await repo.get_for_user(site_id, user_id)
    if existing is None:
        raise NotFoundException(ErrorCode.SITE_NOT_FOUND, f"Site {site_id} not found")
    await repo.delete(site_id)
