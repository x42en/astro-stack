"""REST endpoints for the authenticated user's followed catalog objects.

Each ``/me/followed-objects`` resource is enriched with the resolved
``CatalogObject`` (when one exists in the bundled registry) so the front-end
can render rich cards without a second round-trip.
"""

from __future__ import annotations

import uuid
from datetime import date as date_type
from typing import Optional

from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.middleware.auth import get_user_id_or_mock
from app.core.database import get_async_session
from app.core.errors import ConflictException, ErrorCode, NotFoundException
from app.domain.followed_object import (
    FollowedObject,
    FollowedObjectCreate,
)
from app.domain.visibility import ObjectVisibility
from app.infrastructure.catalog.messier import CatalogObject
from app.infrastructure.catalog.registry import lookup_object
from app.infrastructure.repositories.followed_object_repo import (
    FollowedObjectRepository,
)
from app.services.planner_service import PlannerService

router = APIRouter(prefix="/me/followed-objects", tags=["me", "followed-objects"])


class FollowedObjectReadEnriched(BaseModel):
    """Wire-format for followed objects, with the resolved catalog entry."""

    id: uuid.UUID
    catalog_id: str
    note: Optional[str]
    notify_when_visible: bool
    created_at: str
    catalog_object: Optional[CatalogObject] = None


def _to_read(obj: FollowedObject) -> FollowedObjectReadEnriched:
    catalog = lookup_object(obj.catalog_id)
    return FollowedObjectReadEnriched(
        id=obj.id,
        catalog_id=obj.catalog_id,
        note=obj.note,
        notify_when_visible=obj.notify_when_visible,
        created_at=obj.created_at.isoformat(),
        catalog_object=catalog,
    )


@router.get("", response_model=list[FollowedObjectReadEnriched])
async def list_followed(
    user_id: uuid.UUID = Depends(get_user_id_or_mock),
    session: AsyncSession = Depends(get_async_session),
) -> list[FollowedObjectReadEnriched]:
    """Return every catalog object followed by the caller."""
    repo = FollowedObjectRepository(session)
    items = await repo.list_for_user(user_id)
    return [_to_read(o) for o in items]


@router.post(
    "", response_model=FollowedObjectReadEnriched, status_code=status.HTTP_201_CREATED
)
async def follow(
    payload: FollowedObjectCreate,
    user_id: uuid.UUID = Depends(get_user_id_or_mock),
    session: AsyncSession = Depends(get_async_session),
) -> FollowedObjectReadEnriched:
    """Subscribe the caller to a catalog object."""
    if lookup_object(payload.catalog_id) is None:
        raise NotFoundException(
            ErrorCode.FOLLOW_OBJECT_UNKNOWN,
            f"Unknown catalog object: {payload.catalog_id}",
        )
    repo = FollowedObjectRepository(session)
    existing = await repo.get_for_user_by_catalog(user_id, payload.catalog_id)
    if existing is not None:
        raise ConflictException(
            ErrorCode.FOLLOW_ALREADY_EXISTS,
            f"Object {payload.catalog_id} is already followed",
        )
    obj = FollowedObject(
        owner_user_id=user_id,
        catalog_id=payload.catalog_id,
        note=payload.note,
        notify_when_visible=payload.notify_when_visible,
    )
    created = await repo.create(obj)
    return _to_read(created)


@router.delete("/{catalog_id}", status_code=status.HTTP_204_NO_CONTENT)
async def unfollow(
    catalog_id: str,
    user_id: uuid.UUID = Depends(get_user_id_or_mock),
    session: AsyncSession = Depends(get_async_session),
) -> None:
    """Remove a catalog object from the caller's followed list."""
    normalized = catalog_id.strip().upper().replace(" ", "")
    repo = FollowedObjectRepository(session)
    deleted = await repo.delete_for_user_by_catalog(user_id, normalized)
    if not deleted:
        raise NotFoundException(
            ErrorCode.FOLLOW_NOT_FOUND, f"Object {normalized} is not followed"
        )


@router.get("/{catalog_id}/visibility", response_model=ObjectVisibility)
async def visibility(
    catalog_id: str,
    lat: float = Query(..., ge=-90.0, le=90.0),
    lon: float = Query(..., ge=-180.0, le=180.0),
    elevation_m: float = Query(0.0, ge=-500.0, le=9000.0),
    date: date_type = Query(...),
    user_id: uuid.UUID = Depends(get_user_id_or_mock),
) -> ObjectVisibility:
    """Compute visibility for a single followed object on a given night."""
    planner = PlannerService()
    window = await planner.night_window(lat, lon, elevation_m, date)
    return await planner.visibility_for_object(lat, lon, elevation_m, window, catalog_id)
