"""Repository for user-followed catalog objects."""

from __future__ import annotations

import uuid

from sqlmodel import select

from app.domain.followed_object import FollowedObject
from app.infrastructure.repositories.base import BaseRepository


class FollowedObjectRepository(BaseRepository[FollowedObject]):
    """Async repository for followed object records."""

    model = FollowedObject

    async def list_for_user(
        self,
        owner_user_id: uuid.UUID,
        offset: int = 0,
        limit: int = 200,
    ) -> list[FollowedObject]:
        """Retrieve followed objects for a user (newest first)."""
        stmt = (
            select(FollowedObject)
            .where(FollowedObject.owner_user_id == owner_user_id)
            .order_by(FollowedObject.created_at.desc())  # type: ignore[attr-defined]
            .offset(offset)
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_for_user_by_catalog(
        self,
        owner_user_id: uuid.UUID,
        catalog_id: str,
    ) -> FollowedObject | None:
        """Retrieve a followed object by ``(owner_user_id, catalog_id)``."""
        stmt = select(FollowedObject).where(
            FollowedObject.owner_user_id == owner_user_id,
            FollowedObject.catalog_id == catalog_id,
        )
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def delete_for_user_by_catalog(
        self,
        owner_user_id: uuid.UUID,
        catalog_id: str,
    ) -> bool:
        """Delete a followed object; return True if deleted, False if missing."""
        obj = await self.get_for_user_by_catalog(owner_user_id, catalog_id)
        if obj is None:
            return False
        await self.session.delete(obj)
        await self.session.commit()
        return True
