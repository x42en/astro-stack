"""Repository for user-defined observation sites."""

from __future__ import annotations

import uuid

from sqlmodel import select

from app.domain.observation_site import ObservationSite
from app.infrastructure.repositories.base import BaseRepository


class ObservationSiteRepository(BaseRepository[ObservationSite]):
    """Async repository for observation site records."""

    model = ObservationSite

    async def list_for_user(
        self,
        owner_user_id: uuid.UUID,
        offset: int = 0,
        limit: int = 100,
    ) -> list[ObservationSite]:
        """Retrieve all sites owned by ``owner_user_id`` (sorted by name)."""
        stmt = (
            select(ObservationSite)
            .where(ObservationSite.owner_user_id == owner_user_id)
            .order_by(ObservationSite.name)  # type: ignore[arg-type]
            .offset(offset)
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_for_user(
        self,
        record_id: uuid.UUID,
        owner_user_id: uuid.UUID,
    ) -> ObservationSite | None:
        """Retrieve a site by ID, returning ``None`` if missing or not owned."""
        stmt = select(ObservationSite).where(
            ObservationSite.id == record_id,
            ObservationSite.owner_user_id == owner_user_id,
        )
        result = await self.session.execute(stmt)
        return result.scalars().first()
