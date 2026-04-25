"""Repository for user-saved advanced processing profiles.

Handles persistence of :class:`~app.domain.profile.ProcessingProfile` records
in PostgreSQL, including per-user ownership queries.
"""

from __future__ import annotations

import uuid
from typing import Optional

from sqlmodel import select

from app.domain.profile import ProcessingProfile
from app.infrastructure.repositories.base import BaseRepository


class ProfileRepository(BaseRepository[ProcessingProfile]):
    """Async repository for advanced processing profile records.

    Extends :class:`BaseRepository` with ownership-aware queries.
    """

    model = ProcessingProfile

    async def list_for_user(
        self,
        owner_user_id: uuid.UUID,
        include_shared: bool = True,
        offset: int = 0,
        limit: int = 100,
    ) -> list[ProcessingProfile]:
        """Retrieve profiles owned by a specific user, optionally including shared ones.

        Shared profiles have ``owner_user_id = NULL`` and are accessible to all users.

        Args:
            owner_user_id: UUID of the requesting user.
            include_shared: If ``True``, also return profiles with no owner.
            offset: Pagination offset.
            limit: Maximum number of results.

        Returns:
            List of profiles, ordered by name.
        """
        if include_shared:
            stmt = (
                select(ProcessingProfile)
                .where(
                    (ProcessingProfile.owner_user_id == owner_user_id)
                    | (ProcessingProfile.owner_user_id.is_(None))  # type: ignore[union-attr]
                )
                .order_by(ProcessingProfile.name)  # type: ignore[attr-defined]
                .offset(offset)
                .limit(limit)
            )
        else:
            stmt = (
                select(ProcessingProfile)
                .where(ProcessingProfile.owner_user_id == owner_user_id)
                .order_by(ProcessingProfile.name)  # type: ignore[attr-defined]
                .offset(offset)
                .limit(limit)
            )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_by_name_for_user(
        self,
        name: str,
        owner_user_id: uuid.UUID,
    ) -> Optional[ProcessingProfile]:
        """Retrieve a profile by name for a specific user.

        Used for duplicate name detection before creating a new profile.

        Args:
            name: Exact profile name to search.
            owner_user_id: UUID of the owning user.

        Returns:
            The matching profile, or ``None`` if not found.
        """
        stmt = select(ProcessingProfile).where(
            ProcessingProfile.name == name,
            ProcessingProfile.owner_user_id == owner_user_id,
        )
        result = await self.session.execute(stmt)
        return result.scalars().first()
