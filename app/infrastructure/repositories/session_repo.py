"""Repository for :class:`~app.domain.session.AstroSession` persistence.

Provides session-specific query methods in addition to the generic CRUD
operations inherited from :class:`~app.infrastructure.repositories.base.BaseRepository`.
"""

from __future__ import annotations

import uuid
from typing import Optional

from sqlmodel import select

from app.domain.session import AstroSession, SessionStatus
from app.infrastructure.repositories.base import BaseRepository


class SessionRepository(BaseRepository[AstroSession]):
    """Async repository for astrophotography session records.

    Extends :class:`BaseRepository` with session-specific queries such as
    filtering by status or inbox path.
    """

    model = AstroSession

    async def get_by_inbox_path(self, inbox_path: str) -> Optional[AstroSession]:
        """Find a session by its exact inbox directory path.

        Args:
            inbox_path: The absolute path to look up.

        Returns:
            The matching session, or ``None`` if not found.
        """
        stmt = select(AstroSession).where(AstroSession.inbox_path == inbox_path)
        result = await self.session.exec(stmt)  # type: ignore[call-overload]
        return result.first()

    async def list_by_status(
        self,
        status: SessionStatus,
        offset: int = 0,
        limit: int = 100,
    ) -> list[AstroSession]:
        """Retrieve sessions filtered by lifecycle status.

        Args:
            status: The desired :class:`~app.domain.session.SessionStatus`.
            offset: Pagination offset.
            limit: Maximum number of results.

        Returns:
            List of matching sessions, ordered by creation time descending.
        """
        stmt = (
            select(AstroSession)
            .where(AstroSession.status == status)
            .order_by(AstroSession.created_at.desc())  # type: ignore[attr-defined]
            .offset(offset)
            .limit(limit)
        )
        result = await self.session.exec(stmt)  # type: ignore[call-overload]
        return list(result.all())

    async def list_all_ordered(
        self,
        offset: int = 0,
        limit: int = 100,
    ) -> list[AstroSession]:
        """Retrieve all sessions ordered by creation time (newest first).

        Args:
            offset: Pagination offset.
            limit: Maximum number of results.

        Returns:
            Paginated list of sessions.
        """
        stmt = (
            select(AstroSession)
            .order_by(AstroSession.created_at.desc())  # type: ignore[attr-defined]
            .offset(offset)
            .limit(limit)
        )
        result = await self.session.exec(stmt)  # type: ignore[call-overload]
        return list(result.all())

    async def update_status(
        self,
        session_id: uuid.UUID,
        status: SessionStatus,
    ) -> Optional[AstroSession]:
        """Atomically update the status of a session.

        Args:
            session_id: UUID of the session to update.
            status: Target :class:`~app.domain.session.SessionStatus`.

        Returns:
            The updated session, or ``None`` if not found.
        """
        return await self.update(session_id, {"status": status})
