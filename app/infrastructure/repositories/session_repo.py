"""Repository for :class:`~app.domain.session.AstroSession` persistence.

Provides session-specific query methods in addition to the generic CRUD
operations inherited from :class:`~app.infrastructure.repositories.base.BaseRepository`.
"""

from __future__ import annotations

import uuid
from typing import Optional

from sqlalchemy import func
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
        result = await self.session.execute(stmt)
        return result.first()

    async def count_all(self, search: str | None = None) -> int:
        """Return the total count of all sessions, optionally filtered by name search.

        Args:
            search: Optional name substring filter (case-insensitive).

        Returns:
            Total matching session count.
        """
        stmt = select(func.count(AstroSession.id))  # type: ignore[arg-type]
        if search:
            stmt = stmt.where(AstroSession.name.ilike(f"%{search}%"))  # type: ignore[union-attr]
        result = await self.session.execute(stmt)
        return result.scalar_one()

    async def count_by_status(self, status: SessionStatus, search: str | None = None) -> int:
        """Return the count of sessions with a given status.

        Args:
            status: The lifecycle status to filter by.
            search: Optional name substring filter (case-insensitive).

        Returns:
            Total matching session count.
        """
        stmt = select(func.count(AstroSession.id)).where(  # type: ignore[arg-type]
            AstroSession.status == status
        )
        if search:
            stmt = stmt.where(AstroSession.name.ilike(f"%{search}%"))  # type: ignore[union-attr]
        result = await self.session.execute(stmt)
        return result.scalar_one()

    async def list_by_status(
        self,
        status: SessionStatus,
        offset: int = 0,
        limit: int = 100,
        search: str | None = None,
    ) -> list[AstroSession]:
        """Retrieve sessions filtered by lifecycle status.

        Args:
            status: The desired :class:`~app.domain.session.SessionStatus`.
            offset: Pagination offset.
            limit: Maximum number of results.
            search: Optional name substring filter (case-insensitive).

        Returns:
            List of matching sessions, ordered by creation time descending.
        """
        stmt = select(AstroSession).where(AstroSession.status == status)
        if search:
            stmt = stmt.where(AstroSession.name.ilike(f"%{search}%"))  # type: ignore[union-attr]
        stmt = stmt.order_by(AstroSession.created_at.desc()).offset(offset).limit(limit)  # type: ignore[attr-defined]
        result = await self.session.execute(stmt)
        return list(result.all())

    async def list_all_ordered(
        self,
        offset: int = 0,
        limit: int = 100,
        search: str | None = None,
    ) -> list[AstroSession]:
        """Retrieve all sessions ordered by creation time (newest first).

        Args:
            offset: Pagination offset.
            limit: Maximum number of results.
            search: Optional name substring filter (case-insensitive).

        Returns:
            Paginated list of sessions.
        """
        stmt = select(AstroSession)
        if search:
            stmt = stmt.where(AstroSession.name.ilike(f"%{search}%"))  # type: ignore[union-attr]
        stmt = stmt.order_by(AstroSession.created_at.desc()).offset(offset).limit(limit)  # type: ignore[attr-defined]
        result = await self.session.execute(stmt)
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
