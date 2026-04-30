"""Repository for gallery download audit records.

Provides read-only analytics queries over the ``astro_gallery_downloads``
table, used exclusively by the admin analytics surface.  Write operations
(inserting new download requests) remain in :mod:`app.services.gallery_service`
to keep the hot path dependency-free.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.gallery import GalleryDownload
from app.domain.session import AstroSession


SortField = Literal["requested_at", "email", "format"]
SortDir = Literal["asc", "desc"]


class GalleryDownloadRepository:
    """Analytics repository for :class:`~app.domain.gallery.GalleryDownload` rows.

    All methods are read-only; they never mutate the audit log.

    Args:
        session: Active async database session.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialise the repository with a database session.

        Args:
            session: An active :class:`~sqlalchemy.ext.asyncio.AsyncSession`.
        """
        self.session = session

    async def count_total(self) -> int:
        """Return the total number of download requests ever recorded.

        Returns:
            Total row count in ``astro_gallery_downloads``.
        """
        result = await self.session.execute(
            sa.select(sa.func.count()).select_from(GalleryDownload)
        )
        return int(result.scalar_one())

    async def count_unique_emails(self) -> int:
        """Return the number of distinct email addresses across all download requests.

        Returns:
            Count of distinct :attr:`~app.domain.gallery.GalleryDownload.email` values.
        """
        result = await self.session.execute(
            sa.select(sa.func.count(sa.distinct(GalleryDownload.email)))
        )
        return int(result.scalar_one())

    async def count_by_format(self) -> dict[str, int]:
        """Return the download count broken down by format.

        Returns:
            Mapping of format string (``"tiff"`` or ``"fits"``) to count.
        """
        result = await self.session.execute(
            sa.select(GalleryDownload.format, sa.func.count().label("n"))
            .group_by(GalleryDownload.format)
        )
        return {row.format: row.n for row in result}

    async def downloads_over_time(self, days: int = 30) -> list[dict[str, object]]:
        """Return daily download counts for the last *days* days.

        Rows with zero downloads on a given day are **not** included; the caller
        should fill gaps for charting purposes.

        Args:
            days: Number of calendar days to look back (inclusive today).

        Returns:
            List of dicts with keys ``date`` (ISO-8601 date string) and ``count`` (int),
            ordered chronologically.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        result = await self.session.execute(
            sa.select(
                sa.func.date_trunc("day", GalleryDownload.requested_at).label("day"),
                sa.func.count().label("n"),
            )
            .where(GalleryDownload.requested_at >= cutoff)
            .group_by(sa.text("1"))
            .order_by(sa.text("1"))
        )
        return [
            {"date": row.day.date().isoformat(), "count": row.n}
            for row in result
        ]

    async def top_sessions(self, limit: int = 10) -> list[dict[str, object]]:
        """Return the top *limit* gallery sessions by download count.

        Uses the denormalised ``gallery_download_count`` column on
        :class:`~app.domain.session.AstroSession` (maintained by the serve-download
        endpoint) to avoid an expensive GROUP BY over the audit table.

        Args:
            limit: Maximum number of sessions to return.

        Returns:
            List of dicts with keys ``session_id`` (str), ``name`` (str) and
            ``download_count`` (int), ordered descending.
        """
        result = await self.session.execute(
            sa.select(
                AstroSession.id,
                AstroSession.name,
                AstroSession.gallery_download_count,
            )
            .where(AstroSession.is_in_gallery.is_(True))
            .order_by(AstroSession.gallery_download_count.desc())  # type: ignore[attr-defined]
            .limit(limit)
        )
        return [
            {
                "session_id": str(row.id),
                "name": row.name,
                "download_count": row.gallery_download_count,
            }
            for row in result
        ]

    async def list_downloads(
        self,
        offset: int = 0,
        limit: int = 25,
        email_filter: Optional[str] = None,
        format_filter: Optional[str] = None,
        session_id_filter: Optional[uuid.UUID] = None,
        sort_by: SortField = "requested_at",
        sort_dir: SortDir = "desc",
    ) -> tuple[list[dict[str, object]], int]:
        """Return a paginated, filtered, sorted list of download records.

        Each row is augmented with the session name via a LEFT OUTER JOIN on
        :class:`~app.domain.session.AstroSession`.

        Args:
            offset: Number of rows to skip.
            limit: Maximum rows to return.
            email_filter: Optional substring match on the email column (case-insensitive).
            format_filter: Optional exact match on the format column.
            session_id_filter: Optional exact UUID match on session_id.
            sort_by: Column to sort by — one of ``"requested_at"``, ``"email"``, ``"format"``.
            sort_dir: Sort direction — ``"asc"`` or ``"desc"``.

        Returns:
            A tuple of ``(rows, total_count)`` where *rows* is a list of dicts
            and *total_count* is the unfiltered match count (for pagination).
        """
        # Build a base column set with the session name joined in.
        base_stmt = (
            sa.select(
                GalleryDownload.id,
                GalleryDownload.session_id,
                AstroSession.name.label("session_name"),  # type: ignore[union-attr]
                GalleryDownload.email,
                GalleryDownload.format,
                GalleryDownload.requester_ip,
                GalleryDownload.requested_at,
            )
            .outerjoin(
                AstroSession,
                GalleryDownload.session_id == AstroSession.id,
            )
        )

        # Apply filters.
        if email_filter:
            base_stmt = base_stmt.where(
                GalleryDownload.email.ilike(f"%{email_filter}%")  # type: ignore[union-attr]
            )
        if format_filter:
            base_stmt = base_stmt.where(GalleryDownload.format == format_filter)
        if session_id_filter is not None:
            base_stmt = base_stmt.where(GalleryDownload.session_id == session_id_filter)

        # Count without pagination.
        count_stmt = sa.select(sa.func.count()).select_from(base_stmt.subquery())
        total_result = await self.session.execute(count_stmt)
        total = int(total_result.scalar_one())

        # Apply sort + pagination.
        sort_col = {
            "requested_at": GalleryDownload.requested_at,
            "email": GalleryDownload.email,
            "format": GalleryDownload.format,
        }[sort_by]
        ordered = (
            base_stmt
            .order_by(sort_col.desc() if sort_dir == "desc" else sort_col.asc())  # type: ignore[union-attr]
            .offset(offset)
            .limit(limit)
        )

        rows_result = await self.session.execute(ordered)
        rows = [
            {
                "id": str(row.id),
                "session_id": str(row.session_id),
                "session_name": row.session_name,
                "email": row.email,
                "format": row.format,
                "requester_ip": row.requester_ip,
                "requested_at": row.requested_at,
            }
            for row in rows_result
        ]
        return rows, total
