"""Admin service for gallery analytics.

Provides aggregated statistics and paginated download records for the
admin analytics dashboard.  All reads go through
:class:`~app.infrastructure.repositories.gallery_download_repo.GalleryDownloadRepository`.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.repositories.gallery_download_repo import (
    GalleryDownloadRepository,
    SortDir,
    SortField,
)


# ── Response schemas ──────────────────────────────────────────────────────────


class DailyCount(BaseModel):
    """Single data point for the downloads-over-time chart.

    Attributes:
        date: ISO-8601 calendar date (``YYYY-MM-DD``).
        count: Number of download requests on that day.
    """

    date: str
    count: int


class TopSession(BaseModel):
    """Session entry in the top-sessions bar chart.

    Attributes:
        session_id: UUID string of the gallery session.
        name: Human-readable session name.
        download_count: Total downloads recorded for that session.
    """

    session_id: str
    name: str
    download_count: int


class GalleryStats(BaseModel):
    """Aggregated KPIs for the gallery analytics dashboard.

    Attributes:
        total_downloads: All-time download request count.
        unique_emails: Count of distinct email addresses that requested a download.
        tiff_count: Download requests for the TIFF format.
        fits_count: Download requests for the FITS format.
        downloads_by_day: Daily counts for the last *N* days (sparse — missing days
            have an implicit count of 0).
        top_sessions: Top sessions ordered by ``download_count`` descending.
    """

    total_downloads: int
    unique_emails: int
    tiff_count: int
    fits_count: int
    downloads_by_day: list[DailyCount]
    top_sessions: list[TopSession]


class GalleryDownloadRow(BaseModel):
    """Single download record for the admin listing table.

    Attributes:
        id: Download record UUID.
        session_id: Parent session UUID.
        session_name: Human-readable session name (may be ``None`` if the session
            was deleted after the download request).
        email: Email address supplied by the requester.
        format: Requested format — ``"tiff"`` or ``"fits"``.
        requester_ip: IP address of the requester (may be ``None``).
        requested_at: UTC timestamp of the request.
    """

    id: str
    session_id: str
    session_name: Optional[str]
    email: str
    format: str
    requester_ip: Optional[str]
    requested_at: datetime


class PaginatedDownloads(BaseModel):
    """Paginated response for the admin download listing endpoint.

    Attributes:
        items: Download records for the current page.
        total: Total number of records matching the applied filters.
        page: Current 1-based page number.
        page_size: Number of records per page.
    """

    items: list[GalleryDownloadRow]
    total: int
    page: int
    page_size: int


# ── Service ────────────────────────────────────────────────────────────────────


class GalleryAdminService:
    """Application-layer service for gallery admin analytics.

    Args:
        db: Active async database session.
    """

    def __init__(self, db: AsyncSession) -> None:
        """Initialise with a database session.

        Args:
            db: An active :class:`~sqlalchemy.ext.asyncio.AsyncSession`.
        """
        self._repo = GalleryDownloadRepository(db)

    async def get_stats(self, days: int = 30) -> GalleryStats:
        """Compute aggregated gallery KPIs.

        Args:
            days: Lookback window in calendar days for the time-series chart.

        Returns:
            A populated :class:`GalleryStats` instance.
        """
        total = await self._repo.count_total()
        unique = await self._repo.count_unique_emails()
        by_format = await self._repo.count_by_format()
        over_time = await self._repo.downloads_over_time(days=days)
        top = await self._repo.top_sessions(limit=10)

        return GalleryStats(
            total_downloads=total,
            unique_emails=unique,
            tiff_count=by_format.get("tiff", 0),
            fits_count=by_format.get("fits", 0),
            downloads_by_day=[DailyCount(**d) for d in over_time],
            top_sessions=[TopSession(**s) for s in top],
        )

    async def list_downloads(
        self,
        page: int = 1,
        page_size: int = 25,
        email: Optional[str] = None,
        format: Optional[str] = None,
        session_id: Optional[uuid.UUID] = None,
        sort_by: SortField = "requested_at",
        sort_dir: SortDir = "desc",
    ) -> PaginatedDownloads:
        """Return a paginated, filtered list of download records.

        Args:
            page: 1-based page number.
            page_size: Records per page (clamped to 1–100).
            email: Optional email substring filter (case-insensitive).
            format: Optional exact format filter (``"tiff"`` or ``"fits"``).
            session_id: Optional session UUID filter.
            sort_by: Column to sort by.
            sort_dir: Sort direction.

        Returns:
            A :class:`PaginatedDownloads` response.
        """
        page = max(1, page)
        page_size = max(1, min(page_size, 100))
        offset = (page - 1) * page_size

        rows, total = await self._repo.list_downloads(
            offset=offset,
            limit=page_size,
            email_filter=email,
            format_filter=format,
            session_id_filter=session_id,
            sort_by=sort_by,
            sort_dir=sort_dir,
        )
        return PaginatedDownloads(
            items=[GalleryDownloadRow(**r) for r in rows],
            total=total,
            page=page,
            page_size=page_size,
        )



