"""Admin REST endpoints for gallery analytics.

All endpoints require the ``admin`` role.  In ``disabled`` auth mode the
role check is bypassed (local development only).

Endpoints
---------
* ``GET /admin/gallery/stats``     — Aggregated KPIs + chart data.
* ``GET /admin/gallery/downloads`` — Paginated, filterable, sortable download log.
"""

from __future__ import annotations

import uuid
from typing import Literal, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.middleware.auth import require_role
from app.core.database import get_async_session
from app.services.gallery_admin_service import (
    GalleryAdminService,
    GalleryStats,
    PaginatedDownloads,
)

router = APIRouter(prefix="/admin/gallery", tags=["admin"])

# Shared admin dependency — all routes in this module require the admin role.
_require_admin = Depends(require_role("admin"))


@router.get(
    "/stats",
    response_model=GalleryStats,
    summary="Gallery analytics KPIs",
    dependencies=[_require_admin],
)
async def get_gallery_stats(
    days: int = Query(default=30, ge=7, le=365, description="Lookback window in days for the time-series chart"),
    db: AsyncSession = Depends(get_async_session),
) -> GalleryStats:
    """Return aggregated gallery download statistics.

    Includes total downloads, unique emails, per-format counts, a daily
    time-series for the last *days* calendar days, and the top 10 sessions
    by download count.

    Args:
        days: Lookback window in calendar days (7–365, default 30).
        db: Injected async database session.

    Returns:
        A :class:`~app.services.gallery_admin_service.GalleryStats` payload.
    """
    service = GalleryAdminService(db)
    return await service.get_stats(days=days)


@router.get(
    "/downloads",
    response_model=PaginatedDownloads,
    summary="Paginated gallery download log",
    dependencies=[_require_admin],
)
async def list_gallery_downloads(
    page: int = Query(default=1, ge=1, description="1-based page number"),
    page_size: int = Query(default=25, ge=1, le=100, description="Records per page"),
    email: Optional[str] = Query(default=None, description="Case-insensitive email substring filter"),
    format: Optional[Literal["tiff", "fits"]] = Query(default=None, description="Exact format filter"),
    session_id: Optional[uuid.UUID] = Query(default=None, description="Filter by session UUID"),
    sort_by: Literal["requested_at", "email", "format"] = Query(
        default="requested_at",
        description="Column to sort by",
    ),
    sort_dir: Literal["asc", "desc"] = Query(default="desc", description="Sort direction"),
    db: AsyncSession = Depends(get_async_session),
) -> PaginatedDownloads:
    """Return a paginated list of gallery download request records.

    Supports filtering by email substring, exact format, and session UUID.
    Results are sortable on ``requested_at``, ``email``, and ``format``.
    Each row includes the session name (``None`` if the session was deleted).

    Args:
        page: 1-based page index.
        page_size: Records per page (1–100).
        email: Optional case-insensitive email substring filter.
        format: Optional exact format filter (``"tiff"`` or ``"fits"``).
        session_id: Optional exact session UUID filter.
        sort_by: Column to sort by.
        sort_dir: Ascending or descending direction.
        db: Injected async database session.

    Returns:
        A :class:`~app.services.gallery_admin_service.PaginatedDownloads` payload.
    """
    service = GalleryAdminService(db)
    return await service.list_downloads(
        page=page,
        page_size=page_size,
        email=email,
        format=format,
        session_id=session_id,
        sort_by=sort_by,
        sort_dir=sort_dir,
    )
