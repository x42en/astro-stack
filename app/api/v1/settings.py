"""REST API endpoints for global application settings.

Endpoints
---------
* ``GET  /settings``  — read the current settings (public, no auth required)
* ``PUT  /settings``  — update settings (admin role required)

The settings object is a singleton row in the database.  Only administrators
(``require_role("admin")``) may write; any authenticated or anonymous client
may read so the frontend can display operational defaults before the user logs
in.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.middleware.auth import get_current_user, require_role
from app.core.database import get_async_session
from app.domain.app_settings import AppSettingsRead, AppSettingsUpdate
from app.services.app_settings_service import get_app_settings, upsert_app_settings

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("", response_model=AppSettingsRead, summary="Get global settings")
async def read_settings(
    db: AsyncSession = Depends(get_async_session),
) -> AppSettingsRead:
    """Return the current global operational settings.

    No authentication required — the frontend needs these before the user
    has signed in in order to display correct defaults.
    """
    row = await get_app_settings(db)
    return AppSettingsRead.model_validate(row, from_attributes=True)


@router.put(
    "",
    response_model=AppSettingsRead,
    summary="Update global settings (admin only)",
    dependencies=[Depends(require_role("admin"))],
)
async def update_settings(
    body: AppSettingsUpdate,
    db: AsyncSession = Depends(get_async_session),
    # get_current_user returns the raw JWT claims dict (or None in disabled mode).
    claims: Optional[dict] = Depends(get_current_user),
) -> AppSettingsRead:
    """Partially update global settings.

    Only users with the ``admin`` role may call this endpoint.  In
    ``disabled`` auth mode the role check is bypassed and ``user_id`` will
    be ``None`` in the audit column.
    """
    user_id: Optional[str] = claims.get("sub") if claims else None
    row = await upsert_app_settings(db, body, user_id)
    return AppSettingsRead.model_validate(row, from_attributes=True)
