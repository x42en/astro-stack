"""Service layer for the global ``AppSettings`` singleton.

All database interaction is isolated here so routers stay thin.  The table
always contains exactly one row (``id = 1``); helper functions create it on
demand if for any reason the migration seed was skipped.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.domain.app_settings import AppSettings, AppSettingsUpdate


async def get_app_settings(db: AsyncSession) -> AppSettings:
    """Return the singleton settings row, creating it from env defaults if absent.

    Args:
        db: Active async database session.

    Returns:
        The ``AppSettings`` ORM object with ``id = 1``.
    """
    result = await db.execute(select(AppSettings).where(AppSettings.id == 1))
    row = result.scalar_one_or_none()
    if row is None:
        row = _make_default_row()
        db.add(row)
        await db.commit()
        await db.refresh(row)
    return row


async def upsert_app_settings(
    db: AsyncSession,
    patch: AppSettingsUpdate,
    user_id: Optional[str],
) -> AppSettings:
    """Apply a partial update to the singleton settings row.

    Uses ``SELECT … FOR UPDATE`` to prevent concurrent admin writes from
    clobbering each other.

    Args:
        db: Active async database session.
        patch: Partial update; ``None`` fields are left untouched.
        user_id: Opaque identifier of the administrator performing the change.
            Stored for audit purposes; may be ``None`` in ``disabled`` auth mode.

    Returns:
        The updated ``AppSettings`` ORM object.
    """
    result = await db.execute(
        select(AppSettings).where(AppSettings.id == 1).with_for_update()
    )
    row = result.scalar_one_or_none()
    if row is None:
        row = _make_default_row()
        db.add(row)

    update_data = patch.model_dump(exclude_none=True)
    for field, value in update_data.items():
        setattr(row, field, value)

    row.updated_at = datetime.now(timezone.utc)
    row.updated_by_user_id = user_id

    await db.commit()
    await db.refresh(row)
    return row


def _make_default_row() -> AppSettings:
    """Build a default singleton row from current environment settings."""
    s = get_settings()
    return AppSettings(
        id=1,
        inbox_path=s.inbox_path,
        ollama_url=s.ollama_url,
        ollama_model=s.ollama_model,
        pipeline_max_retries=s.pipeline_max_retries,
        session_stability_delay=float(s.session_stability_delay),
        updated_at=datetime.now(timezone.utc),
        updated_by_user_id=None,
    )
