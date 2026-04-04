"""REST API endpoints for astrophotography sessions."""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.middleware.auth import get_current_user
from app.core.database import get_async_session
from app.domain.job import ProfilePreset
from app.domain.session import SessionRead, SessionStatus
from app.services.job_service import JobService
from app.services.session_service import SessionService

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get(
    "",
    response_model=list[SessionRead],
    summary="List all sessions",
    description="Returns a paginated list of sessions, optionally filtered by status.",
)
async def list_sessions(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
    status: Optional[SessionStatus] = Query(default=None),
    db: AsyncSession = Depends(get_async_session),
    _user: Optional[dict] = Depends(get_current_user),
) -> list[SessionRead]:
    """List sessions with optional status filter.

    Args:
        offset: Pagination offset.
        limit: Maximum results to return.
        status: Optional status filter.
        db: Injected database session.
        _user: Injected auth user (None if auth disabled).

    Returns:
        List of :class:`~app.domain.session.SessionRead` objects.
    """
    service = SessionService(db)
    sessions = await service.list_sessions(offset=offset, limit=limit, status=status)
    return [SessionRead.model_validate(s.model_dump()) for s in sessions]


@router.get(
    "/{session_id}",
    response_model=SessionRead,
    summary="Get session details",
)
async def get_session(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
    _user: Optional[dict] = Depends(get_current_user),
) -> SessionRead:
    """Retrieve a session by ID.

    Args:
        session_id: Session UUID.
        db: Injected database session.
        _user: Injected auth user.

    Returns:
        :class:`~app.domain.session.SessionRead` for the found session.
    """
    service = SessionService(db)
    session = await service.get_or_404(session_id)
    return SessionRead.model_validate(session.model_dump())


@router.post(
    "/{session_id}/process",
    response_model=dict,
    status_code=202,
    summary="Start pipeline processing",
    description="Enqueue a pipeline job for the specified session.",
)
async def start_processing(
    session_id: uuid.UUID,
    preset: ProfilePreset = Query(
        default=ProfilePreset.STANDARD,
        description="Processing profile preset to use.",
    ),
    profile_id: Optional[uuid.UUID] = Query(
        default=None,
        description="UUID of a saved advanced profile (required for ADVANCED preset).",
    ),
    db: AsyncSession = Depends(get_async_session),
    _user: Optional[dict] = Depends(get_current_user),
) -> dict:
    """Enqueue a pipeline job for a session.

    Args:
        session_id: Session UUID to process.
        preset: Processing preset (quick/standard/quality/advanced).
        profile_id: Saved profile UUID for ADVANCED preset.
        db: Injected database session.
        _user: Injected auth user.

    Returns:
        Dict with ``job_id`` of the created pipeline job.
    """
    service = JobService(db)
    job = await service.start_pipeline(
        session_id=session_id,
        preset=preset,
        profile_id=profile_id,
    )
    return {"job_id": str(job.id), "status": job.status.value}


@router.post(
    "/{session_id}/cancel",
    response_model=dict,
    summary="Cancel active pipeline job",
)
async def cancel_processing(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
    _user: Optional[dict] = Depends(get_current_user),
) -> dict:
    """Cancel the active pipeline job for a session.

    Args:
        session_id: Session UUID.
        db: Injected database session.
        _user: Injected auth user.

    Returns:
        Dict with ``message`` confirming cancellation request.
    """
    job_service = JobService(db)
    sess_service = SessionService(db)
    session = await sess_service.get_or_404(session_id)

    # Find the active job for this session
    from app.infrastructure.repositories.job_repo import JobRepository  # noqa: PLC0415

    job_repo = JobRepository(db)
    active_job = await job_repo.get_active_job_for_session(session_id)

    if active_job is None:
        return {"message": "No active job to cancel for this session."}

    await job_service.cancel_job(active_job.id)
    return {"message": f"Cancellation requested for job {active_job.id}."}
