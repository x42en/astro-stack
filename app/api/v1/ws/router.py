"""WebSocket endpoints for real-time pipeline and session event streaming.

All WebSocket connections support optional JWT authentication: if a ``token``
query parameter is provided, it is validated. When ``AUTH_ENABLED=false``
(the default), the token is accepted but not required.

Endpoints:
    - ``ws://<host>/ws/jobs/{job_id}``     — stream events for a specific job
    - ``ws://<host>/ws/sessions/{session_id}`` — stream events for a session
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from app.api.middleware.auth import extract_ws_token, validate_optional_token
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/ws", tags=["websocket"])


@router.websocket("/jobs/{job_id}")
async def websocket_job_events(
    websocket: WebSocket,
    job_id: uuid.UUID,
    token: str | None = Query(default=None, description="Optional JWT bearer token"),
) -> None:
    """Stream real-time pipeline events for a specific job.

    Emits :class:`~app.domain.ws_event.ProgressEvent`,
    :class:`~app.domain.ws_event.LogEvent`,
    :class:`~app.domain.ws_event.StepStatusEvent`,
    :class:`~app.domain.ws_event.ErrorEvent`, and
    :class:`~app.domain.ws_event.CompletedEvent` payloads as JSON text frames.

    Args:
        websocket: The incoming WebSocket connection.
        job_id: UUID of the pipeline job to watch.
        token: Optional JWT bearer token for authentication.
    """
    # Validate JWT if provided (or required by settings)
    await validate_optional_token(token)

    manager = websocket.app.state.ws_manager
    await manager.connect_job(websocket, job_id)
    logger.info("ws_job_connected", job_id=str(job_id))

    try:
        # Keep connection alive — forward messages from Redis via the manager
        while True:
            # Wait for client messages (ping/pong keepalive or close frame)
            data = await websocket.receive_text()
            # Echo ping frames for keepalive support
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect_job(websocket, job_id)
        logger.info("ws_job_disconnected", job_id=str(job_id))


@router.websocket("/sessions/{session_id}")
async def websocket_session_events(
    websocket: WebSocket,
    session_id: uuid.UUID,
    token: str | None = Query(default=None, description="Optional JWT bearer token"),
) -> None:
    """Stream real-time events for a session (detection, ready, job updates).

    Args:
        websocket: The incoming WebSocket connection.
        session_id: UUID of the session to watch.
        token: Optional JWT bearer token for authentication.
    """
    await validate_optional_token(token)

    manager = websocket.app.state.ws_manager
    await manager.connect_session(websocket, session_id)
    logger.info("ws_session_connected", session_id=str(session_id))

    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect_session(websocket, session_id)
        logger.info("ws_session_disconnected", session_id=str(session_id))
