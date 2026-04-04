"""WebSocket connection manager and Redis pub/sub bridge.

Maintains the registry of active WebSocket connections per job and per session.
Subscribes to the corresponding Redis channels and forwards messages to all
connected clients.

Example:
    >>> manager = WebSocketConnectionManager(event_bus)
    >>> await manager.connect_job(websocket, job_id)
    >>> # WS messages are forwarded automatically from Redis pub/sub
    >>> await manager.disconnect_job(websocket, job_id)
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections import defaultdict

from fastapi import WebSocket

from app.core.logging import get_logger
from app.infrastructure.queue.events_bus import EventBus

logger = get_logger(__name__)


class WebSocketConnectionManager:
    """Registry of active WebSocket connections with Redis pub/sub forwarding.

    Maintains separate registries for job-scoped and session-scoped connections.
    For each unique job/session watched, a background asyncio task subscribes
    to the corresponding Redis channel and broadcasts messages to all connected
    clients.

    Attributes:
        event_bus: The shared :class:`~app.infrastructure.queue.events_bus.EventBus`
            instance used for Redis subscriptions.
    """

    def __init__(self, event_bus: EventBus) -> None:
        """Initialise the manager.

        Args:
            event_bus: Connected :class:`~app.infrastructure.queue.events_bus.EventBus`.
        """
        self.event_bus = event_bus
        # job_id → set of WebSocket connections
        self._job_connections: dict[str, set[WebSocket]] = defaultdict(set)
        # session_id → set of WebSocket connections
        self._session_connections: dict[str, set[WebSocket]] = defaultdict(set)
        # Background tasks per channel
        self._job_tasks: dict[str, asyncio.Task] = {}
        self._session_tasks: dict[str, asyncio.Task] = {}

    async def connect_job(
        self,
        websocket: WebSocket,
        job_id: uuid.UUID,
    ) -> None:
        """Accept and register a WebSocket connection for a job event stream.

        Starts a background task to subscribe to Redis and forward messages
        if this is the first connection for this job.

        Args:
            websocket: The accepted WebSocket connection.
            job_id: The job UUID to watch.
        """
        await websocket.accept()
        key = str(job_id)
        self._job_connections[key].add(websocket)

        if key not in self._job_tasks:
            task = asyncio.create_task(
                self._forward_job_events(job_id),
                name=f"ws_job_{key}",
            )
            self._job_tasks[key] = task
            logger.debug("ws_job_subscription_started", job_id=key)

        logger.info("ws_client_connected_job", job_id=key)

    async def disconnect_job(
        self,
        websocket: WebSocket,
        job_id: uuid.UUID,
    ) -> None:
        """Remove a WebSocket connection from the job registry.

        If this was the last connection for the job, the background forwarding
        task is cancelled.

        Args:
            websocket: The WebSocket connection to remove.
            job_id: The associated job UUID.
        """
        key = str(job_id)
        self._job_connections[key].discard(websocket)

        if not self._job_connections[key]:
            del self._job_connections[key]
            task = self._job_tasks.pop(key, None)
            if task:
                task.cancel()
            logger.debug("ws_job_subscription_stopped", job_id=key)

        logger.info("ws_client_disconnected_job", job_id=key)

    async def connect_session(
        self,
        websocket: WebSocket,
        session_id: uuid.UUID,
    ) -> None:
        """Accept and register a WebSocket connection for a session event stream.

        Args:
            websocket: The accepted WebSocket connection.
            session_id: The session UUID to watch.
        """
        await websocket.accept()
        key = str(session_id)
        self._session_connections[key].add(websocket)

        if key not in self._session_tasks:
            task = asyncio.create_task(
                self._forward_session_events(session_id),
                name=f"ws_session_{key}",
            )
            self._session_tasks[key] = task
            logger.debug("ws_session_subscription_started", session_id=key)

    async def disconnect_session(
        self,
        websocket: WebSocket,
        session_id: uuid.UUID,
    ) -> None:
        """Remove a WebSocket connection from the session registry.

        Args:
            websocket: The WebSocket connection to remove.
            session_id: The associated session UUID.
        """
        key = str(session_id)
        self._session_connections[key].discard(websocket)

        if not self._session_connections[key]:
            del self._session_connections[key]
            task = self._session_tasks.pop(key, None)
            if task:
                task.cancel()

    async def broadcast_to_job(
        self,
        job_id: uuid.UUID,
        message: dict | str,
    ) -> None:
        """Send a message directly to all clients watching a job.

        Args:
            job_id: Target job UUID.
            message: Dict (auto-serialised to JSON) or pre-serialised string.
        """
        payload = message if isinstance(message, str) else json.dumps(message)
        key = str(job_id)
        dead: list[WebSocket] = []
        for ws in list(self._job_connections.get(key, [])):
            try:
                await ws.send_text(payload)
            except Exception:  # noqa: BLE001
                dead.append(ws)
        for ws in dead:
            self._job_connections[key].discard(ws)

    # ── Private forwarding loops ──────────────────────────────────────────────

    async def _forward_job_events(self, job_id: uuid.UUID) -> None:
        """Subscribe to a job's Redis channel and forward messages to WS clients.

        Runs until cancelled (all clients disconnected) or the Redis stream ends.

        Args:
            job_id: UUID of the job channel to subscribe to.
        """
        try:
            async for raw in self.event_bus.subscribe_to_job(job_id):
                await self.broadcast_to_job(job_id, raw)
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.warning("ws_job_forward_error", job_id=str(job_id), error=str(exc))

    async def _forward_session_events(self, session_id: uuid.UUID) -> None:
        """Subscribe to a session's Redis channel and forward to WS clients.

        Args:
            session_id: UUID of the session channel to subscribe to.
        """
        try:
            async for raw in self.event_bus.subscribe_to_session(session_id):
                payload = json.dumps(raw)
                key = str(session_id)
                for ws in list(self._session_connections.get(key, [])):
                    try:
                        await ws.send_text(payload)
                    except Exception:  # noqa: BLE001
                        pass
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.warning("ws_session_forward_error", session_id=str(session_id), error=str(exc))
