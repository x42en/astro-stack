"""Redis pub/sub event bus for decoupled pipeline → WebSocket communication.

Workers publish typed events to a Redis channel keyed by ``job_id`` or
``session_id``. The WebSocket manager subscribes and forwards messages to
all connected clients watching that resource.

Channel naming conventions::

    job:{job_id}        — events scoped to a specific pipeline job
    session:{session_id} — session-level events (watchdog, ready, etc.)
    broadcast           — events sent to all connected clients

Example:
    >>> bus = EventBus(redis_url="redis://localhost:6379/0")
    >>> await bus.connect()
    >>> await bus.publish_job_event(job_id, progress_event)
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncGenerator

import redis.asyncio as aioredis

from app.core.logging import get_logger
from app.domain.ws_event import AnyEvent, BaseEvent

logger = get_logger(__name__)


def _job_channel(job_id: uuid.UUID) -> str:
    """Return the Redis channel name for a job-scoped event stream.

    Args:
        job_id: UUID of the target job.

    Returns:
        Channel name string, e.g. ``"job:abc123"``.
    """
    return f"job:{job_id}"


def _session_channel(session_id: uuid.UUID) -> str:
    """Return the Redis channel name for a session-scoped event stream.

    Args:
        session_id: UUID of the target session.

    Returns:
        Channel name string, e.g. ``"session:abc123"``.
    """
    return f"session:{session_id}"


BROADCAST_CHANNEL = "broadcast"


class EventBus:
    """Redis pub/sub bus used by workers to emit pipeline events.

    A single shared ``EventBus`` instance is held by the FastAPI application
    state. Worker tasks obtain a separate publisher connection per process via
    their own ``EventBus`` instance configured from the same Redis URL.

    Attributes:
        _redis_url: Redis connection string.
        _publisher: Async Redis client used for publishing.
        _subscriber: Async Redis client used for subscribing (separate connection).
    """

    def __init__(self, redis_url: str) -> None:
        """Initialise the bus with a Redis connection URL.

        Args:
            redis_url: Full Redis DSN, e.g. ``"redis://redis:6379/0"``.
        """
        self._redis_url = redis_url
        self._publisher: aioredis.Redis | None = None  # type: ignore[type-arg]
        self._subscriber: aioredis.Redis | None = None  # type: ignore[type-arg]

    async def connect(self) -> None:
        """Open publisher and subscriber Redis connections.

        Must be called once before any publish or subscribe operations.
        Idempotent — subsequent calls are no-ops if already connected.
        """
        if self._publisher is None:
            self._publisher = await aioredis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        if self._subscriber is None:
            self._subscriber = await aioredis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        logger.debug("event_bus_connected", redis_url=self._redis_url)

    async def disconnect(self) -> None:
        """Close both Redis connections.

        Should be called during application shutdown.
        """
        if self._publisher is not None:
            await self._publisher.aclose()
            self._publisher = None
        if self._subscriber is not None:
            await self._subscriber.aclose()
            self._subscriber = None
        logger.debug("event_bus_disconnected")

    async def publish_job_event(
        self,
        job_id: uuid.UUID,
        event: BaseEvent,
    ) -> None:
        """Publish an event to the job-scoped Redis channel.

        Args:
            job_id: UUID of the job this event belongs to.
            event: Any typed event that inherits from :class:`~app.domain.ws_event.BaseEvent`.

        Raises:
            RuntimeError: If the bus has not been connected via :meth:`connect`.
        """
        if self._publisher is None:
            raise RuntimeError("EventBus not connected. Call connect() first.")
        channel = _job_channel(job_id)
        payload = event.model_dump_json()
        await self._publisher.publish(channel, payload)
        logger.debug("event_published", channel=channel, event_type=event.type)

    async def publish_session_event(
        self,
        session_id: uuid.UUID,
        event: BaseEvent,
    ) -> None:
        """Publish an event to the session-scoped Redis channel.

        Args:
            session_id: UUID of the session this event belongs to.
            event: Any typed event inheriting from :class:`~app.domain.ws_event.BaseEvent`.

        Raises:
            RuntimeError: If the bus has not been connected.
        """
        if self._publisher is None:
            raise RuntimeError("EventBus not connected. Call connect() first.")
        channel = _session_channel(session_id)
        payload = event.model_dump_json()
        await self._publisher.publish(channel, payload)
        logger.debug("session_event_published", channel=channel, event_type=event.type)

    async def publish_broadcast(self, event: BaseEvent) -> None:
        """Publish an event to the global broadcast channel.

        All connected ``/ws/broadcast`` clients will receive this event.

        Args:
            event: Any typed event inheriting from :class:`~app.domain.ws_event.BaseEvent`.

        Raises:
            RuntimeError: If the bus has not been connected.
        """
        if self._publisher is None:
            raise RuntimeError("EventBus not connected. Call connect() first.")
        payload = event.model_dump_json()
        await self._publisher.publish(BROADCAST_CHANNEL, payload)
        logger.debug("broadcast_event_published", event_type=event.type)

    async def subscribe_to_job(
        self,
        job_id: uuid.UUID,
    ) -> AsyncGenerator[AnyEvent | dict, None]:
        """Subscribe to the job event channel and yield raw message dicts.

        This is consumed by the WebSocket manager. The caller is responsible
        for unsubscribing when the WebSocket connection closes.

        Args:
            job_id: UUID of the job to subscribe to.

        Yields:
            Parsed JSON dicts from the Redis channel.

        Raises:
            RuntimeError: If the bus has not been connected.
        """
        if self._subscriber is None:
            raise RuntimeError("EventBus not connected. Call connect() first.")
        channel = _job_channel(job_id)
        pubsub = self._subscriber.pubsub()
        await pubsub.subscribe(channel)
        logger.debug("subscribed_to_job", channel=channel)
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    yield json.loads(message["data"])
        finally:
            await pubsub.unsubscribe(channel)
            logger.debug("unsubscribed_from_job", channel=channel)

    async def subscribe_to_session(
        self,
        session_id: uuid.UUID,
    ) -> AsyncGenerator[dict, None]:
        """Subscribe to the session event channel and yield raw message dicts.

        Args:
            session_id: UUID of the session to subscribe to.

        Yields:
            Parsed JSON dicts from the Redis channel.

        Raises:
            RuntimeError: If the bus has not been connected.
        """
        if self._subscriber is None:
            raise RuntimeError("EventBus not connected. Call connect() first.")
        channel = _session_channel(session_id)
        pubsub = self._subscriber.pubsub()
        await pubsub.subscribe(channel)
        logger.debug("subscribed_to_session", channel=channel)
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    yield json.loads(message["data"])
        finally:
            await pubsub.unsubscribe(channel)
            logger.debug("unsubscribed_from_session", channel=channel)

    async def subscribe_to_broadcast(self) -> AsyncGenerator[dict, None]:
        """Subscribe to the global broadcast channel and yield raw message dicts.

        Used by the ``/ws/broadcast`` WebSocket endpoint to forward session-level
        status changes to all connected clients.

        Yields:
            Parsed JSON dicts from the Redis broadcast channel.

        Raises:
            RuntimeError: If the bus has not been connected.
        """
        if self._subscriber is None:
            raise RuntimeError("EventBus not connected. Call connect() first.")
        pubsub = self._subscriber.pubsub()
        await pubsub.subscribe(BROADCAST_CHANNEL)
        logger.debug("subscribed_to_broadcast")
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    yield json.loads(message["data"])
        finally:
            await pubsub.unsubscribe(BROADCAST_CHANNEL)
            logger.debug("unsubscribed_from_broadcast")
