"""File system watcher service for automatic session detection.

Monitors the ``/inbox/`` directory for new session folders using the
``watchdog`` library. When a new directory is detected and remains stable
(no new files written for ``session_stability_delay`` seconds), it is
automatically ingested as a new session.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from watchdog.events import (
    DirCreatedEvent,
    FileCreatedEvent,
    FileSystemEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer

from app.core.config import get_settings
from app.core.logging import get_logger
from app.domain.ws_event import SessionDetectedEvent, SessionReadyEvent

logger = get_logger(__name__)


class _InboxEventHandler(FileSystemEventHandler):
    """Watchdog event handler that queues inbox events for async processing.

    Rather than calling async code directly (not safe from watchdog's thread),
    events are placed in an asyncio queue consumed by the watcher service.

    Attributes:
        queue: asyncio.Queue shared with the owning :class:`FileWatcherService`.
        loop: Reference to the event loop for thread-safe queue puts.
    """

    def __init__(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> None:
        """Initialise the handler.

        Args:
            queue: Queue to receive event paths.
            loop: The running asyncio event loop.
        """
        super().__init__()
        self.queue = queue
        self.loop = loop

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file or directory creation events.

        Args:
            event: The watchdog filesystem event.
        """
        asyncio.run_coroutine_threadsafe(
            self.queue.put(event.src_path),
            self.loop,
        )

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle modification events (re-queues path for stability timer reset).

        Args:
            event: The watchdog filesystem event.
        """
        asyncio.run_coroutine_threadsafe(
            self.queue.put(event.src_path),
            self.loop,
        )


class FileWatcherService:
    """Monitors ``/inbox/`` and triggers automatic session ingestion.

    Uses a stability timer: once a directory has been quiet for
    ``session_stability_delay`` seconds, it is considered "ready" and a
    session creation is triggered via the provided callback.

    Attributes:
        inbox_path: Root directory being watched.
        stability_delay: Seconds without new writes before a session is ingested.
        on_session_ready: Async callback invoked with the stable inbox path.
    """

    def __init__(
        self,
        on_session_ready: "Callable[[str], Awaitable[None]]",
        on_session_detected: Optional["Callable[[str], Awaitable[None]]"] = None,
        inbox_path: Optional[str] = None,
        stability_delay: Optional[float] = None,
    ) -> None:
        """Initialise the watcher service.

        Args:
            on_session_ready: Async callback invoked when a session directory
                has been stable for ``stability_delay`` seconds.
            on_session_detected: Optional async callback invoked immediately
                when a new directory is first detected.
            inbox_path: Override the inbox path from settings.
            stability_delay: Override the stability delay from settings.
        """
        settings = get_settings()
        self.inbox_path = Path(inbox_path or settings.inbox_path)
        self.stability_delay = stability_delay or settings.session_stability_delay
        self.on_session_ready = on_session_ready
        self.on_session_detected = on_session_detected

        self._observer: Optional[Observer] = None
        self._pending: dict[str, asyncio.TimerHandle] = {}
        self._queue: asyncio.Queue = asyncio.Queue()

    async def start(self) -> None:
        """Start the watchdog observer and the async event processing loop.

        The observer runs in a background thread; the processing loop runs
        as an asyncio task in the current event loop.
        """
        self.inbox_path.mkdir(parents=True, exist_ok=True)
        loop = asyncio.get_event_loop()

        handler = _InboxEventHandler(self._queue, loop)
        self._observer = Observer()
        self._observer.schedule(handler, str(self.inbox_path), recursive=True)
        self._observer.start()

        asyncio.create_task(self._process_events())
        logger.info("file_watcher_started", path=str(self.inbox_path))

    async def stop(self) -> None:
        """Stop the watchdog observer and cancel pending timers."""
        if self._observer is not None:
            self._observer.stop()
            self._observer.join()
            self._observer = None

        for handle in self._pending.values():
            handle.cancel()
        self._pending.clear()

        logger.info("file_watcher_stopped")

    async def _process_events(self) -> None:
        """Consume the event queue and manage per-directory stability timers.

        For each path received, determines the session root directory,
        cancels any existing stability timer for it, and schedules a new one.
        """
        while True:
            path_str: str = await self._queue.get()
            session_root = self._resolve_session_root(path_str)
            if session_root is None:
                continue

            session_str = str(session_root)

            # First detection: notify client immediately
            if session_str not in self._pending and self.on_session_detected:
                asyncio.create_task(self.on_session_detected(session_str))
                logger.info("session_detected", path=session_str)

            # Reset stability timer
            if session_str in self._pending:
                self._pending[session_str].cancel()

            loop = asyncio.get_event_loop()
            self._pending[session_str] = loop.call_later(
                self.stability_delay,
                lambda p=session_str: asyncio.create_task(self._on_stable(p)),
            )

    async def _on_stable(self, session_path: str) -> None:
        """Called when a session directory has been stable for the configured delay.

        Args:
            session_path: Absolute path of the stable session directory.
        """
        self._pending.pop(session_path, None)
        logger.info("session_stable", path=session_path)
        try:
            await self.on_session_ready(session_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("session_ingest_failed", path=session_path, error=str(exc))

    def _resolve_session_root(self, path_str: str) -> Optional[Path]:
        """Determine the session root directory from any path inside it.

        The session root is defined as the direct child of ``inbox_path``
        that contains the changed file/directory.

        Args:
            path_str: Absolute path of a changed file or directory.

        Returns:
            The session root :class:`~pathlib.Path`, or ``None`` if the path
            is outside the inbox or is the inbox itself.
        """
        try:
            path = Path(path_str).resolve()
            inbox = self.inbox_path.resolve()
            rel = path.relative_to(inbox)
            parts = rel.parts
            if not parts:
                return None
            return inbox / parts[0]
        except (ValueError, OSError):
            return None
