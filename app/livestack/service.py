"""High-level orchestration for live-stacking ingestion.

The service exposed here is the single entry point used by:

* the FastAPI endpoint ``POST /sessions/{id}/live-frames`` (after a
  frame has been written to disk),
* and the ARQ worker task ``livestack_ingest_frame`` that performs
  the heavy CPU work off the API event loop.

A worker holds a per-session lock (Redis ``SET NX EX``) so two
concurrent uploads of the same session are processed in arrival order
without colliding on the memmap accumulator.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np

from app.core.logging import get_logger
from app.domain.ws_event import (
    LiveStackFrameAcceptedEvent,
    LiveStackFrameRejectedEvent,
    LiveStackPreviewUpdatedEvent,
)
from app.infrastructure.queue.events_bus import EventBus
from app.infrastructure.storage.file_store import FileStore
from app.livestack.autostretch import StretchMethod, stretch_to_uint8
from app.livestack.preview import encode_preview_jpeg
from app.livestack.processors import (
    AlignmentError,
    FrameReadError,
    accumulate_running_mean,
    align_to_reference,
    open_or_create_accumulator,
    read_frame,
    remove_hot_pixels,
)
from app.livestack.state import LiveStackState, LiveStackStateRepository

logger = get_logger(__name__)


class LiveStackService:
    """Orchestrate ingestion, stacking and preview generation.

    The service is stateless beyond its dependencies; all per-session
    state is stored in :class:`LiveStackState` (Redis).

    Attributes:
        _store: Filesystem helper used to resolve live-mode paths.
        _state_repo: Redis-backed state repository.
        _events: Event bus used to push WebSocket notifications.
    """

    def __init__(
        self,
        file_store: FileStore,
        state_repo: LiveStackStateRepository,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        """Build a service bound to the provided collaborators.

        ``event_bus`` may be ``None`` for read-only / lifecycle calls
        (start, stop, get_state) that do not emit events. Ingestion
        requires a connected event bus.
        """
        self._store = file_store
        self._state_repo = state_repo
        self._events = event_bus

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def start(self, session_id: uuid.UUID) -> LiveStackState:
        """Mark a session as actively live-stacking.

        Initialises the on-disk live directory and persists a fresh
        :class:`LiveStackState` if none exists. Idempotent — calling
        it on an already-running session is a no-op.

        Args:
            session_id: UUID of the parent session.

        Returns:
            The current (possibly new) live-stack state.
        """
        self._store.ensure_live_dir(session_id)
        state = await self._state_repo.get(str(session_id))
        if state is None:
            state = LiveStackState(session_id=str(session_id))
        state.is_running = True
        await self._state_repo.save(state)
        return state

    async def stop(self, session_id: uuid.UUID) -> Optional[LiveStackState]:
        """Pause live-stacking. Persisted artefacts are kept on disk."""
        state = await self._state_repo.get(str(session_id))
        if state is None:
            return None
        state.is_running = False
        await self._state_repo.save(state)
        return state

    async def get_state(self, session_id: uuid.UUID) -> Optional[LiveStackState]:
        """Return the current state for ``session_id`` or ``None``."""
        return await self._state_repo.get(str(session_id))

    # ── Ingestion ─────────────────────────────────────────────────────────

    async def ingest_frame(
        self,
        session_id: uuid.UUID,
        frame_path: str | Path,
        *,
        method: StretchMethod = StretchMethod.MTF,
    ) -> LiveStackState:
        """Process a single frame into the running stack.

        The heavy pixel work is dispatched to a worker thread to avoid
        blocking the ARQ event loop. WebSocket events are emitted as
        each milestone (accept/reject + preview updated) completes.

        Args:
            session_id: Parent session UUID.
            frame_path: Absolute path to a raw or FITS frame on disk.
                Must already be written and stable.
            method: Stretch algorithm for the preview (only MTF for now).

        Returns:
            Updated :class:`LiveStackState` after the frame was merged
            (or after the rejection was recorded).
        """
        sid = str(session_id)
        state = await self._state_repo.get(sid) or LiveStackState(session_id=sid)
        if not state.is_running:
            # Auto-start: receiving a frame implies the user wants to stack.
            state.is_running = True

        frame_index = state.frame_count + state.rejected_count + 1

        try:
            new_state = await asyncio.to_thread(
                self._process_frame_sync,
                session_id,
                Path(frame_path),
                state,
                method,
            )
        except (FrameReadError, AlignmentError) as exc:
            state.rejected_count += 1
            await self._state_repo.save(state)
            if self._events is not None:
                await self._events.publish_session_event(
                    session_id,
                    LiveStackFrameRejectedEvent(
                        frame_index=frame_index,
                        reason=type(exc).__name__,
                        message=str(exc),
                    ),
                )
            logger.warning(
                "livestack_frame_rejected",
                session_id=sid,
                reason=type(exc).__name__,
                error=str(exc),
            )
            return state

        await self._state_repo.save(new_state)

        if self._events is not None:
            await self._events.publish_session_event(
                session_id,
                LiveStackFrameAcceptedEvent(
                    frame_index=frame_index,
                    frame_count=new_state.frame_count,
                    total_integration_seconds=new_state.total_integration_seconds,
                    fwhm=new_state.last_fwhm,
                    message=f"Frame {frame_index} stacked",
                ),
            )
            if new_state.shape is not None:
                h, w = new_state.shape[0], new_state.shape[1]
                await self._events.publish_session_event(
                    session_id,
                    LiveStackPreviewUpdatedEvent(
                        preview_generation=new_state.preview_generation,
                        frame_count=new_state.frame_count,
                        width=w,
                        height=h,
                    ),
                )
        return new_state

    # ── Sync hot path (runs in a worker thread) ───────────────────────────

    def _process_frame_sync(
        self,
        session_id: uuid.UUID,
        frame_path: Path,
        state: LiveStackState,
        method: StretchMethod,
    ) -> LiveStackState:
        """Synchronous stacking core. Returns the updated state.

        Side effects: updates the on-disk accumulator and rewrites the
        preview JPEG atomically.
        """
        live_dir = self._store.ensure_live_dir(session_id)
        frame = read_frame(frame_path)
        frame = remove_hot_pixels(frame)

        ref_path = self._store.live_reference_path(session_id)

        if state.shape is None or not ref_path.exists():
            # First frame becomes the reference and seeds the accumulator.
            np.save(ref_path, frame)
            shape = frame.shape
            accumulator = open_or_create_accumulator(
                self._store.live_accumulator_path(session_id),
                shape,
                np.dtype(np.float32),
            )
            accumulator[:] = frame.astype(np.float32, copy=False)
            accumulator.flush()
            aligned = frame
            fwhm: Optional[float] = None
            new_state = replace(
                state,
                accumulator_path=str(self._store.live_accumulator_path(session_id)),
                reference_path=str(ref_path),
                shape=shape,
                frame_count=1,
                last_fwhm=fwhm,
            )
        else:
            reference = np.load(ref_path)
            aligned, fwhm_val = align_to_reference(frame, reference)
            fwhm = None if np.isnan(fwhm_val) else float(fwhm_val)

            accumulator = open_or_create_accumulator(
                self._store.live_accumulator_path(session_id),
                state.shape,
                np.dtype(np.float32),
            )
            accumulate_running_mean(accumulator, aligned, state.frame_count)
            new_state = replace(
                state,
                frame_count=state.frame_count + 1,
                last_fwhm=fwhm,
            )

        # Generate the preview from the (now updated) accumulator.
        if method is not StretchMethod.MTF:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported stretch method: {method}")
        preview_uint8 = stretch_to_uint8(np.asarray(accumulator))
        encode_preview_jpeg(preview_uint8, self._store.live_preview_path(session_id))

        new_state.preview_generation += 1

        # Persist a copy of the accepted frame for later batch reprocessing.
        try:
            target = live_dir / "frames" / frame_path.name
            if not target.exists():
                target.write_bytes(frame_path.read_bytes())
        except OSError as exc:  # pragma: no cover
            logger.warning("livestack_frame_copy_failed", error=str(exc))

        return new_state
