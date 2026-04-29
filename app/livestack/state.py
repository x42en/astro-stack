"""Persistent state for the live-stacking pipeline.

A live session needs several pieces of state shared between the API
container and the worker(s):

* the path to the on-disk accumulator (running mean of aligned frames);
* the path to the reference frame used for alignment;
* counters (accepted / rejected frames, total integration time);
* a monotonic ``preview_generation`` used as a cache-busting key for
  the JPEG preview served over HTTP and broadcast over WebSocket.

We persist this state in Redis as a single hash per session
(``livestack:{session_id}``) so it survives worker restarts and is
visible to the API process for ``GET /live/state`` queries.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Optional

import redis.asyncio as aioredis

__all__ = ["LiveStackState", "LiveStackStateRepository"]


def _state_key(session_id: str) -> str:
    return f"livestack:{session_id}"


@dataclass(slots=True)
class LiveStackState:
    """Snapshot of a live-stacking session's state.

    Attributes:
        session_id: UUID of the parent astro session.
        is_running: ``True`` between explicit ``start`` and ``stop``
            calls. Frames received while ``False`` are still stored
            (write-through) but skipped by the live stacker.
        frame_count: Number of frames merged into the running stack.
        rejected_count: Number of frames rejected (alignment failure,
            decode error, ...).
        accumulator_path: Absolute path to the memmap holding the
            running mean. Populated lazily on first frame.
        reference_path: Path to the reference frame stored on disk
            (a copy of the first accepted frame). ``None`` until the
            first frame is ingested.
        shape: Shape of the running stack (``(H, W)`` or ``(H, W, 3)``)
            serialised as a tuple. ``None`` until the first frame.
        preview_generation: Monotonic counter incremented at every
            preview regeneration. Used as ETag / cache-busting key.
        last_fwhm: Last estimated star FWHM (pixels). ``None`` if not
            available.
        total_integration_seconds: Sum of per-frame exposure times when
            FITS headers provide them; otherwise ``None``.
    """

    session_id: str
    is_running: bool = False
    frame_count: int = 0
    rejected_count: int = 0
    accumulator_path: Optional[str] = None
    reference_path: Optional[str] = None
    shape: Optional[tuple[int, ...]] = None
    preview_generation: int = 0
    last_fwhm: Optional[float] = None
    total_integration_seconds: Optional[float] = None
    # Last stretch parameters computed for the preview, kept for
    # client-side display / re-stretching.
    last_stretch: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialise to a JSON string for Redis storage."""
        payload = asdict(self)
        # Tuples don't roundtrip through JSON — store as list.
        if payload["shape"] is not None:
            payload["shape"] = list(payload["shape"])
        return json.dumps(payload)

    @classmethod
    def from_json(cls, raw: str) -> "LiveStackState":
        """Deserialise from JSON, rebuilding the ``shape`` tuple."""
        data = json.loads(raw)
        if data.get("shape") is not None:
            data["shape"] = tuple(data["shape"])
        return cls(**data)


class LiveStackStateRepository:
    """Async Redis-backed CRUD for :class:`LiveStackState`.

    A thin wrapper that hides the Redis key naming scheme from the
    service layer.  All methods are awaitable.
    """

    def __init__(self, redis: aioredis.Redis) -> None:
        """Bind the repository to a configured Redis client.

        Args:
            redis: Async Redis connection (decoded responses recommended).
        """
        self._redis = redis

    async def get(self, session_id: str) -> Optional[LiveStackState]:
        """Return the persisted state for ``session_id`` or ``None``."""
        raw = await self._redis.get(_state_key(session_id))
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return LiveStackState.from_json(raw)

    async def save(self, state: LiveStackState) -> None:
        """Persist ``state`` (full overwrite)."""
        await self._redis.set(_state_key(state.session_id), state.to_json())

    async def delete(self, session_id: str) -> None:
        """Remove the state entry, e.g. when the session is deleted."""
        await self._redis.delete(_state_key(session_id))
