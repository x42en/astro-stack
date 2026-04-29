"""Live-stacking pipeline for incremental astrophotography sessions.

Public surface
--------------
- :class:`~app.livestack.service.LiveStackService` orchestrates ingestion
  of a single frame: read \u2192 align \u2192 accumulate \u2192 autostretch \u2192 JPEG.
- :class:`~app.livestack.state.LiveStackState` carries the persistent
  state of a live session (reference frame, accumulator path,
  preview generation counter).
- :func:`~app.livestack.autostretch.apply_mtf_autostretch` produces the
  display-ready 8-bit image fed to the JPEG encoder.

The full pipeline is async-friendly: heavy CPU work is wrapped in
``asyncio.to_thread`` by the service layer so the ARQ event loop stays
responsive.
"""

from __future__ import annotations

from app.livestack.autostretch import StretchMethod, apply_mtf_autostretch
from app.livestack.service import LiveStackService
from app.livestack.state import LiveStackState

__all__ = [
    "LiveStackService",
    "LiveStackState",
    "StretchMethod",
    "apply_mtf_autostretch",
]
