"""Acquisition-time extraction from camera RAW and FITS light frames.

The pipeline tags each :class:`AstroSession` with the timestamp of the
*earliest* light frame so the UI can date sessions to the actual sky run
rather than to the moment the files happened to land in the inbox.

Two readers are supported:

* RAW files (NEF / CR2 / ARW / DNG) — parsed with :mod:`exifread` (pure
  Python, no native deps), looking up ``EXIF DateTimeOriginal`` or
  ``Image DateTime`` as a fallback.
* FITS files — parsed with :func:`astropy.io.fits.getheader`, reading the
  standard ``DATE-OBS`` card (preferred) or ``DATE`` as a last resort.

All readers return ``None`` on any failure (missing card, malformed value,
unreadable file).  A bad frame must never block ingestion; the caller
simply gets a ``None`` ``acquired_at`` and the UI falls back to
``created_at``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import structlog

logger = structlog.get_logger(__name__)


_RAW_SUFFIXES = {".nef", ".cr2", ".cr3", ".arw", ".dng", ".raf", ".rw2", ".orf"}
_FITS_SUFFIXES = {".fit", ".fits", ".fts"}


def _parse_exif_datetime(value: str) -> Optional[datetime]:
    """Parse an EXIF datetime string ``"YYYY:MM:DD HH:MM:SS"``.

    EXIF stores naive local-time strings.  We tag them as UTC because the
    DB column is timezone-aware and we have no reliable per-file zone
    information.  This is acceptable for display ordering: the relative
    order of frames within a session is preserved.
    """
    try:
        dt = datetime.strptime(value.strip(), "%Y:%m:%d %H:%M:%S")
    except (ValueError, AttributeError):
        return None
    return dt.replace(tzinfo=timezone.utc)


def _read_raw_acquired_at(path: Path) -> Optional[datetime]:
    """Extract ``DateTimeOriginal`` from a camera RAW file."""
    try:
        import exifread  # type: ignore[import-untyped]
    except ImportError:  # pragma: no cover — declared in pyproject
        logger.warning("exifread missing, cannot read RAW EXIF", path=str(path))
        return None

    try:
        with path.open("rb") as fh:
            tags = exifread.process_file(
                fh, details=False, stop_tag="EXIF DateTimeOriginal"
            )
    except Exception as exc:  # noqa: BLE001 — best-effort
        logger.debug("exifread failed", path=str(path), error=str(exc))
        return None

    for key in ("EXIF DateTimeOriginal", "Image DateTimeOriginal", "Image DateTime"):
        tag = tags.get(key)
        if tag is None:
            continue
        parsed = _parse_exif_datetime(str(tag))
        if parsed is not None:
            return parsed
    return None


def _read_fits_acquired_at(path: Path) -> Optional[datetime]:
    """Extract ``DATE-OBS`` (or ``DATE``) from a FITS primary header."""
    try:
        from astropy.io import fits
    except ImportError:  # pragma: no cover
        return None

    try:
        header = fits.getheader(str(path), ext=0)
    except Exception as exc:  # noqa: BLE001
        logger.debug("fits header read failed", path=str(path), error=str(exc))
        return None

    for card in ("DATE-OBS", "DATE"):
        raw = header.get(card)
        if not raw:
            continue
        text = str(raw).strip()
        # Try ISO 8601 first (most common: "2024-11-03T22:14:08.123")
        for parser in (
            lambda s: datetime.fromisoformat(s.replace("Z", "+00:00")),
            lambda s: datetime.strptime(s, "%Y-%m-%dT%H:%M:%S"),
            lambda s: datetime.strptime(s, "%Y-%m-%d"),
        ):
            try:
                dt = parser(text)
            except ValueError:
                continue
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
    return None


def extract_acquired_at(path: Path) -> Optional[datetime]:
    """Return the capture timestamp of a single light frame, or ``None``.

    Dispatches on file suffix; unknown suffixes return ``None`` silently.
    """
    suffix = path.suffix.lower()
    if suffix in _RAW_SUFFIXES:
        return _read_raw_acquired_at(path)
    if suffix in _FITS_SUFFIXES:
        return _read_fits_acquired_at(path)
    return None


def earliest_acquired_at(paths: Iterable[Path]) -> Optional[datetime]:
    """Return the earliest valid acquisition timestamp across ``paths``.

    Files that fail extraction are skipped silently.  Returns ``None``
    only when *no* frame yields a usable timestamp.
    """
    best: Optional[datetime] = None
    for path in paths:
        dt = extract_acquired_at(path)
        if dt is None:
            continue
        if best is None or dt < best:
            best = dt
    return best
