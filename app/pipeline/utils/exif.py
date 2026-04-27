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

from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Optional

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


# ─────────────────────── Capture metadata aggregator ──────────────────────
#
# In addition to ``acquired_at`` we extract a richer set of camera /
# exposure parameters and aggregate them across the light frames so the UI
# can display "ISO 800 · 240 s · f/5.6 · Canon EOS Ra · 600 mm" cards.
# Aggregation rules:
#   * categorical fields (camera_make/model, lens_model) → most frequent
#     value (mode); ties broken by insertion order.
#   * numeric fields (exposure_seconds, iso, f_number, focal_length_mm,
#     temperature_c) → arithmetic mean **iff** all observed values lie
#     within ±5 % of the mean; otherwise the field is set to None and a
#     ``mixed`` flag records that frames disagree.
#   * Frame count is always reported.
# Any individual frame failure is silently ignored — the aggregator never
# raises.


def _to_float(value: Any) -> Optional[float]:
    """Best-effort conversion of an EXIF tag value to a Python float.

    Handles :class:`fractions.Fraction`, :class:`exifread.utils.Ratio`
    (which exposes ``num`` / ``den`` attributes), ints, floats, and
    strings of the form ``"1/250"`` or ``"4.5"``.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Fraction):
        return float(value)
    # exifread.utils.Ratio
    num = getattr(value, "num", None)
    den = getattr(value, "den", None)
    if num is not None and den:
        try:
            return float(num) / float(den)
        except (TypeError, ZeroDivisionError):
            return None
    text = str(value).strip()
    if not text:
        return None
    if "/" in text:
        try:
            n, d = text.split("/", 1)
            return float(n) / float(d)
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return float(text)
    except ValueError:
        return None


def _first_tag(tags: dict, *names: str) -> Any:
    for name in names:
        v = tags.get(name)
        if v is not None:
            return v
    return None


def _read_raw_capture(path: Path) -> dict[str, Any]:
    """Return per-frame capture parameters for a single RAW file.

    Missing values are simply omitted from the returned dict.
    """
    try:
        import exifread  # type: ignore[import-untyped]
    except ImportError:  # pragma: no cover
        return {}
    try:
        with path.open("rb") as fh:
            tags = exifread.process_file(fh, details=False)
    except Exception as exc:  # noqa: BLE001
        logger.debug("exifread capture failed", path=str(path), error=str(exc))
        return {}

    out: dict[str, Any] = {}
    iso_tag = _first_tag(tags, "EXIF ISOSpeedRatings", "EXIF PhotographicSensitivity")
    if iso_tag is not None:
        # ISO can appear as a list-like ratio.  Take the first integer-ish.
        iso_val = _to_float(iso_tag.values[0] if hasattr(iso_tag, "values") and iso_tag.values else iso_tag)
        if iso_val is not None:
            out["iso"] = int(iso_val)

    exp = _to_float(_first_tag(tags, "EXIF ExposureTime"))
    if exp is not None:
        out["exposure_seconds"] = exp

    fn = _to_float(_first_tag(tags, "EXIF FNumber"))
    if fn is not None:
        out["f_number"] = fn

    fl = _to_float(_first_tag(tags, "EXIF FocalLength"))
    if fl is not None:
        out["focal_length_mm"] = fl

    make = _first_tag(tags, "Image Make")
    if make is not None:
        out["camera_make"] = str(make).strip()
    model = _first_tag(tags, "Image Model")
    if model is not None:
        out["camera_model"] = str(model).strip()
    lens = _first_tag(tags, "EXIF LensModel", "MakerNote LensModel")
    if lens is not None:
        out["lens_model"] = str(lens).strip()
    return out


def _read_fits_capture(path: Path) -> dict[str, Any]:
    """Return per-frame capture parameters for a single FITS file."""
    try:
        from astropy.io import fits
    except ImportError:  # pragma: no cover
        return {}
    try:
        header = fits.getheader(str(path), ext=0)
    except Exception as exc:  # noqa: BLE001
        logger.debug("fits capture header failed", path=str(path), error=str(exc))
        return {}

    out: dict[str, Any] = {}
    exp = header.get("EXPTIME") or header.get("EXPOSURE")
    if exp is not None:
        v = _to_float(exp)
        if v is not None:
            out["exposure_seconds"] = v
    iso = header.get("ISO") or header.get("GAIN")
    if iso is not None:
        v = _to_float(iso)
        if v is not None:
            out["iso"] = int(v) if header.get("ISO") is not None else None
            if out["iso"] is None:
                out.pop("iso")
            out["gain"] = v
    fn = header.get("APERTURE") or header.get("FNUMBER")
    if fn is not None:
        v = _to_float(fn)
        if v is not None:
            out["f_number"] = v
    fl = header.get("FOCALLEN") or header.get("FOCAL")
    if fl is not None:
        v = _to_float(fl)
        if v is not None:
            out["focal_length_mm"] = v
    instr = header.get("INSTRUME")
    if instr:
        out["camera_model"] = str(instr).strip()
    tele = header.get("TELESCOP")
    if tele:
        out["telescope"] = str(tele).strip()
    flt = header.get("FILTER")
    if flt:
        out["filter"] = str(flt).strip()
    temp = header.get("CCD-TEMP") or header.get("CCDTEMP")
    if temp is not None:
        v = _to_float(temp)
        if v is not None:
            out["temperature_c"] = v
    return out


def _extract_capture(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in _RAW_SUFFIXES:
        return _read_raw_capture(path)
    if suffix in _FITS_SUFFIXES:
        return _read_fits_capture(path)
    return {}


_NUMERIC_FIELDS = (
    "exposure_seconds",
    "iso",
    "f_number",
    "focal_length_mm",
    "temperature_c",
    "gain",
)
_CATEGORICAL_FIELDS = (
    "camera_make",
    "camera_model",
    "lens_model",
    "telescope",
    "filter",
)


def _aggregate_numeric(values: list[float]) -> tuple[Optional[float], bool]:
    """Return ``(mean, mixed)``.

    ``mixed`` is True when individual values diverge by more than ±5 %
    from the mean; in that case the caller should set the field to None.
    """
    if not values:
        return None, False
    mean = sum(values) / len(values)
    if mean == 0:
        # All zero → not really a meaningful value, but coherent.
        return mean, False
    tol = abs(mean) * 0.05
    mixed = any(abs(v - mean) > tol for v in values)
    return mean, mixed


def _aggregate_categorical(values: list[str]) -> Optional[str]:
    if not values:
        return None
    counter = Counter(values)
    most_common, _count = counter.most_common(1)[0]
    return most_common


def extract_capture_metadata(paths: Iterable[Path]) -> dict[str, Any]:
    """Aggregate camera + exposure parameters across light frames.

    Returns a JSON-serialisable dict suitable for persisting on
    :class:`AstroSession.capture_metadata`.  Always returns a dict (may
    be empty if every frame failed extraction).
    """
    paths = list(paths)
    per_frame: list[dict[str, Any]] = []
    for path in paths:
        try:
            data = _extract_capture(path)
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.debug("capture extract failed", path=str(path), error=str(exc))
            continue
        if data:
            per_frame.append(data)

    if not per_frame:
        return {"frame_count": len(paths)}

    out: dict[str, Any] = {"frame_count": len(paths), "with_metadata": len(per_frame)}

    for field in _NUMERIC_FIELDS:
        values = [
            float(d[field])
            for d in per_frame
            if d.get(field) is not None
        ]
        if not values:
            continue
        mean, mixed = _aggregate_numeric(values)
        if mean is None:
            continue
        if mixed:
            # Surface the range so the UI can label it (e.g. "120-300 s").
            out[field] = None
            out[f"{field}_min"] = min(values)
            out[f"{field}_max"] = max(values)
        else:
            # Round ISO / gain to int when applicable.
            if field in ("iso",):
                out[field] = int(round(mean))
            else:
                out[field] = round(mean, 3)

    for field in _CATEGORICAL_FIELDS:
        values = [
            str(d[field])
            for d in per_frame
            if d.get(field)
        ]
        agg = _aggregate_categorical(values)
        if agg is not None:
            out[field] = agg

    # Total integration is convenient for the cartouche even when exposures vary.
    exp_values = [
        float(d["exposure_seconds"])
        for d in per_frame
        if d.get("exposure_seconds") is not None
    ]
    if exp_values:
        out["total_integration_seconds"] = round(sum(exp_values), 3)

    return out
