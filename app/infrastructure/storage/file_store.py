"""File system storage management for sessions, outputs, and temporary files.

Provides a :class:`FileStore` that abstracts path resolution, directory
creation, and cleanup for the three main storage areas:

- ``/inbox/``    — incoming session directories (read-only after detection)
- ``/sessions/`` — intermediate working files for active processing
- ``/output/``   — final processed files served to users

Example:
    >>> store = FileStore(settings)
    >>> work_dir = store.session_work_dir(session_id)
    >>> output_dir = store.session_output_dir(session_id)
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import aiofiles.os

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Recognised FITS file extensions
FITS_EXTENSIONS: frozenset[str] = frozenset({".fit", ".fits", ".fts"})

# Recognised DSLR RAW file extensions (non-exhaustive, LibRaw handles the rest)
RAW_DSLR_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".cr2",
        ".cr3",
        ".nef",
        ".nrw",
        ".arw",
        ".srf",
        ".sr2",
        ".dng",
        ".orf",
        ".rw2",
        ".pef",
        ".srw",
        ".raf",
    }
)

# Frame type detection by directory name (case-insensitive)
DARK_DIR_NAMES: frozenset[str] = frozenset({"darks", "dark"})
FLAT_DIR_NAMES: frozenset[str] = frozenset({"flats", "flat"})
BIAS_DIR_NAMES: frozenset[str] = frozenset({"bias", "biases", "offset", "offsets"})
LIGHT_DIR_NAMES: frozenset[str] = frozenset({"lights", "light", "raws", "raw", "subs", "frames"})


class FileStore:
    """Manages all file system paths and I/O operations for the pipeline.

    Attributes:
        inbox_root: Root directory watched for incoming sessions.
        sessions_root: Root directory for intermediate processing files.
        output_root: Root directory for final output files.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialise the store with path roots from settings.

        Args:
            settings: Optional settings override; defaults to the app singleton.
        """
        cfg = settings or get_settings()
        self.inbox_root = Path(cfg.inbox_path)
        self.sessions_root = Path(cfg.sessions_path)
        self.output_root = Path(cfg.output_path)

    # ── Working directory helpers ─────────────────────────────────────────────

    def session_work_dir(self, session_id: uuid.UUID) -> Path:
        """Return the working directory path for a session.

        Args:
            session_id: Session UUID.

        Returns:
            Path under ``/sessions/{session_id}/``.
        """
        return self.sessions_root / str(session_id)

    def session_output_dir(self, session_id: uuid.UUID) -> Path:
        """Return the output directory path for a session.

        Args:
            session_id: Session UUID.

        Returns:
            Path under ``/output/{session_id}/``.
        """
        return self.output_root / str(session_id)

    def ensure_work_dir(self, session_id: uuid.UUID) -> Path:
        """Create the working directory for a session if it does not exist.

        Args:
            session_id: Session UUID.

        Returns:
            The created (or existing) working directory path.
        """
        work_dir = self.session_work_dir(session_id)
        work_dir.mkdir(parents=True, exist_ok=True)
        for sub in ("process", "calibrated", "registered", "output"):
            (work_dir / sub).mkdir(exist_ok=True)
        return work_dir

    def ensure_output_dir(self, session_id: uuid.UUID) -> Path:
        """Create the output directory for a session if it does not exist.

        Args:
            session_id: Session UUID.

        Returns:
            The created (or existing) output directory path.
        """
        out_dir = self.session_output_dir(session_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    # ── Frame discovery ───────────────────────────────────────────────────────

    def discover_frames(
        self,
        session_inbox_path: Path,
    ) -> dict[str, list[Path]]:
        """Scan a session directory and categorise all frame files.

        Detects frame types first from subdirectory names, then falls back
        to FITS ``IMAGETYP`` header inspection (not implemented here — deferred
        to the session service).

        Args:
            session_inbox_path: Root path of the session inside ``/inbox/``.

        Returns:
            Dict with keys ``"lights"``, ``"darks"``, ``"flats"``, ``"bias"``
            each mapping to a list of discovered file paths.
        """
        result: dict[str, list[Path]] = {
            "lights": [],
            "darks": [],
            "flats": [],
            "bias": [],
        }

        if not session_inbox_path.is_dir():
            logger.warning("session_path_not_a_dir", path=str(session_inbox_path))
            return result

        for child in session_inbox_path.iterdir():
            if not child.is_dir():
                # Top-level files: try to classify as lights
                if _is_image_file(child):
                    result["lights"].append(child)
                continue

            dir_lower = child.name.lower()
            if dir_lower in DARK_DIR_NAMES:
                result["darks"].extend(_collect_images(child))
            elif dir_lower in FLAT_DIR_NAMES:
                result["flats"].extend(_collect_images(child))
            elif dir_lower in BIAS_DIR_NAMES:
                result["bias"].extend(_collect_images(child))
            elif dir_lower in LIGHT_DIR_NAMES:
                result["lights"].extend(_collect_images(child))
            else:
                # Unknown subdirectory: recurse one level as lights
                logger.debug(
                    "unknown_subdir_treated_as_lights",
                    path=str(child),
                )
                result["lights"].extend(_collect_images(child))

        logger.info(
            "frames_discovered",
            path=str(session_inbox_path),
            lights=len(result["lights"]),
            darks=len(result["darks"]),
            flats=len(result["flats"]),
            bias=len(result["bias"]),
        )
        return result

    def detect_input_format(self, frames: dict[str, list[Path]]) -> str:
        """Detect the predominant input file format from discovered frames.

        Args:
            frames: Output of :meth:`discover_frames`.

        Returns:
            One of ``"fits"``, ``"raw_dslr"``, or ``"mixed"``.
        """
        all_files = [f for files in frames.values() for f in files]
        if not all_files:
            return "fits"

        fits_count = sum(1 for f in all_files if f.suffix.lower() in FITS_EXTENSIONS)
        raw_count = sum(1 for f in all_files if f.suffix.lower() in RAW_DSLR_EXTENSIONS)

        if fits_count > 0 and raw_count == 0:
            return "fits"
        if raw_count > 0 and fits_count == 0:
            return "raw_dslr"
        return "mixed"

    # ── Cleanup ───────────────────────────────────────────────────────────────

    async def cleanup_work_dir(self, session_id: uuid.UUID) -> None:
        """Remove the working directory for a session after successful completion.

        Args:
            session_id: UUID of the session to clean up.
        """
        work_dir = self.session_work_dir(session_id)
        if work_dir.exists():
            await aiofiles.os.wrap(shutil.rmtree)(work_dir)
            logger.info("work_dir_cleaned", session_id=str(session_id))


# ── Private helpers ───────────────────────────────────────────────────────────


def _is_image_file(path: Path) -> bool:
    """Return ``True`` if the path has a recognised image file extension.

    Args:
        path: File path to check.

    Returns:
        ``True`` if FITS or RAW DSLR extension detected.
    """
    suffix = path.suffix.lower()
    return suffix in FITS_EXTENSIONS or suffix in RAW_DSLR_EXTENSIONS


def _collect_images(directory: Path) -> list[Path]:
    """Recursively collect all image files under a directory.

    Args:
        directory: Root directory to scan.

    Returns:
        Flat list of image file paths found anywhere under ``directory``.
    """
    return [p for p in directory.rglob("*") if p.is_file() and _is_image_file(p)]
