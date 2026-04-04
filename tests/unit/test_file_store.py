"""Unit tests for the FileStore frame discovery logic."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from app.infrastructure.storage.file_store import FileStore, _is_image_file, _collect_images


class TestIsImageFile:
    """Tests for _is_image_file() helper."""

    @pytest.mark.parametrize("ext", [".fit", ".fits", ".fts"])
    def test_fits_extensions_are_images(self, ext: str, tmp_path: Path) -> None:
        """FITS file extensions should be recognised as images."""
        p = tmp_path / f"image{ext}"
        p.touch()
        assert _is_image_file(p) is True

    @pytest.mark.parametrize("ext", [".cr2", ".nef", ".arw", ".dng"])
    def test_raw_extensions_are_images(self, ext: str, tmp_path: Path) -> None:
        """DSLR RAW file extensions should be recognised as images."""
        p = tmp_path / f"image{ext}"
        p.touch()
        assert _is_image_file(p) is True

    @pytest.mark.parametrize("ext", [".jpg", ".png", ".txt", ".xml"])
    def test_other_extensions_not_images(self, ext: str, tmp_path: Path) -> None:
        """Non-image file extensions should return False."""
        p = tmp_path / f"file{ext}"
        p.touch()
        assert _is_image_file(p) is False


class TestDiscoverFrames:
    """Tests for FileStore.discover_frames() directory scanning."""

    def _make_session(self, tmp_path: Path) -> Path:
        """Create a session folder structure in tmp_path."""
        session = tmp_path / "2024-03-15_M42"
        (session / "darks").mkdir(parents=True)
        (session / "flats").mkdir(parents=True)
        (session / "bias").mkdir(parents=True)
        (session / "lights").mkdir(parents=True)

        for i in range(3):
            (session / "darks" / f"dark_{i:03d}.fits").touch()
            (session / "flats" / f"flat_{i:03d}.fits").touch()
            (session / "lights" / f"light_{i:03d}.fits").touch()
        (session / "bias" / "bias_001.fits").touch()

        return session

    def test_discovers_all_frame_types(self, tmp_path: Path) -> None:
        """All four frame types are detected with correct counts."""
        session = self._make_session(tmp_path)
        store = FileStore.__new__(FileStore)
        store.inbox_root = tmp_path
        store.sessions_root = tmp_path / "sessions"
        store.output_root = tmp_path / "output"

        frames = store.discover_frames(session)
        assert len(frames["lights"]) == 3
        assert len(frames["darks"]) == 3
        assert len(frames["flats"]) == 3
        assert len(frames["bias"]) == 1

    def test_empty_session_returns_empty_lists(self, tmp_path: Path) -> None:
        """An empty directory returns all empty frame lists."""
        session = tmp_path / "empty_session"
        session.mkdir()
        store = FileStore.__new__(FileStore)
        store.inbox_root = tmp_path
        store.sessions_root = tmp_path / "sessions"
        store.output_root = tmp_path / "output"

        frames = store.discover_frames(session)
        assert sum(len(v) for v in frames.values()) == 0

    def test_non_existent_path_returns_empty(self, tmp_path: Path) -> None:
        """A non-existent path returns empty frame lists without error."""
        store = FileStore.__new__(FileStore)
        store.inbox_root = tmp_path
        store.sessions_root = tmp_path / "sessions"
        store.output_root = tmp_path / "output"

        frames = store.discover_frames(tmp_path / "does_not_exist")
        assert all(len(v) == 0 for v in frames.values())


class TestDetectInputFormat:
    """Tests for FileStore.detect_input_format()."""

    def test_detects_fits(self, tmp_path: Path) -> None:
        """Returns 'fits' when all files are FITS."""
        store = FileStore.__new__(FileStore)
        store.inbox_root = tmp_path
        store.sessions_root = tmp_path
        store.output_root = tmp_path

        frames = {"lights": [Path("a.fits"), Path("b.fit")], "darks": [], "flats": [], "bias": []}
        assert store.detect_input_format(frames) == "fits"

    def test_detects_raw_dslr(self, tmp_path: Path) -> None:
        """Returns 'raw_dslr' when all files are RAW."""
        store = FileStore.__new__(FileStore)
        store.inbox_root = tmp_path
        store.sessions_root = tmp_path
        store.output_root = tmp_path

        frames = {"lights": [Path("a.cr2"), Path("b.nef")], "darks": [], "flats": [], "bias": []}
        assert store.detect_input_format(frames) == "raw_dslr"

    def test_detects_mixed(self, tmp_path: Path) -> None:
        """Returns 'mixed' when both FITS and RAW files are present."""
        store = FileStore.__new__(FileStore)
        store.inbox_root = tmp_path
        store.sessions_root = tmp_path
        store.output_root = tmp_path

        frames = {
            "lights": [Path("a.fits"), Path("b.cr2")],
            "darks": [],
            "flats": [],
            "bias": [],
        }
        assert store.detect_input_format(frames) == "mixed"
