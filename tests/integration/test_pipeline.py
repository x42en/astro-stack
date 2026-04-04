"""Integration tests for the automated pipeline (inbox -> session -> job).

These tests verify the core end-to-end flow:
1. File watcher detects a new session directory
2. Session is created in the database with frame discovery
3. Job can be created to process the session

Note: Requires PostgreSQL and Redis to run. Use the docker-compose stack
for full integration testing.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from app.core.base import Base
from app.core.config import Settings
from app.domain.session import AstroSession, SessionStatus
from app.infrastructure.repositories.session_repo import SessionRepository
from app.infrastructure.storage.file_store import FileStore
from app.services.session_service import SessionService


@pytest.fixture
def mock_settings(tmp_path: Path) -> Settings:
    """Create mock settings with temporary directories."""
    return Settings(
        app_name="Astro-Stack-Test",
        app_version="0.1.0",
        database_url="sqlite+aiosqlite:///:memory:",
        redis_url="redis://localhost:6379/0",
        inbox_path=str(tmp_path / "inbox"),
        sessions_path=str(tmp_path / "sessions"),
        output_path=str(tmp_path / "output"),
        models_path=str(tmp_path / "models"),
        log_level="warning",
        session_stability_delay=0.1,
    )


@pytest.fixture
async def db_session(mock_settings: Settings) -> AsyncGenerator[AsyncSession, None]:
    """Create an in-memory SQLite database session for testing."""
    engine = create_async_engine(mock_settings.database_url, echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session

    await engine.dispose()


@pytest.fixture
def session_inbox(tmp_path: Path) -> Path:
    """Create a valid session folder structure in the inbox."""
    session = tmp_path / "inbox" / "2024-03-15_M42"
    (session / "darks").mkdir(parents=True)
    (session / "flats").mkdir(parents=True)
    (session / "bias").mkdir(parents=True)
    (session / "lights").mkdir(parents=True)

    for i in range(3):
        (session / "darks" / f"dark_{i:03d}.fits").touch()
        (session / "flats" / f"flat_{i:03d}.fits").touch()
        (session / "lights" / f"light_{i:03d}.cr2").touch()
    (session / "bias" / "bias_001.fits").touch()

    return session


class TestSessionCreation:
    """Tests for session creation from inbox paths."""

    @pytest.mark.asyncio
    async def test_create_session_discovers_frames(
        self,
        db_session: AsyncSession,
        session_inbox: Path,
    ) -> None:
        """Creating a session should discover and count all frame types."""
        service = SessionService(db_session)

        session = await service.create_from_path(str(session_inbox))

        assert session.name == "2024-03-15_M42"
        assert session.status == SessionStatus.READY
        assert session.frame_count_lights == 3
        assert session.frame_count_darks == 3
        assert session.frame_count_flats == 3
        assert session.frame_count_bias == 1

    @pytest.mark.asyncio
    async def test_create_session_detects_input_format(
        self,
        db_session: AsyncSession,
        session_inbox: Path,
    ) -> None:
        """Creating a session should detect raw_dslr format from .cr2 files."""
        service = SessionService(db_session)

        session = await service.create_from_path(str(session_inbox))

        assert session.input_format is not None
        assert session.input_format.value == "raw_dslr"

    @pytest.mark.asyncio
    async def test_duplicate_session_raises_conflict(
        self,
        db_session: AsyncSession,
        session_inbox: Path,
    ) -> None:
        """Creating a session for an already-existing inbox path should raise ConflictException."""
        service = SessionService(db_session)

        await service.create_from_path(str(session_inbox))

        with pytest.raises(Exception) as exc_info:
            await service.create_from_path(str(session_inbox))

        assert (
            "already exists" in str(exc_info.value).lower()
            or "conflict" in str(exc_info.value).lower()
        )

    @pytest.mark.asyncio
    async def test_session_without_lights_raises_error(
        self,
        db_session: AsyncSession,
        tmp_path: Path,
    ) -> None:
        """Creating a session with no light frames should raise an error."""
        empty_session = tmp_path / "inbox" / "empty"
        empty_session.mkdir(parents=True)
        (empty_session / "darks").mkdir()

        service = SessionService(db_session)

        with pytest.raises(Exception) as exc_info:
            await service.create_from_path(str(empty_session))

        assert "light" in str(exc_info.value).lower()


class TestSessionRepository:
    """Tests for session repository CRUD operations."""

    @pytest.mark.asyncio
    async def test_get_session_by_inbox_path(
        self,
        db_session: AsyncSession,
        session_inbox: Path,
    ) -> None:
        """Repository should find a session by its inbox path."""
        service = SessionService(db_session)
        created = await service.create_from_path(str(session_inbox))

        repo = SessionRepository(db_session)
        found = await repo.get_by_inbox_path(str(session_inbox))

        assert found is not None
        assert found.id == created.id

    @pytest.mark.asyncio
    async def test_list_sessions_ordered_by_created_at(
        self,
        db_session: AsyncSession,
        tmp_path: Path,
    ) -> None:
        """List should return sessions ordered by creation date (newest first)."""
        service = SessionService(db_session)

        s1 = await service.create_from_path(str(tmp_path / "inbox" / "session1"))
        s2 = await service.create_from_path(str(tmp_path / "inbox" / "session2"))

        repo = SessionRepository(db_session)
        sessions = await repo.list_all_ordered()

        assert len(sessions) == 2
        assert sessions[0].id == s2.id
        assert sessions[1].id == s1.id

    @pytest.mark.asyncio
    async def test_list_sessions_by_status(
        self,
        db_session: AsyncSession,
        session_inbox: Path,
    ) -> None:
        """List should filter sessions by status."""
        service = SessionService(db_session)
        await service.create_from_path(str(session_inbox))

        repo = SessionRepository(db_session)

        ready = await repo.list_by_status(SessionStatus.READY)
        assert len(ready) == 1

        processing = await repo.list_by_status(SessionStatus.PROCESSING)
        assert len(processing) == 0


class TestFileStoreIntegration:
    """Integration tests for FileStore with real filesystem."""

    def test_discover_frames_in_session_folder(
        self,
        session_inbox: Path,
    ) -> None:
        """FileStore should discover all frame types in a session folder."""
        with patch("app.infrastructure.storage.file_store.get_settings") as mock_get:
            mock_settings = MagicMock()
            mock_settings.inbox_path = str(session_inbox.parent)
            mock_settings.sessions_path = str(session_inbox.parent / "sessions")
            mock_settings.output_path = str(session_inbox.parent / "output")
            mock_get.return_value = mock_settings

            store = FileStore()
            frames = store.discover_frames(session_inbox)

        assert len(frames["lights"]) == 3
        assert len(frames["darks"]) == 3
        assert len(frames["flats"]) == 3
        assert len(frames["bias"]) == 1

    def test_detect_input_format_fits(
        self,
        tmp_path: Path,
    ) -> None:
        """Should detect FITS format when only FITS files are present."""
        with patch("app.infrastructure.storage.file_store.get_settings") as mock_get:
            mock_settings = MagicMock()
            mock_settings.inbox_path = str(tmp_path)
            mock_settings.sessions_path = str(tmp_path)
            mock_settings.output_path = str(tmp_path)
            mock_get.return_value = mock_settings

            store = FileStore()
            frames = {
                "lights": [Path("a.fits"), Path("b.fits")],
                "darks": [],
                "flats": [],
                "bias": [],
            }
            assert store.detect_input_format(frames) == "fits"

    def test_detect_input_format_raw_dslr(
        self,
        tmp_path: Path,
    ) -> None:
        """Should detect RAW DSLR format when only RAW files are present."""
        with patch("app.infrastructure.storage.file_store.get_settings") as mock_get:
            mock_settings = MagicMock()
            mock_settings.inbox_path = str(tmp_path)
            mock_settings.sessions_path = str(tmp_path)
            mock_settings.output_path = str(tmp_path)
            mock_get.return_value = mock_settings

            store = FileStore()
            frames = {
                "lights": [Path("a.cr2"), Path("b.nef")],
                "darks": [],
                "flats": [],
                "bias": [],
            }
            assert store.detect_input_format(frames) == "raw_dslr"
