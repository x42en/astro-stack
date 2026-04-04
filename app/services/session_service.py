"""Session business logic service.

Orchestrates the creation, retrieval and status management of
:class:`~app.domain.session.AstroSession` records. Decouples the API
layer from direct repository access and enforces lifecycle rules.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import ConflictException, ErrorCode, NotFoundException
from app.core.logging import get_logger
from app.domain.session import AstroSession, InputFormat, SessionCreate, SessionStatus
from app.infrastructure.repositories.session_repo import SessionRepository
from app.infrastructure.storage.file_store import FileStore

logger = get_logger(__name__)


class SessionService:
    """Manages session lifecycle and frame inventory.

    Attributes:
        _session_repo: Repository for session persistence.
        _file_store: File system storage abstraction.
    """

    def __init__(self, db_session: AsyncSession) -> None:
        """Initialise the service.

        Args:
            db_session: Active database session injected by FastAPI dependency.
        """
        self._session_repo = SessionRepository(db_session)
        self._file_store = FileStore()

    async def create_from_path(self, inbox_path: str) -> AstroSession:
        """Create a session record from a detected inbox directory.

        Automatically discovers frame counts and input format.

        Args:
            inbox_path: Absolute path to the session's inbox directory.

        Returns:
            The created :class:`~app.domain.session.AstroSession`.

        Raises:
            ConflictException: If a session already exists for this path.
        """
        existing = await self._session_repo.get_by_inbox_path(inbox_path)
        if existing is not None:
            raise ConflictException(
                ErrorCode.SESS_ALREADY_PROCESSING,
                f"A session already exists for inbox path: {inbox_path}",
                details={"inbox_path": inbox_path},
            )

        inbox = Path(inbox_path)
        frames = self._file_store.discover_frames(inbox)
        input_format_str = self._file_store.detect_input_format(frames)
        input_format = InputFormat(input_format_str)

        if len(frames["lights"]) == 0:
            raise ConflictException(
                ErrorCode.SESS_NO_LIGHTS_FOUND,
                f"No light frames found in {inbox_path}",
                details={"inbox_path": inbox_path},
            )

        name = inbox.name or inbox_path

        session = AstroSession(
            name=name,
            inbox_path=inbox_path,
            status=SessionStatus.READY,
            input_format=input_format,
            frame_count_lights=len(frames["lights"]),
            frame_count_darks=len(frames["darks"]),
            frame_count_flats=len(frames["flats"]),
            frame_count_bias=len(frames["bias"]),
        )

        created = await self._session_repo.create(session)
        logger.info(
            "session_created",
            session_id=str(created.id),
            name=name,
            lights=session.frame_count_lights,
        )
        return created

    async def get_or_404(self, session_id: uuid.UUID) -> AstroSession:
        """Retrieve a session by ID or raise a 404 error.

        Args:
            session_id: Session UUID to look up.

        Returns:
            The found :class:`~app.domain.session.AstroSession`.

        Raises:
            NotFoundException: If the session does not exist.
        """
        session = await self._session_repo.get(session_id)
        if session is None:
            raise NotFoundException(
                ErrorCode.SESS_NOT_FOUND,
                f"Session '{session_id}' not found.",
                details={"session_id": str(session_id)},
            )
        return session

    async def list_sessions(
        self,
        offset: int = 0,
        limit: int = 100,
        status: SessionStatus | None = None,
    ) -> list[AstroSession]:
        """List all sessions, optionally filtered by status.

        Args:
            offset: Pagination offset.
            limit: Maximum number of results.
            status: Optional status filter.

        Returns:
            Ordered list of sessions.
        """
        if status is not None:
            return await self._session_repo.list_by_status(status, offset, limit)
        return await self._session_repo.list_all_ordered(offset, limit)

    async def update_plate_solve_result(
        self,
        session_id: uuid.UUID,
        ra: float | None,
        dec: float | None,
        object_name: str | None,
    ) -> AstroSession:
        """Persist plate-solving results to the session record.

        Args:
            session_id: Session UUID to update.
            ra: Right ascension in decimal degrees.
            dec: Declination in decimal degrees.
            object_name: Resolved astronomical object name.

        Returns:
            The updated session.

        Raises:
            NotFoundException: If session does not exist.
        """
        await self.get_or_404(session_id)
        return await self._session_repo.update(
            session_id,
            {"ra": ra, "dec": dec, "object_name": object_name},
        )  # type: ignore[return-value]
