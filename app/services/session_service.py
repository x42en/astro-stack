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
from app.domain.session import (
    AstroSession,
    InputFormat,
    SessionCreate,
    SessionMode,
    SessionStatus,
)
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

        # Best-effort acquisition timestamp + capture parameters from the
        # earliest light frame.  Failures are silent: the columns stay NULL
        # and the UI falls back gracefully.
        from app.pipeline.utils.exif import (
            earliest_acquired_at,
            extract_capture_metadata,
        )

        acquired_at = earliest_acquired_at(frames["lights"])
        capture_metadata = extract_capture_metadata(frames["lights"])

        session = AstroSession(
            name=name,
            inbox_path=inbox_path,
            status=SessionStatus.READY,
            input_format=input_format,
            frame_count_lights=len(frames["lights"]),
            frame_count_darks=len(frames["darks"]),
            frame_count_flats=len(frames["flats"]),
            frame_count_dark_flats=len(frames.get("dark_flats", [])),
            frame_count_bias=len(frames["bias"]),
            acquired_at=acquired_at,
            capture_metadata=capture_metadata or None,
        )

        created = await self._session_repo.create(session)
        logger.info(
            "session_created",
            session_id=str(created.id),
            name=name,
            lights=session.frame_count_lights,
        )
        return created

    async def create_session(
        self,
        payload: SessionCreate,
        owner_id: uuid.UUID | None = None,
    ) -> AstroSession:
        """Create a session with no frames yet (live or empty batch).

        Used by the REST API ``POST /sessions`` endpoint when the user
        starts a live-stacking session from the planner: no frames have
        been uploaded yet, but we want a persistent record so the
        worker can stream incremental events to a known session UUID.

        Live sessions enforce a per-owner uniqueness rule: an owner
        cannot have more than one active live session at any time. The
        check runs *before* persistence to keep the conflict semantics
        clean (no orphaned row on rejection).

        Args:
            payload: Validated :class:`~app.domain.session.SessionCreate`
                payload from the API. ``inbox_path`` is auto-allocated
                when missing.
            owner_id: UUID of the user creating the session. Mandatory
                for live sessions (uniqueness rule); optional for batch
                sessions to preserve backwards compatibility with the
                anonymous/legacy upload flow.

        Returns:
            The freshly persisted :class:`AstroSession`.

        Raises:
            ConflictException: When the owner already has an active live
                session.
        """
        from app.core.config import get_settings  # noqa: PLC0415

        if payload.mode == SessionMode.LIVE and owner_id is not None:
            existing = await self.get_active_live_session(owner_id)
            if existing is not None:
                raise ConflictException(
                    ErrorCode.SESS_ALREADY_PROCESSING,
                    "A live session is already active for this user.",
                    details={"active_session_id": str(existing.id)},
                )

        session_id = uuid.uuid4()
        settings = get_settings()
        inbox_path = payload.inbox_path or str(
            Path(settings.inbox_path) / str(session_id)
        )

        session = AstroSession(
            id=session_id,
            name=payload.name,
            inbox_path=inbox_path,
            status=SessionStatus.READY if payload.mode == SessionMode.LIVE else SessionStatus.PENDING,
            input_format=None,
            mode=payload.mode.value,
            owner_id=owner_id,
            object_name=payload.object_name,
            target_ra=payload.target_ra,
            target_dec=payload.target_dec,
            acquired_at=payload.acquired_at,
        )
        created = await self._session_repo.create(session)

        # Pre-create the inbox directory so live frame uploads can write
        # immediately without a TOCTOU race.
        Path(inbox_path).mkdir(parents=True, exist_ok=True)

        logger.info(
            "session_created_empty",
            session_id=str(created.id),
            mode=payload.mode.value,
            owner_id=str(owner_id) if owner_id else None,
            name=payload.name,
        )
        return created

    async def get_active_live_session(
        self,
        owner_id: uuid.UUID,
    ) -> AstroSession | None:
        """Return the currently-active live session for ``owner_id`` or None.

        "Active" means: ``mode == LIVE`` AND ``status NOT IN (COMPLETED,
        FAILED, CANCELLED)``. The most-recently-created match wins if
        several rows somehow satisfy the predicate (defensive — the
        creation path enforces uniqueness).
        """
        return await self._session_repo.find_active_live_for_owner(owner_id)

    async def terminate_session(
        self,
        session_id: uuid.UUID,
        owner_id: uuid.UUID | None = None,
    ) -> AstroSession:
        """Mark a session as completed (used to close a live session).

        When ``owner_id`` is provided the call is rejected (404, to
        avoid leaking the existence of other users' sessions) if the
        session belongs to a different owner. Sessions with NULL
        owner_id are accessible to anyone for backwards compatibility.
        """
        session = await self.get_or_404(session_id)
        if (
            owner_id is not None
            and session.owner_id is not None
            and session.owner_id != owner_id
        ):
            raise NotFoundException(
                ErrorCode.SESS_NOT_FOUND,
                f"Session '{session_id}' not found.",
                details={"session_id": str(session_id)},
            )
        updated = await self._session_repo.update(
            session_id,
            {"status": SessionStatus.COMPLETED.value},
        )
        logger.info("session_terminated", session_id=str(session_id))
        return updated  # type: ignore[return-value]

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
        search: str | None = None,
        owner_id: uuid.UUID | None = None,
    ) -> list[AstroSession]:
        """List all sessions, optionally filtered by status and/or name.

        Args:
            offset: Pagination offset.
            limit: Maximum number of results.
            status: Optional status filter.
            search: Optional name substring filter (case-insensitive).
            owner_id: When set, restrict the listing to sessions owned
                by this user. Sessions with NULL owner_id are excluded
                from the result.

        Returns:
            Ordered list of sessions.
        """
        if status is not None:
            return await self._session_repo.list_by_status(
                status, offset, limit, search, owner_id=owner_id,
            )
        return await self._session_repo.list_all_ordered(
            offset, limit, search, owner_id=owner_id,
        )

    async def count_sessions(
        self,
        status: SessionStatus | None = None,
        search: str | None = None,
        owner_id: uuid.UUID | None = None,
    ) -> int:
        """Return the total count of sessions matching the given filters.

        Args:
            status: Optional status filter.
            search: Optional name substring filter (case-insensitive).
            owner_id: When set, restrict the count to sessions owned by
                this user.

        Returns:
            Total matching session count.
        """
        if status is not None:
            return await self._session_repo.count_by_status(status, search, owner_id=owner_id)
        return await self._session_repo.count_all(search, owner_id=owner_id)

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
