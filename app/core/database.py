"""Async PostgreSQL database engine and session factory.

Uses SQLModel (SQLAlchemy 2 + Pydantic) with the asyncpg driver.
The engine is created once at application startup via the lifespan context.

Example:
    >>> async with get_async_session() as session:
    ...     result = await session.exec(select(Session))
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings

# Module-level engine instance, initialised by ``init_db``.
_engine: AsyncEngine | None = None
_async_session_factory: Any = None


def init_db() -> None:
    """Initialise the async database engine and session factory.

    Must be called once at application startup (inside the lifespan handler).
    Uses connection pool settings tuned for a background-processing workload.
    """
    global _engine, _async_session_factory  # noqa: PLW0603

    settings = get_settings()

    _engine = create_async_engine(
        settings.database_url_str,
        echo=settings.log_level == "debug",
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
    )

    _async_session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )


async def create_all_tables() -> None:
    """Create all database tables registered in SQLAlchemy metadata.

    This is a development convenience. In production, prefer Alembic migrations.
    Imports all ORM models lazily so their table definitions are registered
    in the shared ``Base.metadata`` before the DDL is emitted.

    Raises:
        RuntimeError: If the engine has not been initialised.
    """
    if _engine is None:
        raise RuntimeError("Database engine not initialised. Call init_db() first.")

    # Lazy imports ensure circular-import safety; each module registers its
    # Table objects in the shared SQLAlchemy Base.metadata on import.
    import app.domain.session  # noqa: F401, PLC0415
    import app.domain.job  # noqa: F401, PLC0415
    import app.domain.profile  # noqa: F401, PLC0415

    from app.core.base import Base  # noqa: PLC0415

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session for use as a FastAPI dependency.

    The session is automatically closed after the request completes (or fails).

    Yields:
        An :class:`AsyncSession` bound to the global engine.

    Raises:
        RuntimeError: If the engine has not been initialised.

    Example:
        >>> @router.get("/items")
        ... async def list_items(db: AsyncSession = Depends(get_async_session)):
        ...     ...
    """
    if _async_session_factory is None:
        raise RuntimeError("Database session factory not initialised. Call init_db() first.")
    async with _async_session_factory() as session:
        yield session


async def close_db() -> None:
    """Dispose the connection pool and close the engine.

    Must be called at application shutdown (inside the lifespan handler).
    """
    global _engine  # noqa: PLW0603
    if _engine is not None:
        await _engine.dispose()
        _engine = None
