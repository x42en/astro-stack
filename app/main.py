"""FastAPI application factory and lifecycle management.

This module creates the FastAPI application instance, registers middleware,
mounts API routers, and manages startup/shutdown lifecycle events including
database initialisation, event bus connections, and file watcher startup.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.middleware.error_handler import register_error_handlers
from app.api.v1.catalog import router as catalog_router
from app.api.v1.gallery import router as gallery_router
from app.api.v1.jobs import router as jobs_router
from app.api.v1.profiles import router as profiles_router
from app.api.v1.sessions import router as sessions_router
from app.api.v1.ws.manager import WebSocketConnectionManager
from app.api.v1.ws.router import router as ws_router
from app.core.config import get_settings
from app.core.database import close_db, create_all_tables, init_db
from app.core.logging import configure_logging, get_logger
from app.infrastructure.queue.events_bus import EventBus
from app.services.watcher_service import FileWatcherService

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager.

    Handles startup and shutdown of all long-lived resources:
    - Logging configuration
    - Database engine + table creation
    - Redis event bus
    - WebSocket connection manager
    - File watcher service

    Args:
        app: The FastAPI application instance.

    Yields:
        Nothing — control is passed back to FastAPI during the app's lifetime.
    """
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info("astro_stack_starting", version=settings.app_version)

    # ── Database ──────────────────────────────────────────────────────────────
    init_db()
    await create_all_tables()
    logger.info("database_ready")

    # ── Event bus ─────────────────────────────────────────────────────────────
    event_bus = EventBus(redis_url=settings.redis_url_str)
    await event_bus.connect()
    app.state.event_bus = event_bus
    logger.info("event_bus_ready")

    # ── WebSocket manager ─────────────────────────────────────────────────────
    ws_manager = WebSocketConnectionManager(event_bus)
    app.state.ws_manager = ws_manager
    logger.info("websocket_manager_ready")

    # ── File watcher ──────────────────────────────────────────────────────────
    async def on_session_ready(inbox_path: str) -> None:
        """Auto-ingest a stable session directory.

        Args:
            inbox_path: Absolute path of the stable session directory.
        """
        from sqlalchemy.ext.asyncio import AsyncSession  # noqa: PLC0415
        from app.core.database import get_async_session  # noqa: PLC0415
        from app.services.session_service import SessionService  # noqa: PLC0415

        async for db in get_async_session():
            try:
                service = SessionService(db)
                session = await service.create_from_path(inbox_path)
                logger.info("session_auto_ingested", session_id=str(session.id), path=inbox_path)

                from app.domain.ws_event import SessionReadyEvent  # noqa: PLC0415
                import uuid  # noqa: PLC0415

                event = SessionReadyEvent(
                    session_id=session.id,
                    frame_count_lights=session.frame_count_lights,
                    frame_count_darks=session.frame_count_darks,
                    frame_count_flats=session.frame_count_flats,
                    frame_count_bias=session.frame_count_bias,
                    input_format=session.input_format if session.input_format else "fits",
                )
                await event_bus.publish_session_event(session.id, event)
            except Exception as exc:  # noqa: BLE001
                logger.error("session_auto_ingest_failed", path=inbox_path, error=str(exc))

    watcher = FileWatcherService(on_session_ready=on_session_ready)
    await watcher.start()
    app.state.watcher = watcher
    logger.info("file_watcher_ready", path=settings.inbox_path)

    logger.info("astro_stack_ready")
    yield  # ── Application running ──────────────────────────────────────

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("astro_stack_shutting_down")
    await watcher.stop()
    await event_bus.disconnect()
    await close_db()
    logger.info("astro_stack_stopped")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        A fully configured :class:`~fastapi.FastAPI` instance.
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Automated astrophotography processing pipeline with real-time "
            "event-driven API. Supports FITS/RAW input, Siril pre-processing, "
            "ASTAP plate solving, GraXpert gradient removal, and Cosmic Clarity "
            "AI denoise/sharpen/super-resolution."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Error handlers ────────────────────────────────────────────────────────
    register_error_handlers(app)

    # ── API routers ───────────────────────────────────────────────────────────
    app.include_router(sessions_router, prefix="/api/v1")
    app.include_router(jobs_router, prefix="/api/v1")
    app.include_router(profiles_router, prefix="/api/v1")
    app.include_router(catalog_router, prefix="/api/v1")
    app.include_router(gallery_router, prefix="/api/v1")
    app.include_router(ws_router)

    # ── Health check ──────────────────────────────────────────────────────────
    # Exposed on two paths:
    #   /health        — used by Docker HEALTHCHECK and Traefik infra probes
    #   /api/v1/health — used by the frontend Settings page (axios baseURL = /api/v1)
    def _health_body() -> dict:
        return {
            "status": "ok",
            "version": settings.app_version,
            "app_name": settings.app_name,
        }

    @app.get("/health", tags=["health"], summary="Health check", include_in_schema=False)
    async def health_check() -> dict:
        return _health_body()

    @app.get("/api/v1/health", tags=["health"], summary="Health check (v1)")
    async def health_check_v1() -> dict:
        """Return application health status.

        Returns:
            Dict with ``status``, ``version``, and ``app_name``.
        """
        return _health_body()

    return app


# Module-level app instance for Uvicorn
app = create_app()
