"""ARQ task definitions for background pipeline execution.

ARQ tasks are plain async functions decorated with the ``ctx`` context dict
that ARQ provides. The ``run_pipeline`` task is the main entry point called
by :class:`~app.services.job_service.JobService` when a pipeline is started.

Each task receives its own database session and event bus connection to
remain fully isolated from the API process.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

from arq import Retry
from arq.worker import Worker

from app.core.config import get_settings
from app.core.database import get_async_session
from app.core.errors import AstroStackException
from app.core.logging import get_logger
from app.domain.job import JobStatus
from app.domain.profile import ProcessingProfileConfig
from app.domain.session import SessionStatus
from app.domain.ws_event import SessionStatusEvent
from app.infrastructure.queue.events_bus import EventBus
from app.infrastructure.repositories.job_repo import JobRepository
from app.infrastructure.repositories.session_repo import SessionRepository
from app.infrastructure.storage.file_store import FileStore

logger = get_logger(__name__)


async def run_pipeline(
    ctx: dict[str, Any],
    job_id_str: str,
    session_id_str: str,
    profile_config_dict: dict[str, Any],
) -> dict[str, Any]:
    """ARQ task: execute the full astrophotography processing pipeline.

    Instantiates the :class:`~app.pipeline.orchestrator.PipelineOrchestrator`
    and runs all configured steps. Emits events via the Redis event bus
    throughout execution.

    Args:
        ctx: ARQ worker context (contains ``db_pool``, ``event_bus``, etc.).
        job_id_str: UUID string of the pipeline job.
        session_id_str: UUID string of the associated session.
        profile_config_dict: Serialised :class:`~app.domain.profile.ProcessingProfileConfig`.

    Returns:
        Dict of output file paths on success.

    Raises:
        Retry: If the failure is transient and a retry should be attempted.
    """
    job_id = uuid.UUID(job_id_str)
    session_id = uuid.UUID(session_id_str)
    profile_config = ProcessingProfileConfig(**profile_config_dict)

    # Determine GPU device for this worker based on environment variable
    settings = get_settings()
    gpu_device_index = os.environ.get("WORKER_GPU_DEVICE", "0")
    gpu_device = f"cuda:{gpu_device_index}"

    logger.info(
        "pipeline_task_started",
        job_id=job_id_str,
        session_id=session_id_str,
        gpu=gpu_device,
    )

    event_bus: EventBus = ctx["event_bus"]
    file_store = FileStore()

    async for db_session in get_async_session():
        try:
            from app.pipeline.orchestrator import PipelineOrchestrator  # noqa: PLC0415

            orchestrator = PipelineOrchestrator(
                job_id=job_id,
                session_id=session_id,
                profile_config=profile_config,
                event_bus=event_bus,
                db_session=db_session,
                gpu_device=gpu_device,
            )
            outputs = await orchestrator.run()
            logger.info("pipeline_task_completed", job_id=job_id_str, outputs=list(outputs.keys()))
            await file_store.cleanup_work_dir(session_id)
            return outputs

        except AstroStackException as exc:
            logger.error(
                "pipeline_task_failed",
                job_id=job_id_str,
                error_code=exc.error_code.value,
                message=exc.message,
            )
            # Cap ARQ-level retries so the session is eventually marked FAILED
            # rather than staying in Processing indefinitely. job_try is 1-based.
            _MAX_ARQ_TRIES = 3
            if exc.retryable and ctx.get("job_try", 1) < _MAX_ARQ_TRIES:
                # Notify the UI that the session is still alive but is about
                # to retry — without this the front-end shows "Processing"
                # silently for the entire 30 s defer, which looks like a hang.
                # ``job_status="retrying"`` lets the UI render an explicit
                # retry indicator while keeping ``new_status="processing"``
                # so the session doesn't move out of the active list.
                retrying_event = SessionStatusEvent(
                    session_id=session_id,
                    new_status="processing",
                    job_status="retrying",
                )
                await event_bus.publish_session_event(session_id, retrying_event)
                await event_bus.publish_broadcast(retrying_event)
                # Do NOT clean up: the orchestrator resumes from already-succeeded
                # steps (e.g. raw_conversion), so converted FITS in work_dir must
                # be preserved across the ARQ-level retry.
                raise Retry(defer=30) from exc
            # Retries exhausted or non-retryable: persist FAILED, notify, clean up
            await SessionRepository(db_session).update(
                session_id, {"status": SessionStatus.FAILED.value}
            )
            failed_event = SessionStatusEvent(
                session_id=session_id,
                new_status="failed",
                job_status="failed",
            )
            await event_bus.publish_session_event(session_id, failed_event)
            await event_bus.publish_broadcast(failed_event)
            await file_store.cleanup_work_dir(session_id)
            return {"error": exc.message, "error_code": exc.error_code.value}

        except Exception as exc:  # noqa: BLE001
            logger.exception("pipeline_task_unexpected_error", job_id=job_id_str)
            await SessionRepository(db_session).update(
                session_id, {"status": SessionStatus.FAILED.value}
            )
            failed_event = SessionStatusEvent(
                session_id=session_id,
                new_status="failed",
                job_status="failed",
            )
            await event_bus.publish_session_event(session_id, failed_event)
            await event_bus.publish_broadcast(failed_event)
            await file_store.cleanup_work_dir(session_id)
            return {"error": str(exc), "error_code": "SYS_INTERNAL_ERROR"}


# ── ARQ context lifecycle ─────────────────────────────────────────────────────


async def startup(ctx: dict[str, Any]) -> None:
    """Initialise shared resources for the worker process.

    Called once when the ARQ worker starts. Sets up database connections,
    Redis event bus, etc.

    Args:
        ctx: Mutable ARQ context dict — resources placed here are available
            in all task functions via ``ctx``.
    """
    from app.core.database import init_db  # noqa: PLC0415
    from app.core.logging import configure_logging  # noqa: PLC0415

    settings = get_settings()
    configure_logging(settings.log_level)
    init_db()

    event_bus = EventBus(redis_url=settings.redis_url_str)
    await event_bus.connect()
    ctx["event_bus"] = event_bus

    logger.info("arq_worker_started", redis=settings.redis_url_str)


async def shutdown(ctx: dict[str, Any]) -> None:
    """Clean up shared resources when the worker stops.

    Args:
        ctx: ARQ context dict with resources to close.
    """
    from app.core.database import close_db  # noqa: PLC0415

    event_bus: EventBus | None = ctx.get("event_bus")
    if event_bus:
        await event_bus.disconnect()

    await close_db()
    logger.info("arq_worker_stopped")
