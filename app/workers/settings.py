"""ARQ WorkerSettings class for worker process configuration.

The ``WorkerSettings`` class is discovered by the ARQ CLI when launching
workers with ``python -m arq app.workers.settings.WorkerSettings``.
"""

from __future__ import annotations

from arq.connections import RedisSettings

from app.core.config import get_settings
from app.infrastructure.queue.broker import get_arq_redis_settings
from app.workers.tasks import run_pipeline, shutdown, startup


class WorkerSettings:
    """ARQ worker settings class.

    Attributes:
        functions: List of task functions exposed to ARQ.
        on_startup: Coroutine called once when the worker process starts.
        on_shutdown: Coroutine called once when the worker process stops.
        redis_settings: Redis connection settings for ARQ.
        max_jobs: Maximum number of concurrent jobs per worker process.
        job_timeout: Maximum seconds a single job may run.
        keep_result: Seconds to retain job results in Redis.
        retry_jobs: Whether to automatically retry failed jobs.
    """

    functions = [run_pipeline]
    on_startup = startup
    on_shutdown = shutdown

    @property
    def redis_settings(self) -> RedisSettings:
        """Return ARQ Redis settings derived from application config.

        Returns:
            :class:`arq.connections.RedisSettings` instance.
        """
        return get_arq_redis_settings()

    max_jobs: int = 2  # One per GPU; set to 1 for single GPU
    job_timeout: int = 3600  # 1 hour max per job
    keep_result: int = 86400  # Keep results for 24 hours
    retry_jobs: bool = False  # Retry is handled by the orchestrator internally
