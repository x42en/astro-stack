"""ARQ Redis broker configuration and utilities.

Provides the ARQ ``RedisSettings`` singleton and a helper to enqueue tasks
without coupling calling code to ARQ internals.

Example:
    >>> broker = get_arq_redis_settings()
    >>> async with ArqRedis(broker) as arq:
    ...     job = await arq.enqueue_job("run_pipeline", job_id, session_id)
"""

from __future__ import annotations

from functools import lru_cache
from urllib.parse import urlparse

from arq.connections import ArqRedis, RedisSettings, create_pool

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_arq_redis_settings() -> RedisSettings:
    """Return the ARQ RedisSettings derived from the application config.

    Parsed from ``settings.redis_url``.

    Returns:
        :class:`arq.connections.RedisSettings` ready to pass to ARQ workers.
    """
    settings = get_settings()
    parsed = urlparse(settings.redis_url_str)

    return RedisSettings(
        host=parsed.hostname or "localhost",
        port=parsed.port or 6379,
        database=int(parsed.path.lstrip("/") or "0"),
        password=parsed.password,
    )


async def get_arq_pool() -> ArqRedis:
    """Create and return an ARQ connection pool for job enqueueing.

    This pool is distinct from the pub/sub connections in :mod:`~app.infrastructure.queue.events_bus`.
    The pool is ephemeral — callers should close it when done.

    Returns:
        An open :class:`~arq.connections.ArqRedis` pool connection.
    """
    redis_settings = get_arq_redis_settings()
    pool = await create_pool(redis_settings)
    logger.debug("arq_pool_created")
    return pool
