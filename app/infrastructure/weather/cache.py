"""Redis-backed JSON cache for weather and reverse-geocoding responses.

All errors are swallowed and logged at warning level so a Redis outage never
blocks an upstream API call.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any, Optional

import redis.asyncio as redis

from app.core.logging import get_logger

logger = get_logger(__name__)


def weather_cache_key(latitude: float, longitude: float, day: date) -> str:
    """Build a cache key for a per-day weather forecast lookup."""
    return f"weather:{latitude:.3f}:{longitude:.3f}:{day.isoformat()}"


def geocode_cache_key(latitude: float, longitude: float) -> str:
    """Build a cache key for a reverse-geocode lookup."""
    return f"geocode:{latitude:.3f}:{longitude:.3f}"


class WeatherCache:
    """Tiny async wrapper around Redis for JSON-encoded payloads."""

    def __init__(self, redis_url: str) -> None:
        self._redis_url = redis_url
        self._redis: Optional[redis.Redis] = None

    async def _conn(self) -> redis.Redis:
        if self._redis is None:
            self._redis = redis.from_url(self._redis_url, decode_responses=True)
        return self._redis

    async def get_json(self, key: str) -> Optional[dict[str, Any]]:
        """Return the cached JSON payload for ``key`` or ``None``."""
        try:
            client = await self._conn()
            raw = await client.get(key)
            if raw is None:
                return None
            decoded: dict[str, Any] = json.loads(raw)
            return decoded
        except Exception as exc:  # pragma: no cover - fail-soft
            logger.warning("weather cache get(%s) failed: %s", key, exc)
            return None

    async def set_json(self, key: str, value: dict[str, Any], ttl_s: int) -> None:
        """Store ``value`` under ``key`` with a TTL in seconds."""
        try:
            client = await self._conn()
            await client.set(key, json.dumps(value, default=str), ex=ttl_s)
        except Exception as exc:  # pragma: no cover - fail-soft
            logger.warning("weather cache set(%s) failed: %s", key, exc)

    async def aclose(self) -> None:
        """Release the underlying Redis connection pool."""
        if self._redis is not None:
            try:
                await self._redis.close()
            finally:
                self._redis = None
