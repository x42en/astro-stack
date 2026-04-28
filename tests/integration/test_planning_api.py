"""Integration tests for the planning REST endpoints.

The Open-Meteo client and the Redis-backed cache are stubbed via
:attr:`FastAPI.dependency_overrides`. The planner runs against the in-process
catalog; tests that need the DE421 ephemeris are skipped when it is missing.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1.planning import (
    get_cache,
    get_client,
    get_planner,
    router as planning_router,
)
from app.api.middleware.error_handler import register_error_handlers
from app.infrastructure.weather.openmeteo_client import (
    DailyWeather,
    HourlyWeather,
    WeatherForecast,
)
from app.services.planner_service import PlannerService

EPHEMERIS_PATH = "/opt/ephemerides/de421.bsp"
_HAVE_EPHEMERIS = Path(EPHEMERIS_PATH).exists()


# ── Fakes ─────────────────────────────────────────────────────────────────────


class _FakeCache:
    """In-memory replacement for :class:`WeatherCache`."""

    def __init__(self) -> None:
        self.store: dict[str, dict[str, Any]] = {}

    async def get_json(self, key: str) -> Optional[dict[str, Any]]:
        return self.store.get(key)

    async def set_json(self, key: str, value: dict[str, Any], ttl_s: int) -> None:
        self.store[key] = value


class _FakeClient:
    """In-memory replacement for :class:`OpenMeteoClient`."""

    async def forecast(self, latitude: float, longitude: float, days: int = 16) -> WeatherForecast:
        # Build 24h of cloud-free hourly samples spanning the next night.
        now = datetime.now(timezone.utc).replace(microsecond=0, second=0, minute=0)
        hourly = [
            HourlyWeather(
                time=now + timedelta(hours=i),
                cloud_cover_pct=20.0,
                cloud_cover_low_pct=5.0,
                visibility_m=20000.0,
                relative_humidity_pct=60.0,
                dew_point_c=10.0,
                wind_speed_kmh=4.0,
            )
            for i in range(48)
        ]
        daily = [
            DailyWeather(
                date=now.date(),
                sunrise=now,
                sunset=now + timedelta(hours=12),
                moonrise=None,
                moonset=None,
                moon_phase=0.2,
            )
        ]
        return WeatherForecast(
            latitude=latitude,
            longitude=longitude,
            timezone="UTC",
            elevation_m=0.0,
            hourly=hourly,
            daily=daily,
        )

    async def reverse_geocode(self, *_args: Any, **_kw: Any) -> Any:
        raise NotImplementedError

    async def aclose(self) -> None:
        pass


@pytest.fixture
def app() -> FastAPI:
    """Build a minimal FastAPI app exposing only the planning router."""
    application = FastAPI()
    register_error_handlers(application)
    application.include_router(planning_router, prefix="/api/v1")
    application.dependency_overrides[get_client] = lambda: _FakeClient()
    application.dependency_overrides[get_cache] = lambda: _FakeCache()
    application.dependency_overrides[get_planner] = lambda: PlannerService(
        ephemeris_path=EPHEMERIS_PATH
    )
    return application


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestDateValidation:
    def test_date_too_far_in_future_returns_422(self, client: TestClient) -> None:
        far = (datetime.now(timezone.utc).date() + timedelta(days=30)).isoformat()
        response = client.get(
            "/api/v1/planning/recommendations",
            params={
                "lat": 48.85,
                "lon": 2.35,
                "elevation": 35.0,
                "date": far,
                "timezone": "Europe/Paris",
            },
        )
        assert response.status_code == 422
        body = response.json()
        # Error envelope from register_error_handlers exposes error_code.
        assert "PLAN_DATE_OUT_OF_RANGE" in str(body)


@pytest.mark.skipif(not _HAVE_EPHEMERIS, reason="DE421 ephemeris not available")
class TestRecommendationsSmoke:
    def test_recommendations_paris_today(self, client: TestClient) -> None:
        today = datetime.now(timezone.utc).date().isoformat()
        response = client.get(
            "/api/v1/planning/recommendations",
            params={
                "lat": 48.85,
                "lon": 2.35,
                "elevation": 35.0,
                "date": today,
                "timezone": "Europe/Paris",
                "min_altitude": 20,
                "limit": 5,
            },
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert "window" in payload
        assert "recommendations" in payload
        assert isinstance(payload["recommendations"], list)
