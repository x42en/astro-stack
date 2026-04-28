"""Unit tests for :mod:`app.infrastructure.weather.openmeteo_client`."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from app.core.errors import ExternalServiceException, ValidationException
from app.infrastructure.weather import openmeteo_client as openmeteo_module
from app.infrastructure.weather.openmeteo_client import (
    OpenMeteoClient,
    _validate_coords,
)


class _FakeSettings:
    app_version = "0.1.0"
    openmeteo_forecast_url = "https://api.open-meteo.test/forecast"
    openmeteo_geocode_url = "https://geocode.open-meteo.test/search"
    nominatim_reverse_url = "https://nominatim.test/reverse"


@pytest.fixture(autouse=True)
def stub_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid loading the real .env (which may contain unrelated keys)."""
    monkeypatch.setattr(openmeteo_module, "get_settings", lambda: _FakeSettings())


class TestValidateCoords:
    def test_valid_coords_pass(self) -> None:
        _validate_coords(0.0, 0.0)
        _validate_coords(90.0, 180.0)
        _validate_coords(-90.0, -180.0)

    def test_rejects_invalid_latitude(self) -> None:
        with pytest.raises(ValidationException):
            _validate_coords(91.0, 0.0)

    def test_rejects_invalid_longitude(self) -> None:
        with pytest.raises(ValidationException):
            _validate_coords(0.0, 200.0)


def _forecast_payload() -> dict[str, Any]:
    return {
        "latitude": 48.85,
        "longitude": 2.35,
        "timezone": "Europe/Paris",
        "elevation": 35.0,
        "hourly": {
            "time": ["2024-07-15T22:00", "2024-07-15T23:00"],
            "cloud_cover": [10.0, 20.0],
            "cloud_cover_low": [5.0, 5.0],
            "visibility": [20000.0, 18000.0],
            "relative_humidity_2m": [60.0, 65.0],
            "dew_point_2m": [12.0, 11.0],
            "wind_speed_10m": [5.0, 6.0],
        },
        "daily": {
            "time": ["2024-07-15"],
            "sunrise": ["2024-07-15T06:00"],
            "sunset": ["2024-07-15T21:30"],
            "moonrise": ["2024-07-15T18:00"],
            "moonset": ["2024-07-16T03:00"],
            "moon_phase": [0.5],
        },
    }


class TestForecast:
    @pytest.mark.asyncio
    async def test_parses_open_meteo_payload(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert "/forecast" in str(request.url) or "forecast" in str(request.url)
            return httpx.Response(200, json=_forecast_payload())

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http:
            client = OpenMeteoClient(http_client=http)
            forecast = await client.forecast(48.85, 2.35, days=1)

        assert forecast.timezone == "Europe/Paris"
        assert len(forecast.hourly) == 2
        assert forecast.hourly[0].cloud_cover_pct == 10.0
        assert len(forecast.daily) == 1
        assert forecast.daily[0].moon_phase == 0.5

    @pytest.mark.asyncio
    async def test_retries_once_then_raises(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(503, text="boom")

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http:
            client = OpenMeteoClient(http_client=http)
            with pytest.raises(ExternalServiceException):
                await client.forecast(48.85, 2.35, days=1)

        assert calls["n"] == 2  # initial attempt + one retry

    @pytest.mark.asyncio
    async def test_rejects_invalid_lat(self) -> None:
        client = OpenMeteoClient(http_client=httpx.AsyncClient())
        try:
            with pytest.raises(ValidationException):
                await client.forecast(91.0, 0.0)
        finally:
            await client.aclose()


class TestReverseGeocode:
    @pytest.mark.asyncio
    async def test_returns_first_result(self) -> None:
        # Nominatim payload shape (jsonv2 with addressdetails=1).
        payload = {
            "display_name": "Paris, Île-de-France, France",
            "address": {
                "city": "Paris",
                "country": "France",
            },
            "lat": "48.8566",
            "lon": "2.3522",
        }

        def handler(request: httpx.Request) -> httpx.Response:
            assert "nominatim" in str(request.url)
            assert request.headers.get("user-agent", "").startswith("AstroStack/")
            return httpx.Response(200, json=payload)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http:
            client = OpenMeteoClient(http_client=http)
            location = await client.reverse_geocode(48.85, 2.35)

        assert location.name == "Paris"
        assert location.country == "France"
        # Nominatim does not return a timezone; the synthetic value is "UTC".
        assert location.timezone == "UTC"

    @pytest.mark.asyncio
    async def test_falls_back_when_nominatim_errors(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="boom")

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http:
            client = OpenMeteoClient(http_client=http)
            location = await client.reverse_geocode(0.0, 0.0)

        # Synthetic placeholder so the planning UI keeps working.
        assert location.timezone == "UTC"
        assert "0.0000" in location.name
