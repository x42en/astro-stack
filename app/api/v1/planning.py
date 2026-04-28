"""Public planning endpoints: weather, geocoding, night windows, recommendations.

All endpoints are anonymous (no authentication required) so the discovery
flow on the marketing landing page works without sign-in.
"""

from __future__ import annotations

from datetime import date as date_type
from datetime import datetime, timedelta
from datetime import timezone as timezone_module
from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from app.core.config import get_settings
from app.domain.visibility import ObjectForecast, ObjectVisibility, ObservationWindow
from app.infrastructure.weather.cache import (
    WeatherCache,
    geocode_cache_key,
    weather_cache_key,
)
from app.infrastructure.weather.openmeteo_client import (
    GeoLocation,
    OpenMeteoClient,
    WeatherForecast,
)
from app.services.planner_service import PlannerService

router = APIRouter(prefix="/planning", tags=["planning"])


# ── Singletons (module-scoped lazy) ──────────────────────────────────────────


_client: Optional[OpenMeteoClient] = None
_cache: Optional[WeatherCache] = None
_planner: Optional[PlannerService] = None


def get_client() -> OpenMeteoClient:
    """FastAPI dependency: shared :class:`OpenMeteoClient`."""
    global _client
    if _client is None:
        _client = OpenMeteoClient()
    return _client


def get_cache() -> WeatherCache:
    """FastAPI dependency: shared :class:`WeatherCache`."""
    global _cache
    if _cache is None:
        _cache = WeatherCache(get_settings().redis_url)
    return _cache


def get_planner() -> PlannerService:
    """FastAPI dependency: shared :class:`PlannerService`."""
    global _planner
    if _planner is None:
        _planner = PlannerService()
    return _planner


# ── Schemas ───────────────────────────────────────────────────────────────────


class WeatherSummary(BaseModel):
    """Aggregated weather statistics for a single night."""

    cloud_cover_avg_pct: float
    cloud_cover_min_pct: float
    visibility_min_m: float


class RecommendationBundle(BaseModel):
    """Combined night-window + weather + ranked recommendations payload."""

    window: ObservationWindow
    weather_summary: Optional[WeatherSummary] = None
    recommendations: list[ObjectVisibility]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _validate_date(value: date_type) -> date_type:
    today = datetime.now(timezone_module.utc).date()
    earliest = today - timedelta(days=1)
    latest = today + timedelta(days=15)
    if value < earliest or value > latest:
        from app.core.errors import ErrorCode, ValidationException

        raise ValidationException(
            ErrorCode.PLAN_DATE_OUT_OF_RANGE,
            "date must be within today..today+15 days",
        )
    return value


async def _summarise_weather(
    client: OpenMeteoClient,
    cache: WeatherCache,
    lat: float,
    lon: float,
    day: date_type,
    window: ObservationWindow,
) -> Optional[WeatherSummary]:
    cache_key = weather_cache_key(lat, lon, day)
    payload = await cache.get_json(cache_key)
    if payload is None:
        try:
            forecast = await client.forecast(lat, lon, days=16)
        except Exception:  # pragma: no cover - fail-soft
            return None
        payload = forecast.model_dump(mode="json")
        await cache.set_json(cache_key, payload, get_settings().weather_cache_ttl_s)

    forecast = WeatherForecast.model_validate(payload)
    samples = [
        h
        for h in forecast.hourly
        if window.astronomical_twilight_end <= h.time <= window.astronomical_twilight_start
    ]
    if not samples:
        return None
    cloud_avg = sum(h.cloud_cover_pct for h in samples) / len(samples)
    cloud_min = min(h.cloud_cover_pct for h in samples)
    vis_min = min(h.visibility_m for h in samples)
    return WeatherSummary(
        cloud_cover_avg_pct=cloud_avg,
        cloud_cover_min_pct=cloud_min,
        visibility_min_m=vis_min,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get("/geocode/reverse", response_model=GeoLocation)
async def reverse_geocode(
    lat: float = Query(..., ge=-90.0, le=90.0),
    lon: float = Query(..., ge=-180.0, le=180.0),
    client: OpenMeteoClient = Depends(get_client),
    cache: WeatherCache = Depends(get_cache),
) -> GeoLocation:
    """Resolve coordinates to the nearest named place via Open-Meteo."""
    cache_key = geocode_cache_key(lat, lon)
    cached = await cache.get_json(cache_key)
    if cached is not None:
        return GeoLocation.model_validate(cached)
    location = await client.reverse_geocode(lat, lon)
    await cache.set_json(cache_key, location.model_dump(mode="json"), get_settings().geocode_cache_ttl_s)
    return location


@router.get("/weather", response_model=WeatherForecast)
async def get_weather(
    lat: float = Query(..., ge=-90.0, le=90.0),
    lon: float = Query(..., ge=-180.0, le=180.0),
    days: int = Query(16, ge=1, le=16),
    client: OpenMeteoClient = Depends(get_client),
    planner: PlannerService = Depends(get_planner),
) -> WeatherForecast:
    """Return the raw Open-Meteo forecast for the given location.

    The ``daily.moon_phase`` field is **not** provided by Open-Meteo's free
    tier; we enrich each daily entry with a Skyfield-computed phase (0..1,
    waxing→waning) so the UI can render moon glyphs correctly.
    """
    forecast = await client.forecast(lat, lon, days=days)
    if forecast.daily:
        try:
            phases = await planner.moon_phases_for_dates([d.date for d in forecast.daily])
        except Exception:  # pragma: no cover - fail-soft if ephemeris missing
            phases = []
        for day, phase in zip(forecast.daily, phases, strict=False):
            day.moon_phase = phase
    return forecast


@router.get("/window", response_model=ObservationWindow)
async def get_window(
    lat: float = Query(..., ge=-90.0, le=90.0),
    lon: float = Query(..., ge=-180.0, le=180.0),
    elevation: float = Query(0.0, ge=-500.0, le=9000.0),
    date: date_type = Query(...),
    timezone: str = Query("UTC", max_length=64),
    planner: PlannerService = Depends(get_planner),
) -> ObservationWindow:
    """Return the night observation window for one site/date."""
    _validate_date(date)
    return await planner.night_window(lat, lon, elevation, date, timezone)


@router.get("/recommendations", response_model=RecommendationBundle)
async def get_recommendations(
    lat: float = Query(..., ge=-90.0, le=90.0),
    lon: float = Query(..., ge=-180.0, le=180.0),
    elevation: float = Query(0.0, ge=-500.0, le=9000.0),
    date: date_type = Query(...),
    timezone: str = Query("UTC", max_length=64),
    min_altitude: float = Query(30.0, ge=0.0, le=89.0),
    limit: int = Query(50, ge=1, le=100),
    type: list[str] = Query(default_factory=list),
    planner: PlannerService = Depends(get_planner),
    client: OpenMeteoClient = Depends(get_client),
    cache: WeatherCache = Depends(get_cache),
) -> RecommendationBundle:
    """Combined window + weather summary + ranked DSO recommendations."""
    _validate_date(date)
    window = await planner.night_window(lat, lon, elevation, date, timezone)
    summary = await _summarise_weather(client, cache, lat, lon, date, window)
    recs = await planner.rank_objects(
        lat,
        lon,
        elevation,
        window,
        min_altitude_deg=min_altitude,
        max_results=limit,
        type_filter=type or None,
    )
    return RecommendationBundle(window=window, weather_summary=summary, recommendations=recs)


@router.get("/object/{catalog_id}/visibility", response_model=ObjectVisibility)
async def get_object_visibility(
    catalog_id: str,
    lat: float = Query(..., ge=-90.0, le=90.0),
    lon: float = Query(..., ge=-180.0, le=180.0),
    elevation: float = Query(0.0, ge=-500.0, le=9000.0),
    date: date_type = Query(...),
    timezone: str = Query("UTC", max_length=64),
    planner: PlannerService = Depends(get_planner),
) -> ObjectVisibility:
    """Visibility curve + score for one named catalog object."""
    _validate_date(date)
    window = await planner.night_window(lat, lon, elevation, date, timezone)
    return await planner.visibility_for_object(lat, lon, elevation, window, catalog_id)


@router.get("/object/{catalog_id}/forecast", response_model=ObjectForecast)
async def get_object_forecast(
    catalog_id: str,
    lat: float = Query(..., ge=-90.0, le=90.0),
    lon: float = Query(..., ge=-180.0, le=180.0),
    elevation: float = Query(0.0, ge=-500.0, le=9000.0),
    start_date: Optional[date_type] = Query(None, description="Defaults to today (UTC)."),
    days: int = Query(90, ge=1, le=365),
    min_altitude: float = Query(30.0, ge=0.0, le=89.0),
    timezone: str = Query("UTC", max_length=64),
    planner: PlannerService = Depends(get_planner),
    cache: WeatherCache = Depends(get_cache),
) -> ObjectForecast:
    """Multi-night observability forecast for a single object.

    Returns a per-night entry (max altitude, hours above ``min_altitude``,
    moon context, score) for ``days`` consecutive nights starting at
    ``start_date`` (defaults to today, UTC). The result is purely geometric
    (no weather) and is cached for one day per (object, site, horizon).
    """
    if start_date is None:
        start_date = datetime.now(timezone_module.utc).date()

    cache_key = (
        f"objfc:{catalog_id}:{lat:.4f}:{lon:.4f}:{elevation:.0f}:"
        f"{start_date.isoformat()}:{days}:{min_altitude:.0f}"
    )
    cached = await cache.get_json(cache_key)
    if cached is not None:
        return ObjectForecast.model_validate(cached)

    forecast = await planner.forecast_for_object(
        lat,
        lon,
        elevation,
        catalog_id,
        start_date=start_date,
        days=days,
        min_altitude_deg=min_altitude,
        timezone_name=timezone,
    )
    await cache.set_json(cache_key, forecast.model_dump(mode="json"), 24 * 3600)
    return forecast
