"""Async client for the public Open-Meteo forecast and geocoding APIs.

The two endpoints we use are documented at:
  * https://open-meteo.com/en/docs               (forecast)
  * https://open-meteo.com/en/docs/geocoding-api (reverse geocoding)

No API key is required. The client retries each request once on transient
failure and raises :class:`ExternalServiceException` on definitive failure.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

import httpx
from pydantic import BaseModel, Field, field_validator

from app.core.config import get_settings
from app.core.errors import ErrorCode, ExternalServiceException, ValidationException
from app.core.logging import get_logger

logger = get_logger(__name__)


def _ensure_utc(value: datetime | str | None) -> datetime | None:
    """Coerce a datetime (or ISO string) into a tz-aware UTC datetime.

    Pydantic re-validation of *cached* JSON payloads can re-introduce naive
    datetimes when the cached string lacks an offset (i.e. payloads created
    before the timezone-normalisation fix). Forcing UTC here makes the
    forecast model safe to compare against the planner's tz-aware windows
    regardless of the payload origin.
    """
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


# ── DTOs ──────────────────────────────────────────────────────────────────────


class GeoLocation(BaseModel):
    """Result of a reverse-geocoding call."""

    name: str
    country: Optional[str] = None
    timezone: str
    elevation_m: float
    latitude: float
    longitude: float


class HourlyWeather(BaseModel):
    """One sample of hourly weather for the planner."""

    time: datetime
    cloud_cover_pct: float
    cloud_cover_low_pct: float
    visibility_m: float
    relative_humidity_pct: float
    dew_point_c: float
    wind_speed_kmh: float

    @field_validator("time", mode="before")
    @classmethod
    def _normalise_time(cls, v: Any) -> Any:
        return _ensure_utc(v) if v is not None else v


class DailyWeather(BaseModel):
    """Per-day astronomical events from Open-Meteo."""

    date: date
    sunrise: datetime
    sunset: datetime
    moonrise: Optional[datetime] = None
    moonset: Optional[datetime] = None
    moon_phase: float = Field(ge=0.0, le=1.0)

    @field_validator("sunrise", "sunset", "moonrise", "moonset", mode="before")
    @classmethod
    def _normalise_dt(cls, v: Any) -> Any:
        return _ensure_utc(v) if v is not None else v


class WeatherForecast(BaseModel):
    """Bundled hourly + daily forecast returned by :meth:`forecast`."""

    latitude: float
    longitude: float
    timezone: str
    elevation_m: float
    hourly: list[HourlyWeather]
    daily: list[DailyWeather]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _validate_coords(latitude: float, longitude: float) -> None:
    if not (-90.0 <= latitude <= 90.0):
        raise ValidationException(
            ErrorCode.PLAN_INVALID_COORDS,
            "latitude must be between -90 and 90 degrees",
        )
    if not (-180.0 <= longitude <= 180.0):
        raise ValidationException(
            ErrorCode.PLAN_INVALID_COORDS,
            "longitude must be between -180 and 180 degrees",
        )


def _parse_dt(raw: str | None, tz: timezone = timezone.utc) -> Optional[datetime]:
    """Parse an ISO-8601 datetime, attaching ``tz`` if the string is naive.

    Open-Meteo returns local-time strings without any offset suffix when the
    ``timezone=auto`` parameter is used (e.g. ``2026-04-28T00:00``). Downstream
    consumers (planner_service) work with **timezone-aware** UTC datetimes, so
    we normalise here to avoid ``can't compare offset-naive and offset-aware``
    comparison errors.
    """
    if raw is None or raw == "":
        return None
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    return dt.astimezone(timezone.utc)


def _parse_dt_required(raw: str, tz: timezone = timezone.utc) -> datetime:
    dt = _parse_dt(raw, tz)
    assert dt is not None  # raw is non-empty by contract
    return dt


def _synthetic_location(latitude: float, longitude: float) -> "GeoLocation":
    """Return a placeholder :class:`GeoLocation` when geocoding fails."""
    return GeoLocation(
        name=f"{latitude:.4f}, {longitude:.4f}",
        country=None,
        timezone="UTC",
        elevation_m=0.0,
        latitude=latitude,
        longitude=longitude,
    )


# ── Client ────────────────────────────────────────────────────────────────────


class OpenMeteoClient:
    """Async client wrapping the Open-Meteo forecast and geocoding APIs.

    Args:
        http_client: Optional pre-configured ``httpx.AsyncClient``. When
            omitted, the client owns and closes its own transport.
    """

    def __init__(self, http_client: Optional[httpx.AsyncClient] = None) -> None:
        self._http: httpx.AsyncClient = http_client or httpx.AsyncClient(timeout=15.0)
        self._owns_http: bool = http_client is None

    async def aclose(self) -> None:
        """Close the underlying HTTP client when owned."""
        if self._owns_http:
            await self._http.aclose()

    async def _get_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                response = await self._http.get(url, params=params)
                response.raise_for_status()
                payload: dict[str, Any] = response.json()
                return payload
            except (httpx.HTTPError, ValueError) as exc:
                last_exc = exc
                logger.warning(
                    "open-meteo call failed (attempt %d/2): %s", attempt + 1, exc
                )
                if attempt == 0:
                    await asyncio.sleep(0.2)
        raise ExternalServiceException(
            f"Open-Meteo request to {url} failed: {last_exc}",
            details={"url": url, "params": params},
        )

    async def forecast(
        self, latitude: float, longitude: float, days: int = 16
    ) -> WeatherForecast:
        """Fetch hourly + daily forecast for ``days`` (clamped to 1..16)."""
        _validate_coords(latitude, longitude)
        days = max(1, min(16, days))
        # NOTE: ``visibility`` is only available for the next ~7 days on
        # Open-Meteo's free tier and triggers an HTTP 400 when combined with
        # ``forecast_days=16``. We omit it here and default the field to 0
        # in :func:`_parse_forecast`; downstream callers do not display it.
        params: dict[str, Any] = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": (
                "cloud_cover,cloud_cover_low,"
                "relative_humidity_2m,dew_point_2m,wind_speed_10m"
            ),
            # NOTE: Open-Meteo's free ``/v1/forecast`` daily aggregation only
            # exposes solar events (``sunrise``/``sunset``); moon variables
            # (``moonrise``, ``moonset``, ``moon_phase``) are NOT supported and
            # requesting them yields HTTP 400. Moon data is computed by
            # :mod:`app.services.planner_service` via Skyfield instead.
            "daily": "sunrise,sunset",
            "timezone": "auto",
            "forecast_days": days,
        }
        payload = await self._get_json(get_settings().openmeteo_forecast_url, params)
        return _parse_forecast(payload)

    async def reverse_geocode(self, latitude: float, longitude: float) -> GeoLocation:
        """Return the nearest named place for ``(latitude, longitude)``.

        Open-Meteo only exposes *forward* geocoding (``/v1/search``); it has
        no reverse-geocoding endpoint. We use OpenStreetMap Nominatim instead
        — its public instance is free and well-suited to low-volume traffic
        provided each request carries a descriptive ``User-Agent``.

        On any failure we fall back to a synthetic :class:`GeoLocation` so
        the planning UI keeps working — only the human-readable name is lost.
        """
        _validate_coords(latitude, longitude)
        params: dict[str, Any] = {
            "lat": latitude,
            "lon": longitude,
            "format": "jsonv2",
            "zoom": 10,
            "addressdetails": 1,
            "accept-language": "en",
        }
        try:
            response = await self._http.get(
                get_settings().nominatim_reverse_url,
                params=params,
                headers={
                    "User-Agent": (
                        f"AstroStack/{get_settings().app_version} "
                        "(+https://github.com/circle-cyber/astro-stack)"
                    ),
                    "Accept": "application/json",
                },
            )
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("nominatim reverse-geocode failed: %s", exc)
            return _synthetic_location(latitude, longitude)

        address = payload.get("address") or {}
        name = (
            address.get("city")
            or address.get("town")
            or address.get("village")
            or address.get("hamlet")
            or address.get("municipality")
            or address.get("county")
            or payload.get("name")
            or payload.get("display_name")
            or f"{latitude:.4f}, {longitude:.4f}"
        )
        country = address.get("country")
        return GeoLocation(
            name=str(name),
            country=str(country) if country else None,
            timezone="UTC",  # Nominatim does not return tz; client may upgrade.
            elevation_m=0.0,
            latitude=latitude,
            longitude=longitude,
        )


def _parse_forecast(payload: dict[str, Any]) -> WeatherForecast:
    """Convert a raw Open-Meteo response into a :class:`WeatherForecast`.

    All datetimes are normalised to UTC. Open-Meteo returns local-time strings
    when ``timezone=auto`` is used, together with a ``utc_offset_seconds``
    field; we use that offset to interpret the strings before converting to
    UTC, so downstream comparisons against tz-aware datetimes succeed.
    """
    offset_seconds = int(payload.get("utc_offset_seconds") or 0)
    local_tz = timezone(timedelta(seconds=offset_seconds))
    hourly_raw = payload.get("hourly") or {}
    times: list[str] = hourly_raw.get("time") or []
    cloud: list[float] = hourly_raw.get("cloud_cover") or []
    cloud_low: list[float] = hourly_raw.get("cloud_cover_low") or []
    vis: list[float] = hourly_raw.get("visibility") or []
    rh: list[float] = hourly_raw.get("relative_humidity_2m") or []
    dew: list[float] = hourly_raw.get("dew_point_2m") or []
    wind: list[float] = hourly_raw.get("wind_speed_10m") or []

    def _sf(seq: list, idx: int, default: float = 0.0) -> float:
        """Return ``float(seq[idx])`` or *default* when out-of-bounds or None.

        Open-Meteo returns ``null`` for incomplete hourly slots at the tail of
        a 16-day forecast (the model hasn't produced data for those hours yet).
        Using ``float(None)`` raises ``TypeError``, so we guard every access.
        """
        v = seq[idx] if idx < len(seq) else None
        return float(v) if v is not None else default

    hourly: list[HourlyWeather] = []
    for i, t in enumerate(times):
        hourly.append(
            HourlyWeather(
                time=_parse_dt_required(t, local_tz),
                cloud_cover_pct=_sf(cloud, i),
                cloud_cover_low_pct=_sf(cloud_low, i),
                visibility_m=_sf(vis, i),
                relative_humidity_pct=_sf(rh, i),
                dew_point_c=_sf(dew, i),
                wind_speed_kmh=_sf(wind, i),
            )
        )

    daily_raw = payload.get("daily") or {}
    d_times: list[str] = daily_raw.get("time") or []
    sunrise: list[str] = daily_raw.get("sunrise") or []
    sunset: list[str] = daily_raw.get("sunset") or []
    moonrise: list[str | None] = daily_raw.get("moonrise") or []
    moonset: list[str | None] = daily_raw.get("moonset") or []
    moon_phase: list[float] = daily_raw.get("moon_phase") or []

    daily: list[DailyWeather] = []
    for i, dt in enumerate(d_times):
        daily.append(
            DailyWeather(
                date=date.fromisoformat(dt),
                sunrise=_parse_dt_required(sunrise[i], local_tz),
                sunset=_parse_dt_required(sunset[i], local_tz),
                moonrise=_parse_dt(moonrise[i] if i < len(moonrise) else None, local_tz),
                moonset=_parse_dt(moonset[i] if i < len(moonset) else None, local_tz),
                moon_phase=float(moon_phase[i]) if i < len(moon_phase) and moon_phase[i] is not None else 0.0,
            )
        )

    return WeatherForecast(
        latitude=float(payload.get("latitude") or 0.0),
        longitude=float(payload.get("longitude") or 0.0),
        timezone=str(payload.get("timezone") or "UTC"),
        elevation_m=float(payload.get("elevation") or 0.0),
        hourly=hourly,
        daily=daily,
    )
