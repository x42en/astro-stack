"""Visibility/planning DTOs returned by the session-prep endpoints.

These are pure pydantic models, not ORM tables — they describe response
payloads for the planner service.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field


class ObservationWindow(BaseModel):
    """Per-night observability summary for a given site."""

    site_latitude: float
    site_longitude: float
    site_elevation_m: float
    date: date
    sunset: datetime
    sunrise_next: datetime
    astronomical_twilight_end: datetime
    astronomical_twilight_start: datetime
    moon_illumination: float = Field(ge=0.0, le=1.0)
    moonrise: Optional[datetime] = None
    moonset: Optional[datetime] = None
    moon_above_horizon_during_window: bool
    darkness_score: float = Field(ge=0.0, le=100.0)


class AltAzPoint(BaseModel):
    """One sample of an object's altitude/azimuth track."""

    time: datetime
    altitude_deg: float
    azimuth_deg: float


class ObjectVisibility(BaseModel):
    """Per-object visibility summary inside an :class:`ObservationWindow`."""

    catalog_id: str
    name: str
    type: str
    constellation: str
    ra_deg: float
    dec_deg: float
    magnitude: Optional[float] = None
    max_altitude_deg: float
    transit_time: Optional[datetime] = None
    rise_time: Optional[datetime] = None
    set_time: Optional[datetime] = None
    moon_separation_deg: float
    score: float = Field(ge=0.0, le=100.0)
    altitude_curve: list[AltAzPoint] = Field(default_factory=list)


class NightlyForecastEntry(BaseModel):
    """One night of observability for a single object (no weather)."""

    date: date
    max_altitude_deg: float
    hours_above_min_altitude: float = Field(ge=0.0)
    transit_time: Optional[datetime] = None
    moon_separation_deg: float
    moon_illumination: float = Field(ge=0.0, le=1.0)
    moon_above_horizon_during_window: bool
    darkness_score: float = Field(ge=0.0, le=100.0)
    score: float = Field(ge=0.0, le=100.0)


class ObjectForecast(BaseModel):
    """Multi-night observation forecast for one catalog object.

    Returned by ``GET /planning/object/{id}/forecast``. Contains a per-night
    score (no weather; purely geometric + lunar) so users can identify the
    best dates to observe a target months ahead.
    """

    catalog_id: str
    name: str
    type: str
    constellation: str
    ra_deg: float
    dec_deg: float
    magnitude: Optional[float] = None
    site_latitude: float
    site_longitude: float
    site_elevation_m: float
    min_altitude_deg: float = Field(ge=0.0, le=89.0)
    nights: list[NightlyForecastEntry] = Field(default_factory=list)

