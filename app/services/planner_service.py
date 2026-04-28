"""Sky-planning service: night windows and per-object visibility ranking.

All ephemeris computation is delegated to :mod:`skyfield`. The DE440s planet
ephemeris is loaded lazily (once per process) from
``settings.ephemeris_path``. Every public method dispatches the synchronous
skyfield work to a worker thread via :func:`asyncio.to_thread`.
"""

from __future__ import annotations

import asyncio
import os
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Optional

from app.core.config import get_settings
from app.core.errors import ErrorCode, ExternalServiceException, NotFoundException
from app.core.logging import get_logger
from app.domain.visibility import AltAzPoint, ObjectVisibility, ObservationWindow
from app.infrastructure.catalog.messier import CatalogObject
from app.infrastructure.catalog.registry import all_objects, lookup_object

logger = get_logger(__name__)


# ── Skyfield lazy loaders ─────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _load_skyfield_modules() -> dict[str, Any]:
    """Import skyfield lazily so the module is importable without the dep."""
    from skyfield import almanac
    from skyfield.api import Star, load, load_file, wgs84

    return {
        "almanac": almanac,
        "Star": Star,
        "load": load,
        "load_file": load_file,
        "wgs84": wgs84,
        "ts": load.timescale(),
    }


@lru_cache(maxsize=1)
def _load_ephemeris(path: str) -> Any:
    """Load and cache the DE440s ephemeris file from ``path``.

    The file is validated up front: a missing or zero-byte file (the typical
    symptom of a failed Docker build-time download) raises a clear
    :class:`ExternalServiceException` instead of letting :mod:`jplephem`
    crash later with the cryptic ``file starts with b''`` error.
    """
    if not os.path.exists(path):
        raise ExternalServiceException(
            f"Skyfield ephemeris missing at {path}",
            details={"path": path},
        )
    size = os.path.getsize(path)
    if size < 1024:  # DE440s is ~32 MB; anything tiny is a broken download.
        raise ExternalServiceException(
            f"Skyfield ephemeris at {path} appears corrupt ({size} bytes). "
            "Re-download with: wget https://naif.jpl.nasa.gov/pub/naif/"
            "generic_kernels/spk/planets/de440s.bsp -O " + path,
            details={"path": path, "size_bytes": size},
        )
    sf = _load_skyfield_modules()
    try:
        return sf["load_file"](path)
    except (ValueError, OSError) as exc:
        raise ExternalServiceException(
            f"Failed to parse Skyfield ephemeris at {path}: {exc}",
            details={"path": path, "size_bytes": size},
        ) from exc


# ── Pure helpers (no skyfield needed) ─────────────────────────────────────────


def _compute_score(max_alt_deg: float, magnitude: Optional[float], moon_sep_deg: float) -> float:
    """Compute the planner score for a single object."""
    if max_alt_deg <= 0:
        return 0.0
    alt_weight = max(0.0, min(1.0, max_alt_deg / 90.0))
    mag = magnitude if magnitude is not None else 8.0
    mag_weight = max(0.0, min(1.0, (10.0 - mag) / 10.0)) if magnitude is not None else 0.4
    moon_weight = max(0.2, min(1.0, moon_sep_deg / 90.0))
    score = 100.0 * alt_weight * (0.4 + 0.6 * mag_weight) * moon_weight
    return max(0.0, min(100.0, score))


# ── Service ───────────────────────────────────────────────────────────────────


class PlannerService:
    """Compute night observation windows and rank visible deep-sky targets."""

    def __init__(self, ephemeris_path: Optional[str] = None) -> None:
        self._ephemeris_path = ephemeris_path or get_settings().ephemeris_path

    # — async wrappers — ------------------------------------------------------

    async def night_window(
        self,
        latitude: float,
        longitude: float,
        elevation_m: float,
        day: date,
        timezone_name: str = "UTC",
    ) -> ObservationWindow:
        """Compute the observation window (twilights, moon) for one night."""
        return await asyncio.to_thread(
            self._compute_night_window,
            latitude,
            longitude,
            elevation_m,
            day,
            timezone_name,
        )

    async def visibility_for_object(
        self,
        latitude: float,
        longitude: float,
        elevation_m: float,
        window: ObservationWindow,
        catalog_id: str,
        *,
        sample_minutes: int = 30,
    ) -> ObjectVisibility:
        """Compute alt/az curve + score for a single catalog object."""
        obj = lookup_object(catalog_id)
        if obj is None:
            raise NotFoundException(
                ErrorCode.PLAN_OBJECT_NOT_FOUND,
                f"Unknown catalog id: {catalog_id}",
            )
        return await asyncio.to_thread(
            self._compute_visibility,
            latitude,
            longitude,
            elevation_m,
            window,
            obj,
            sample_minutes,
            True,
        )

    async def rank_objects(
        self,
        latitude: float,
        longitude: float,
        elevation_m: float,
        window: ObservationWindow,
        *,
        min_altitude_deg: float = 30.0,
        max_results: int = 50,
        type_filter: Optional[list[str]] = None,
    ) -> list[ObjectVisibility]:
        """Rank all bundled objects by observability inside ``window``."""
        return await asyncio.to_thread(
            self._compute_ranking,
            latitude,
            longitude,
            elevation_m,
            window,
            min_altitude_deg,
            max_results,
            type_filter,
        )

    # — synchronous skyfield workers — ---------------------------------------

    def _compute_night_window(
        self,
        latitude: float,
        longitude: float,
        elevation_m: float,
        day: date,
        timezone_name: str,
    ) -> ObservationWindow:
        sf = _load_skyfield_modules()
        eph = _load_ephemeris(self._ephemeris_path)
        ts = sf["ts"]
        almanac = sf["almanac"]
        wgs84 = sf["wgs84"]
        observer_topos = wgs84.latlon(latitude, longitude, elevation_m=elevation_m)

        # Local-noon UTC anchor: noon local time on the requested day.
        # Without zoneinfo we approximate via longitude offset.
        noon_local = datetime(day.year, day.month, day.day, 12, 0, 0, tzinfo=timezone.utc)
        noon_offset_h = -longitude / 15.0  # west = positive UTC offset
        anchor_utc = noon_local + timedelta(hours=noon_offset_h)
        t0 = ts.from_datetime(anchor_utc)
        t1 = ts.from_datetime(anchor_utc + timedelta(hours=24))

        # Sun rise/set
        sun = eph["sun"]
        moon = eph["moon"]
        earth = eph["earth"]
        observer = earth + observer_topos

        rs_fn = almanac.risings_and_settings(eph, sun, observer_topos)
        times, events = almanac.find_discrete(t0, t1, rs_fn)
        sunset_dt: Optional[datetime] = None
        sunrise_dt: Optional[datetime] = None
        for t, e in zip(times, events, strict=False):
            dt = t.utc_datetime()
            if int(e) == 0 and sunset_dt is None:  # set
                sunset_dt = dt
            elif int(e) == 1 and sunset_dt is not None and sunrise_dt is None:
                sunrise_dt = dt
        if sunset_dt is None:
            sunset_dt = anchor_utc + timedelta(hours=6)
        if sunrise_dt is None:
            sunrise_dt = sunset_dt + timedelta(hours=10)

        # Twilight transitions (astronomical = state >=1 vs 0=dark)
        tw_fn = almanac.dark_twilight_day(eph, observer_topos)
        tw_t0 = ts.from_datetime(sunset_dt - timedelta(hours=1))
        tw_t1 = ts.from_datetime(sunrise_dt + timedelta(hours=1))
        tt, ts_state = almanac.find_discrete(tw_t0, tw_t1, tw_fn)
        astro_end: Optional[datetime] = None
        astro_start: Optional[datetime] = None
        prev_state: Optional[int] = None
        for t, s in zip(tt, ts_state, strict=False):
            state = int(s)
            dt = t.utc_datetime()
            if prev_state is not None:
                if prev_state == 1 and state == 0 and astro_end is None:
                    astro_end = dt
                elif prev_state == 0 and state == 1 and astro_end is not None:
                    astro_start = dt
            prev_state = state
        if astro_end is None:
            astro_end = sunset_dt + timedelta(minutes=90)
        if astro_start is None:
            astro_start = sunrise_dt - timedelta(minutes=90)

        # Moon rise/set within the dark window
        moonrise: Optional[datetime] = None
        moonset: Optional[datetime] = None
        m_fn = almanac.risings_and_settings(eph, moon, observer_topos)
        m_t0 = ts.from_datetime(sunset_dt)
        m_t1 = ts.from_datetime(sunrise_dt)
        mt, mev = almanac.find_discrete(m_t0, m_t1, m_fn)
        for t, e in zip(mt, mev, strict=False):
            dt = t.utc_datetime()
            if int(e) == 1 and moonrise is None:
                moonrise = dt
            elif int(e) == 0 and moonset is None:
                moonset = dt

        # Moon illumination at midnight of the window
        midpoint = sunset_dt + (sunrise_dt - sunset_dt) / 2
        t_mid = ts.from_datetime(midpoint)
        illumination = float(almanac.fraction_illuminated(eph, "moon", t_mid))

        # Is the moon up at any time during the dark window?
        moon_above = False
        sample = sunset_dt
        step = timedelta(minutes=20)
        while sample <= sunrise_dt:
            t_s = ts.from_datetime(sample)
            alt, _, _ = observer.at(t_s).observe(moon).apparent().altaz()
            if alt.degrees > 0:
                moon_above = True
                break
            sample += step

        # Darkness score
        dark_hours = max(
            0.0, (astro_start - astro_end).total_seconds() / 3600.0 if astro_start and astro_end else 0.0
        )
        moon_penalty = illumination if moon_above else illumination * 0.2
        darkness = 100.0 * (1.0 - moon_penalty) * min(1.0, dark_hours / 8.0)
        darkness = max(0.0, min(100.0, darkness))

        return ObservationWindow(
            site_latitude=latitude,
            site_longitude=longitude,
            site_elevation_m=elevation_m,
            date=day,
            sunset=sunset_dt,
            sunrise_next=sunrise_dt,
            astronomical_twilight_end=astro_end,
            astronomical_twilight_start=astro_start,
            moon_illumination=max(0.0, min(1.0, illumination)),
            moonrise=moonrise,
            moonset=moonset,
            moon_above_horizon_during_window=moon_above,
            darkness_score=darkness,
        )

    def _compute_visibility(
        self,
        latitude: float,
        longitude: float,
        elevation_m: float,
        window: ObservationWindow,
        obj: CatalogObject,
        sample_minutes: int,
        with_curve: bool,
    ) -> ObjectVisibility:
        sf = _load_skyfield_modules()
        eph = _load_ephemeris(self._ephemeris_path)
        ts = sf["ts"]
        Star = sf["Star"]
        wgs84 = sf["wgs84"]
        earth = eph["earth"]
        moon = eph["moon"]
        observer = earth + wgs84.latlon(latitude, longitude, elevation_m=elevation_m)

        target = Star(ra_hours=obj.ra_deg / 15.0, dec_degrees=obj.dec_deg)
        start = window.astronomical_twilight_end
        end = window.astronomical_twilight_start

        max_alt = -90.0
        max_t: Optional[datetime] = None
        rise_t: Optional[datetime] = None
        set_t: Optional[datetime] = None
        prev_alt: Optional[float] = None
        curve: list[AltAzPoint] = []
        moon_sep_at_max = 180.0

        sample = start
        step = timedelta(minutes=sample_minutes)
        while sample <= end:
            t = ts.from_datetime(sample)
            astrometric = observer.at(t).observe(target).apparent()
            alt, az, _ = astrometric.altaz()
            altitude_deg = float(alt.degrees)
            azimuth_deg = float(az.degrees)
            if with_curve:
                curve.append(
                    AltAzPoint(time=sample, altitude_deg=altitude_deg, azimuth_deg=azimuth_deg)
                )
            if prev_alt is not None:
                if prev_alt < 0 <= altitude_deg and rise_t is None:
                    rise_t = sample
                if prev_alt >= 0 > altitude_deg and set_t is None:
                    set_t = sample
            if altitude_deg > max_alt:
                max_alt = altitude_deg
                max_t = sample
                moon_apparent = observer.at(t).observe(moon).apparent()
                moon_sep_at_max = float(astrometric.separation_from(moon_apparent).degrees)
            prev_alt = altitude_deg
            sample += step

        score = _compute_score(max_alt, obj.magnitude, moon_sep_at_max)
        return ObjectVisibility(
            catalog_id=obj.id,
            name=obj.name,
            type=obj.type,
            constellation=obj.constellation,
            ra_deg=obj.ra_deg,
            dec_deg=obj.dec_deg,
            magnitude=obj.magnitude,
            max_altitude_deg=max_alt,
            transit_time=max_t,
            rise_time=rise_t,
            set_time=set_t,
            moon_separation_deg=moon_sep_at_max,
            score=score,
            altitude_curve=curve,
        )

    def _compute_ranking(
        self,
        latitude: float,
        longitude: float,
        elevation_m: float,
        window: ObservationWindow,
        min_altitude_deg: float,
        max_results: int,
        type_filter: Optional[list[str]],
    ) -> list[ObjectVisibility]:
        type_set = set(type_filter) if type_filter else None
        results: list[ObjectVisibility] = []
        for obj in all_objects():
            if type_set is not None and obj.type not in type_set:
                continue
            vis = self._compute_visibility(
                latitude, longitude, elevation_m, window, obj, sample_minutes=60, with_curve=False
            )
            if vis.max_altitude_deg < min_altitude_deg:
                continue
            results.append(vis)
        results.sort(key=lambda v: v.score, reverse=True)
        top = results[:max_results]
        # Re-sample altitude curve for top results so the UI can plot them.
        for v in top:
            obj = lookup_object(v.catalog_id)
            if obj is None:
                continue
            v.altitude_curve = self._compute_visibility(
                latitude,
                longitude,
                elevation_m,
                window,
                obj,
                sample_minutes=30,
                with_curve=True,
            ).altitude_curve
        return top
