"""Unit tests for :mod:`app.services.planner_service`.

The pure scoring helper is tested fully. Skyfield-backed behaviour is
exercised only when the DE440s ephemeris file is available locally.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from app.services.planner_service import PlannerService, _compute_score

EPHEMERIS_PATH = "/opt/ephemerides/de440s.bsp"
_HAVE_EPHEMERIS = Path(EPHEMERIS_PATH).exists()


class TestComputeScore:
    """Pure tests for the scoring formula."""

    def test_zero_altitude_yields_zero(self) -> None:
        assert _compute_score(0.0, 5.0, 90.0) == 0.0
        assert _compute_score(-10.0, 5.0, 90.0) == 0.0

    def test_score_monotonic_in_altitude(self) -> None:
        low = _compute_score(20.0, 5.0, 90.0)
        mid = _compute_score(45.0, 5.0, 90.0)
        high = _compute_score(80.0, 5.0, 90.0)
        assert low < mid < high

    def test_score_monotonic_in_brightness(self) -> None:
        # Brighter (lower magnitude) should score higher.
        bright = _compute_score(60.0, 3.0, 90.0)
        dim = _compute_score(60.0, 9.0, 90.0)
        assert bright > dim

    def test_moon_proximity_penalises_score(self) -> None:
        far = _compute_score(60.0, 5.0, 90.0)
        close = _compute_score(60.0, 5.0, 5.0)
        assert close < far

    def test_score_clamped_to_unit_range(self) -> None:
        result = _compute_score(90.0, -5.0, 180.0)
        assert 0.0 <= result <= 100.0


@pytest.mark.skipif(not _HAVE_EPHEMERIS, reason="DE440s ephemeris not available")
class TestNightWindow:
    """Skyfield-backed tests; require the DE440s ephemeris."""

    @pytest.mark.asyncio
    async def test_paris_summer_night_window_has_dark_period(self) -> None:
        planner = PlannerService(ephemeris_path=EPHEMERIS_PATH)
        window = await planner.night_window(
            latitude=48.8566,
            longitude=2.3522,
            elevation_m=35.0,
            day=date(2024, 7, 15),
            timezone_name="Europe/Paris",
        )
        assert window.astronomical_twilight_end < window.astronomical_twilight_start
        assert 0.0 <= window.moon_illumination <= 1.0
        assert 0.0 <= window.darkness_score <= 100.0
