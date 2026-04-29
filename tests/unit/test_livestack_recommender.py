"""Unit tests for the live-stack deterministic recommender.

The thresholds are calibrated for the **linear** accumulator (not the
autostretched preview), so the "healthy" baseline below uses very
small medians (~0.02) — that's how a properly exposed astro stack
actually looks before any stretch.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.livestack.recommender import (
    HistogramStats,
    compute_histogram_stats,
    compute_recommendations,
)


# ── compute_histogram_stats ──────────────────────────────────────────────────


def test_histogram_stats_monochrome_image():
    img = np.full((32, 32), 0.02, dtype=np.float32)
    stats = compute_histogram_stats(img, last_fwhm=2.5)
    assert stats.is_monochrome is True
    assert stats.median_r == pytest.approx(0.02)
    assert stats.median_g == pytest.approx(0.02)
    assert stats.median_b == pytest.approx(0.02)
    assert stats.last_fwhm == 2.5
    assert stats.median_luminance == pytest.approx(0.02)


def test_histogram_stats_rgb_image_per_channel_medians():
    img = np.zeros((10, 10, 3), dtype=np.float32)
    img[..., 0] = 0.04  # red brighter
    img[..., 1] = 0.02  # green nominal
    img[..., 2] = 0.01  # blue dimmer
    stats = compute_histogram_stats(img)
    assert stats.is_monochrome is False
    assert stats.median_r == pytest.approx(0.04)
    assert stats.median_g == pytest.approx(0.02)
    assert stats.median_b == pytest.approx(0.01)


def test_histogram_stats_clipping_counted():
    img = np.zeros((100, 100), dtype=np.float32)
    img[:10, :] = 1.0  # ~10 % saturated
    img[10:20, :] = 0.5  # mid-tones, neither clipped low nor high
    stats = compute_histogram_stats(img)
    assert stats.clip_high_pct == pytest.approx(10.0)
    # 80 % of pixels still at zero
    assert stats.clip_low_pct == pytest.approx(80.0)


# ── compute_recommendations: rule branches ───────────────────────────────────


def _stats(**kwargs) -> HistogramStats:
    """Build a HistogramStats with reasonable defaults for a healthy
    astro stack (linear sky-background median around 0.02)."""
    base = dict(
        median_r=0.02,
        median_g=0.02,
        median_b=0.02,
        clip_low_pct=0.0,
        clip_high_pct=0.0,
        last_fwhm=2.5,
        is_monochrome=False,
    )
    base.update(kwargs)
    return HistogramStats(**base)


def test_healthy_image_returns_single_info_recommendation():
    """A properly exposed astro stack should not trigger any warning."""
    report = compute_recommendations(_stats())
    assert len(report.recommendations) == 1
    assert report.recommendations[0].severity == "info"
    assert report.recommendations[0].category == "general"


def test_under_floor_triggers_critical():
    """Below the read-noise floor → critical exposure warning."""
    report = compute_recommendations(_stats(median_r=0.001, median_g=0.001, median_b=0.001))
    severities = [r.severity for r in report.recommendations]
    categories = [r.category for r in report.recommendations]
    assert "critical" in severities
    assert "exposure" in categories
    assert report.recommendations[0].severity == "critical"


def test_low_but_safe_triggers_only_info():
    """Below the LOW threshold but above the floor → info, not warn."""
    report = compute_recommendations(_stats(median_r=0.005, median_g=0.005, median_b=0.005))
    exposure_recs = [r for r in report.recommendations if r.category == "exposure"]
    assert len(exposure_recs) == 1
    assert exposure_recs[0].severity == "info"


def test_over_high_triggers_warning():
    """Above the HIGH threshold → warn (sky glow / over-exposure)."""
    report = compute_recommendations(_stats(median_r=0.20, median_g=0.20, median_b=0.20))
    exposure_recs = [r for r in report.recommendations if r.category == "exposure"]
    assert exposure_recs and exposure_recs[0].severity == "warn"


def test_over_critical_triggers_critical():
    """Above the CRITICAL threshold → critical (severe over-exposure)."""
    report = compute_recommendations(_stats(median_r=0.40, median_g=0.40, median_b=0.40))
    exposure_recs = [r for r in report.recommendations if r.category == "exposure"]
    assert exposure_recs and exposure_recs[0].severity == "critical"


def test_high_clipping_triggers_critical():
    report = compute_recommendations(_stats(clip_high_pct=5.0))
    crits = [r for r in report.recommendations if r.severity == "critical"]
    assert any("saturated" in r.message.lower() for r in crits)


def test_white_balance_warn_on_red_dominant():
    """Red channel +50 % above green → critical-warn band."""
    report = compute_recommendations(_stats(median_r=0.030, median_g=0.020, median_b=0.020))
    wb_recs = [r for r in report.recommendations if r.category == "white_balance"]
    assert wb_recs
    assert wb_recs[0].severity == "warn"
    assert "red" in wb_recs[0].message.lower()


def test_white_balance_info_on_slight_blue_dominant():
    """Blue channel +25 % above green → info band."""
    report = compute_recommendations(_stats(median_r=0.020, median_g=0.020, median_b=0.025))
    wb_recs = [r for r in report.recommendations if r.category == "white_balance"]
    assert wb_recs and wb_recs[0].severity == "info"
    assert "blue" in wb_recs[0].message.lower()


def test_white_balance_skipped_for_monochrome():
    report = compute_recommendations(
        _stats(median_r=0.02, median_g=0.02, median_b=0.02, is_monochrome=True)
    )
    wb_recs = [r for r in report.recommendations if r.category == "white_balance"]
    assert wb_recs == []


def test_fwhm_warn_above_4_pixels():
    report = compute_recommendations(_stats(last_fwhm=4.5))
    focus_recs = [r for r in report.recommendations if r.category == "focus"]
    assert focus_recs and focus_recs[0].severity == "warn"


def test_fwhm_critical_above_6_pixels():
    report = compute_recommendations(_stats(last_fwhm=7.0))
    focus_recs = [r for r in report.recommendations if r.category == "focus"]
    assert focus_recs and focus_recs[0].severity == "critical"


def test_recommendations_sorted_critical_first():
    report = compute_recommendations(
        _stats(
            median_r=0.001, median_g=0.001, median_b=0.001,  # critical exposure
            last_fwhm=4.5,  # warn focus
        )
    )
    severities = [r.severity for r in report.recommendations]
    assert severities[0] == "critical"
    assert "warn" in severities


def test_messages_are_english():
    """Sanity check — output strings must not contain French diacritics."""
    report = compute_recommendations(_stats(median_r=0.40, median_g=0.40, median_b=0.40))
    for rec in report.recommendations:
        text = f"{rec.message} {rec.action}"
        assert not any(c in text for c in "éèêàùçôîûœ"), (
            f"Found French character in: {text!r}"
        )
