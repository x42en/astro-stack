"""Unit tests for the live-stack deterministic recommender."""

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
    img = np.full((32, 32), 0.3, dtype=np.float32)
    stats = compute_histogram_stats(img, last_fwhm=2.5)
    assert stats.is_monochrome is True
    assert stats.median_r == pytest.approx(0.3)
    assert stats.median_g == pytest.approx(0.3)
    assert stats.median_b == pytest.approx(0.3)
    assert stats.last_fwhm == 2.5
    assert stats.median_luminance == pytest.approx(0.3)


def test_histogram_stats_rgb_image_per_channel_medians():
    img = np.zeros((10, 10, 3), dtype=np.float32)
    img[..., 0] = 0.4  # red bright
    img[..., 1] = 0.2  # green dimmer
    img[..., 2] = 0.1  # blue dimmer still
    stats = compute_histogram_stats(img)
    assert stats.is_monochrome is False
    assert stats.median_r == pytest.approx(0.4)
    assert stats.median_g == pytest.approx(0.2)
    assert stats.median_b == pytest.approx(0.1)


def test_histogram_stats_clipping_counted():
    img = np.zeros((100, 100), dtype=np.float32)
    img[:10, :] = 1.0  # ~10 % saturated
    img[10:20, :] = 0.0  # already ~10 % at floor (the rest is also 0)
    stats = compute_histogram_stats(img)
    assert stats.clip_high_pct == pytest.approx(10.0)
    # everything except top 10% rows is zero -> 90 % at floor
    assert stats.clip_low_pct == pytest.approx(90.0)


# ── compute_recommendations: rule branches ───────────────────────────────────


def _stats(**kwargs) -> HistogramStats:
    """Build a HistogramStats with reasonable defaults (healthy image)."""
    base = dict(
        median_r=0.22,
        median_g=0.22,
        median_b=0.22,
        clip_low_pct=0.0,
        clip_high_pct=0.0,
        last_fwhm=2.5,
        is_monochrome=False,
    )
    base.update(kwargs)
    return HistogramStats(**base)


def test_healthy_image_returns_single_info_recommendation():
    report = compute_recommendations(_stats())
    assert len(report.recommendations) == 1
    assert report.recommendations[0].severity == "info"
    assert report.recommendations[0].category == "general"


def test_underexposed_triggers_critical_when_far_off():
    report = compute_recommendations(_stats(median_r=0.02, median_g=0.02, median_b=0.02))
    severities = [r.severity for r in report.recommendations]
    categories = [r.category for r in report.recommendations]
    assert "critical" in severities
    assert "exposure" in categories
    # critical first
    assert report.recommendations[0].severity == "critical"


def test_underexposed_triggers_warn_when_close_to_threshold():
    report = compute_recommendations(_stats(median_r=0.07, median_g=0.07, median_b=0.07))
    exposure_recs = [r for r in report.recommendations if r.category == "exposure"]
    assert len(exposure_recs) == 1
    assert exposure_recs[0].severity == "warn"


def test_overexposed_image_triggers_warning():
    report = compute_recommendations(_stats(median_r=0.6, median_g=0.6, median_b=0.6))
    exposure_recs = [r for r in report.recommendations if r.category == "exposure"]
    assert exposure_recs and exposure_recs[0].severity == "warn"


def test_high_clipping_triggers_critical():
    report = compute_recommendations(_stats(clip_high_pct=8.0))
    crits = [r for r in report.recommendations if r.severity == "critical"]
    assert any("saturé" in r.message for r in crits)


def test_white_balance_warn_on_red_dominant():
    # Red channel is +35 % above green: critical-warn band.
    report = compute_recommendations(_stats(median_r=0.30, median_g=0.20, median_b=0.20))
    wb_recs = [r for r in report.recommendations if r.category == "white_balance"]
    assert wb_recs
    assert wb_recs[0].severity == "warn"
    assert "rouge" in wb_recs[0].message.lower()


def test_white_balance_info_on_slight_blue_dominant():
    # Blue +20 %: info band.
    report = compute_recommendations(_stats(median_r=0.20, median_g=0.20, median_b=0.24))
    wb_recs = [r for r in report.recommendations if r.category == "white_balance"]
    assert wb_recs and wb_recs[0].severity == "info"
    assert "bleu" in wb_recs[0].message.lower()


def test_white_balance_skipped_for_monochrome():
    report = compute_recommendations(
        _stats(median_r=0.22, median_g=0.22, median_b=0.22, is_monochrome=True)
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
            median_r=0.02, median_g=0.02, median_b=0.02,  # critical exposure
            last_fwhm=4.5,  # warn focus
        )
    )
    severities = [r.severity for r in report.recommendations]
    # First critical, then warn, no info should leak in
    assert severities[0] == "critical"
    assert "warn" in severities
