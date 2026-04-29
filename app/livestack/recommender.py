"""Deterministic exposure / colour-balance recommender.

Inspects the running live-stack accumulator and emits actionable
suggestions for the next acquisition (raise/lower ISO, lengthen the
exposure, adjust the camera white balance, refocus, ...).

The implementation is intentionally rule-based: the goal is to give
fast, reproducible feedback during an acquisition session. A future
``OllamaRecommender`` can wrap this module and feed its output as
context to a large-language model for richer narrative tips.

Important — what the stats describe
-----------------------------------
The accumulator is the **linear** running mean of aligned, normalised
frames in ``[0, 1]``. It is NOT the autostretched preview the user
sees in the canvas. Astro frames are dominated by a dark sky
background, so a healthy linear median sits very low (typically
``0.005`` – ``0.05``); the autostretch only lifts that range to a
display-friendly ``0.20`` – ``0.30`` for visualisation. All exposure
thresholds below are calibrated for that linear regime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

__all__ = [
    "HistogramStats",
    "Recommendation",
    "RecommendationReport",
    "compute_histogram_stats",
    "compute_recommendations",
]


Severity = Literal["info", "warn", "critical"]
Category = Literal["exposure", "iso", "white_balance", "focus", "general"]


@dataclass(slots=True)
class HistogramStats:
    """Summary statistics of the current (linear) accumulator.

    Attributes:
        median_r: Median value of the red channel in [0, 1].
        median_g: Median value of the green channel in [0, 1].
        median_b: Median value of the blue channel in [0, 1].
        clip_low_pct: Percentage of pixels at the sensor floor across
            all channels (linear values <= 1e-4).
        clip_high_pct: Percentage of pixels saturated to 1 across all
            channels.
        last_fwhm: Last measured star FWHM in pixels (``None`` if
            unknown).
        is_monochrome: True when no per-channel statistics are
            available (the accumulator has shape ``(H, W)`` rather
            than ``(H, W, 3)``).
    """

    median_r: float
    median_g: float
    median_b: float
    clip_low_pct: float
    clip_high_pct: float
    last_fwhm: Optional[float] = None
    is_monochrome: bool = False

    @property
    def median_luminance(self) -> float:
        """Rec.709 luminance from the per-channel medians."""
        return 0.2126 * self.median_r + 0.7152 * self.median_g + 0.0722 * self.median_b


@dataclass(slots=True)
class Recommendation:
    """A single suggestion surfaced to the user.

    Attributes:
        severity: ``info`` / ``warn`` / ``critical``. The UI uses this
            to pick an icon and accent colour.
        category: High-level grouping for sorting / filtering.
        message: One-sentence description of the issue.
        action: Concrete actionable step the user can take.
    """

    severity: Severity
    category: Category
    message: str
    action: str


@dataclass(slots=True)
class RecommendationReport:
    """Bundle returned by :func:`compute_recommendations`.

    Attributes:
        stats: The :class:`HistogramStats` the recommendations were
            derived from. Embedded so the client can render the same
            numbers next to the advice without an extra round-trip.
        recommendations: Ordered list of :class:`Recommendation`.
            Always contains at least one entry (a positive "all good"
            note when no rule fires).
    """

    stats: HistogramStats
    recommendations: list[Recommendation] = field(default_factory=list)


# ── Statistics ────────────────────────────────────────────────────────────────


def compute_histogram_stats(
    image: np.ndarray,
    last_fwhm: Optional[float] = None,
) -> HistogramStats:
    """Extract per-channel medians + clipping ratios from ``image``.

    Args:
        image: float32 array in [0, 1], either ``(H, W)`` (monochrome)
            or ``(H, W, 3)`` (RGB).
        last_fwhm: Optionally, the last estimated star FWHM in pixels
            (forwarded as-is for downstream rules to consume).

    Returns:
        A populated :class:`HistogramStats` instance.
    """
    if image.ndim == 2:
        med = float(np.median(image))
        clip_low = float(np.mean(image <= 1e-4) * 100.0)
        clip_high = float(np.mean(image >= 1.0 - 1e-4) * 100.0)
        return HistogramStats(
            median_r=med,
            median_g=med,
            median_b=med,
            clip_low_pct=clip_low,
            clip_high_pct=clip_high,
            last_fwhm=last_fwhm,
            is_monochrome=True,
        )

    if image.ndim != 3 or image.shape[-1] != 3:  # pragma: no cover - defensive
        raise ValueError(
            f"Expected (H,W) or (H,W,3) image, got shape {image.shape}.",
        )

    med_r = float(np.median(image[..., 0]))
    med_g = float(np.median(image[..., 1]))
    med_b = float(np.median(image[..., 2]))
    clip_low = float(np.mean(image <= 1e-4) * 100.0)
    clip_high = float(np.mean(image >= 1.0 - 1e-4) * 100.0)
    return HistogramStats(
        median_r=med_r,
        median_g=med_g,
        median_b=med_b,
        clip_low_pct=clip_low,
        clip_high_pct=clip_high,
        last_fwhm=last_fwhm,
        is_monochrome=False,
    )


# ── Rule engine ───────────────────────────────────────────────────────────────


# Linear-stack thresholds for the median sky-background luminance.
#
# Astrophotography stacks are dominated by a dark sky background. After
# normalisation a healthy median typically sits between ~0.005 and
# ~0.05 (slightly higher under heavy light pollution). Below the floor
# the signal is buried in the sensor read noise; above the ceiling we
# are either over-exposing or fighting strong sky glow.
_LINEAR_MEDIAN_FLOOR = 0.003   # below: under-exposed / read-noise limited
_LINEAR_MEDIAN_LOW = 0.008     # below: shorter exposure is wasteful
_LINEAR_MEDIAN_HIGH = 0.15     # above: sky glow / over-exposure suspected
_LINEAR_MEDIAN_CRITICAL = 0.30 # above: heavily over-exposed or huge LP

# Saturation thresholds measured on the linear stack.
_HIGH_CLIP_WARN_PCT = 0.5
_HIGH_CLIP_CRIT_PCT = 3.0

# Tolerance around 1.0 for the colour-channel ratios. Most astro CMOS
# sensors peak in green so we expect R/G and B/G slightly below 1.
_WB_RATIO_WARN = 0.20      # 20 % deviation
_WB_RATIO_CRITICAL = 0.40

# FWHM thresholds in pixels, tuned for typical small-pixel
# DSLR / cooled-CMOS setups. Long focal length users may want to raise
# these via configuration in a future iteration.
_FWHM_WARN = 4.0
_FWHM_CRITICAL = 6.0


def compute_recommendations(stats: HistogramStats) -> RecommendationReport:
    """Apply the deterministic rule set to ``stats``.

    The rules are deliberately conservative: each one produces at most
    one recommendation. We sort the output by severity (critical first)
    so the UI can render the most urgent advice at the top.
    """
    recs: list[Recommendation] = []

    luminance = stats.median_luminance

    # ── Exposure / ISO ──────────────────────────────────────────────────────
    if luminance < _LINEAR_MEDIAN_FLOOR:
        recs.append(
            Recommendation(
                severity="critical",
                category="exposure",
                message=(
                    f"Linear median {luminance:.4f} is below the read-noise "
                    f"floor (~{_LINEAR_MEDIAN_FLOOR:.3f})."
                ),
                action=(
                    "Lengthen the sub-exposure (try +1 EV) or raise ISO so "
                    "the sky background sits clearly above the sensor read "
                    "noise."
                ),
            )
        )
    elif luminance < _LINEAR_MEDIAN_LOW:
        recs.append(
            Recommendation(
                severity="info",
                category="exposure",
                message=(
                    f"Linear median {luminance:.4f} is on the low side."
                ),
                action=(
                    "If your tracking allows, slightly longer subs would "
                    "improve signal-to-noise. Otherwise, this exposure is "
                    "perfectly safe."
                ),
            )
        )
    elif luminance > _LINEAR_MEDIAN_CRITICAL:
        recs.append(
            Recommendation(
                severity="critical",
                category="exposure",
                message=(
                    f"Linear median {luminance:.3f} is very high — strong "
                    "sky glow or severely over-exposed."
                ),
                action=(
                    "Shorten the sub-exposure or lower ISO; check for "
                    "nearby light pollution and consider a light-pollution "
                    "filter."
                ),
            )
        )
    elif luminance > _LINEAR_MEDIAN_HIGH:
        recs.append(
            Recommendation(
                severity="warn",
                category="exposure",
                message=(
                    f"Linear median {luminance:.3f} suggests heavy sky "
                    "background (light pollution or too much EV)."
                ),
                action=(
                    "Try shortening the sub-exposure by ~1 EV, or use a "
                    "narrowband / LP filter to reclaim dynamic range for "
                    "the target."
                ),
            )
        )

    if stats.clip_high_pct >= _HIGH_CLIP_WARN_PCT:
        is_critical = stats.clip_high_pct >= _HIGH_CLIP_CRIT_PCT
        recs.append(
            Recommendation(
                severity="critical" if is_critical else "warn",
                category="exposure",
                message=(
                    f"{stats.clip_high_pct:.1f}% of pixels are saturated."
                ),
                action=(
                    "Shorten the sub-exposure or lower ISO. A few clipped "
                    "star cores are unavoidable, but more than a fraction "
                    "of a percent costs you star colours and bright nebula "
                    "detail."
                ),
            )
        )

    # ── White balance ───────────────────────────────────────────────────────
    if not stats.is_monochrome:
        green = max(stats.median_g, 1e-6)
        ratio_r = stats.median_r / green
        ratio_b = stats.median_b / green

        for label, ratio in (("red", ratio_r), ("blue", ratio_b)):
            deviation = abs(ratio - 1.0)
            if deviation >= _WB_RATIO_CRITICAL:
                recs.append(
                    Recommendation(
                        severity="warn",
                        category="white_balance",
                        message=(
                            f"Strong {label} cast ({label}/green ratio = "
                            f"{ratio:.2f})."
                        ),
                        action=(
                            "Check the camera white balance (Daylight or "
                            "Auto are usually safest). The RGB sliders in "
                            "the panel below are a visual aid only — a "
                            "hardware fix preserves more dynamic range."
                        ),
                    )
                )
            elif deviation >= _WB_RATIO_WARN:
                recs.append(
                    Recommendation(
                        severity="info",
                        category="white_balance",
                        message=(
                            f"Mild {label} cast ({label}/green ratio = "
                            f"{ratio:.2f})."
                        ),
                        action=(
                            f"Trim the {label} gain in the RGB balance "
                            "panel or fine-tune the camera white balance."
                        ),
                    )
                )

    # ── Focus / tracking ────────────────────────────────────────────────────
    if stats.last_fwhm is not None:
        if stats.last_fwhm >= _FWHM_CRITICAL:
            recs.append(
                Recommendation(
                    severity="critical",
                    category="focus",
                    message=(
                        f"FWHM is degraded ({stats.last_fwhm:.1f} px) — "
                        "stars are very soft."
                    ),
                    action=(
                        "Refocus (Bahtinov mask, autofocus) or check the "
                        "mount tracking. Discard frames with persistent "
                        "drift before the next stack."
                    ),
                )
            )
        elif stats.last_fwhm >= _FWHM_WARN:
            recs.append(
                Recommendation(
                    severity="warn",
                    category="focus",
                    message=(
                        f"FWHM is on the high side ({stats.last_fwhm:.1f} "
                        "px) — resolution is suffering."
                    ),
                    action=(
                        "Verify focus; thermal drift can warrant a "
                        "refocus every 30 minutes or so."
                    ),
                )
            )

    if not recs:
        recs.append(
            Recommendation(
                severity="info",
                category="general",
                message=(
                    "Acquisition looks healthy — exposure, balance and "
                    "focus are all in range."
                ),
                action="Keep going, and watch FWHM for thermal drift over time.",
            )
        )

    severity_rank = {"critical": 0, "warn": 1, "info": 2}
    recs.sort(key=lambda r: severity_rank[r.severity])

    return RecommendationReport(stats=stats, recommendations=recs)
