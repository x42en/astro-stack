"""Deterministic exposure / colour-balance recommender.

Inspects the running live-stack accumulator and emits actionable
suggestions for the next acquisition (move ISO up/down, lengthen the
exposure, adjust the camera white balance, refocus, ...).

The implementation is intentionally rule-based: the goal is to give
fast, reproducible feedback during an acquisition session.  A future
``OllamaRecommender`` can wrap this module and feed its output as
context to a large-language model for richer narrative tips.
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
    """Summary statistics of the current accumulator.

    Attributes:
        median_r: Median value of the red channel in [0, 1].
        median_g: Median value of the green channel in [0, 1].
        median_b: Median value of the blue channel in [0, 1].
        clip_low_pct: Percentage of pixels clipped to 0 across all channels.
        clip_high_pct: Percentage of pixels saturated to 1 across all channels.
        last_fwhm: Last measured star FWHM in pixels (``None`` if unknown).
        is_monochrome: True when no per-channel statistics are available
            (the accumulator has shape ``(H, W)`` rather than ``(H, W, 3)``).
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
        message: One-sentence description of the issue, in French.
        action: Concrete actionable step the user can take, in French.
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
            Empty when everything looks healthy.
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
        # Clipping is rare in 32-bit float intermediates but we still
        # report it so the rule engine can warn the user about it.
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


# Target median luminance for a properly exposed astro-frame stack.
# Below ~0.10 we are under-exposed; above ~0.45 we start losing the
# faintest signal to the noise floor and risk saturating bright stars.
_TARGET_MEDIAN_MIN = 0.10
_TARGET_MEDIAN_MAX = 0.45

# Tolerance around 1.0 for the colour-channel ratios.  Astro CMOS sensors
# typically have a green peak so we expect R/G and B/G slightly below 1.
_WB_RATIO_WARN = 0.15  # 15 % deviation
_WB_RATIO_CRITICAL = 0.35

# FWHM thresholds expressed in pixels.  Tuned for typical small-pixel
# DSLR / cooled-CMOS setups; users with very long focal lengths may
# want to raise these via configuration in a future iteration.
_FWHM_WARN = 4.0
_FWHM_CRITICAL = 6.0

_HIGH_CLIP_WARN_PCT = 1.0
_LOW_CLIP_WARN_PCT = 5.0


def compute_recommendations(stats: HistogramStats) -> RecommendationReport:
    """Apply the deterministic rule set to ``stats``.

    The rules are deliberately conservative: each one produces at most
    one recommendation. We sort the output by severity (critical first)
    so the UI can render the most urgent advice at the top.
    """
    recs: list[Recommendation] = []

    luminance = stats.median_luminance

    # ── Exposure / ISO ──────────────────────────────────────────────────────
    if luminance < _TARGET_MEDIAN_MIN:
        gap = _TARGET_MEDIAN_MIN - luminance
        severity: Severity = "warn" if gap < 0.05 else "critical"
        recs.append(
            Recommendation(
                severity=severity,
                category="exposure",
                message=(
                    f"Image très sombre (médiane luminance {luminance:.2f}, "
                    f"cible ≥ {_TARGET_MEDIAN_MIN:.2f})."
                ),
                action=(
                    "Allonger la pose unitaire d'environ +1 EV, ou monter "
                    "les ISO d'un cran si la pose ne peut pas être rallongée."
                ),
            )
        )
    elif luminance > _TARGET_MEDIAN_MAX:
        recs.append(
            Recommendation(
                severity="warn",
                category="exposure",
                message=(
                    f"Image très lumineuse (médiane luminance {luminance:.2f}, "
                    f"cible ≤ {_TARGET_MEDIAN_MAX:.2f})."
                ),
                action=(
                    "Réduire la pose unitaire d'environ -1 EV ou baisser les "
                    "ISO pour préserver les hautes lumières et l'étoile centrale."
                ),
            )
        )

    if stats.clip_high_pct >= _HIGH_CLIP_WARN_PCT:
        recs.append(
            Recommendation(
                severity="critical" if stats.clip_high_pct >= 5 else "warn",
                category="exposure",
                message=(
                    f"{stats.clip_high_pct:.1f} % de pixels saturés."
                ),
                action=(
                    "Réduire la pose ou les ISO ; envisager de poser plus court "
                    "et de stacker davantage de frames pour préserver les étoiles brillantes."
                ),
            )
        )

    if (
        stats.clip_low_pct >= _LOW_CLIP_WARN_PCT
        and luminance < _TARGET_MEDIAN_MIN
    ):
        recs.append(
            Recommendation(
                severity="info",
                category="iso",
                message=(
                    f"{stats.clip_low_pct:.1f} % de pixels au plancher (signal noyé dans le bruit)."
                ),
                action=(
                    "Préférer une pose plus longue à une montée d'ISO : la dynamique "
                    "se préserve mieux à ISO modéré et le signal sortira du bruit."
                ),
            )
        )

    # ── White balance ───────────────────────────────────────────────────────
    if not stats.is_monochrome:
        green = max(stats.median_g, 1e-6)
        ratio_r = stats.median_r / green
        ratio_b = stats.median_b / green

        for label, ratio, dominant in (
            ("rouge", ratio_r, "rouge"),
            ("bleu", ratio_b, "bleu"),
        ):
            deviation = abs(ratio - 1.0)
            if deviation >= _WB_RATIO_CRITICAL:
                recs.append(
                    Recommendation(
                        severity="warn",
                        category="white_balance",
                        message=(
                            f"Forte dominante {dominant} (ratio {label}/vert = "
                            f"{ratio:.2f})."
                        ),
                        action=(
                            "Vérifier la balance des blancs caméra (mode "
                            "« Lumière du jour » ou « Auto »). Un slider RGB "
                            "compense visuellement, mais la correction matérielle "
                            "reste préférable pour préserver la dynamique."
                        ),
                    )
                )
            elif deviation >= _WB_RATIO_WARN:
                recs.append(
                    Recommendation(
                        severity="info",
                        category="white_balance",
                        message=(
                            f"Légère dominante {dominant} (ratio {label}/vert = "
                            f"{ratio:.2f})."
                        ),
                        action=(
                            f"Ajuster le gain {label} dans le panneau « RGB balance » "
                            "ou affiner la balance des blancs caméra."
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
                        f"FWHM dégradée ({stats.last_fwhm:.1f} px) — étoiles très molles."
                    ),
                    action=(
                        "Refaire la mise au point (masque de Bahtinov, autofocus) ou "
                        "vérifier le suivi. Frames à écarter au stacking si la dérive persiste."
                    ),
                )
            )
        elif stats.last_fwhm >= _FWHM_WARN:
            recs.append(
                Recommendation(
                    severity="warn",
                    category="focus",
                    message=(
                        f"FWHM élevée ({stats.last_fwhm:.1f} px) — perte de piqué."
                    ),
                    action=(
                        "Contrôler la mise au point ; la dérive thermique peut justifier "
                        "un ré-ajustement toutes les 30 minutes."
                    ),
                )
            )

    if not recs:
        recs.append(
            Recommendation(
                severity="info",
                category="general",
                message="Acquisition saine — médiane et balance des blancs dans les clous.",
                action="Continuer ainsi ; surveiller la dérive de FWHM dans le temps.",
            )
        )

    # Critical first, then warn, then info.
    severity_rank = {"critical": 0, "warn": 1, "info": 2}
    recs.sort(key=lambda r: severity_rank[r.severity])

    return RecommendationReport(stats=stats, recommendations=recs)
