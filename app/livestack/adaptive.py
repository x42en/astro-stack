"""Adaptive vision-critic tuning for the live-stacking preview loop.

Applies the same :class:`~app.pipeline.adaptive.critic.VisionCritic` used by
the batch pipeline's adaptive loop (see ``app/pipeline/adaptive/``) to the
live-stacking MTF autostretch parameters (``target_bkg``, ``shadows_clip`` —
see :mod:`app.livestack.autostretch`).

This is deliberately a *separate, much smaller* mechanism than the batch
``run_adaptive_loop``/LangGraph loop, because live-stacking iterates over
real wall-clock time (one incoming frame every ~30s) rather than a tight
in-process retry loop: each evaluation here is a single critique call
triggered by frame arrival, not a synchronous multi-step graph. Denoise/
sharpen/gradient-removal are intentionally out of scope for v1 — they are
GPU-bound (batch-only today) and would reintroduce the VRAM contention risk
already flagged against the shared vLLM GPU allocation; only the CPU-cheap
stretch parameters are tunable here.

Behaviour (see ``Settings.live_adaptive_critic_*``):

* No evaluation runs before ``live_adaptive_critic_warmup_frames`` frames
  have been accepted (early stacks are too noisy to judge).
* After warm-up, at most one evaluation runs every
  ``live_adaptive_critic_recheck_every`` newly accepted frames.
* Once the critic reports satisfaction, **or**
  ``live_adaptive_critic_max_attempts`` evaluations have run, the last
  proposed parameters are frozen (``adaptive_converged=True``) and reused
  for every subsequent frame with no further critic calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from app.core.logging import get_logger
from app.livestack.recommender import HistogramStats
from app.livestack.state import LiveStackState
from app.pipeline.adaptive.critic import VisionCritic

logger = get_logger(__name__)

# Bounds for the two live-tunable parameters. Deliberately narrower than a
# blind numeric range: values far outside these are very unlikely to be a
# genuine improvement over the PixInsight-derived defaults
# (target_bkg=0.25, shadows_clip=-2.8) and are more likely a critic mistake.
_TARGET_BKG_BOUNDS = (0.10, 0.45)
_SHADOWS_CLIP_BOUNDS = (-6.0, -0.5)

_CAPABILITY_CONTEXT = (
    "target_bkg (float, desired background luminance after stretch, "
    "default 0.25, higher = brighter background) | "
    "shadows_clip (float, negative multiplier of the average deviation "
    "controlling how much faint signal is kept vs clipped to black, "
    "default -2.8, more negative = keeps more faint signal but noisier "
    "background)"
)


@dataclass(slots=True)
class LiveAdaptiveDecision:
    """Outcome of one live-critic evaluation.

    Attributes:
        target_bkg: Value to use for subsequent frames.
        shadows_clip: Value to use for subsequent frames.
        satisfied: Whether the critic accepted the current preview as-is.
        reasoning: Critic's short English explanation, for the state's
            ``adaptive_history``.
        confidence: Critic's self-reported confidence (0.0-1.0).
    """

    target_bkg: float
    shadows_clip: float
    satisfied: bool
    reasoning: str
    confidence: float


def should_evaluate(
    state: LiveStackState,
    *,
    warmup_frames: int,
    recheck_every: int,
    max_attempts: int,
) -> bool:
    """Decide whether a live-critic evaluation should run for this frame.

    Args:
        state: Current live-stack state (post frame-ingestion).
        warmup_frames: Minimum accepted frames before any evaluation.
        recheck_every: Minimum frame spacing between evaluations.
        max_attempts: Hard cap on evaluations for this session.

    Returns:
        ``True`` if an evaluation should be triggered now.
    """
    if state.adaptive_converged:
        return False
    if state.adaptive_attempts >= max_attempts:
        return False
    if state.frame_count < warmup_frames:
        return False
    frames_since_last = state.frame_count - state.adaptive_last_evaluated_frame_count
    return frames_since_last >= recheck_every


def _sanitize_live_patch(patch: dict[str, Any], current: LiveAdaptiveDecision) -> tuple[float, float]:
    """Clamp a critic-proposed patch to the live-tunable bounds.

    Unknown fields are ignored; missing fields fall back to ``current``.

    Args:
        patch: Raw patch proposed by the critic.
        current: Values currently in effect (fallback for missing fields).

    Returns:
        ``(target_bkg, shadows_clip)`` clamped into their respective bounds.
    """
    target_bkg = current.target_bkg
    shadows_clip = current.shadows_clip

    raw_bkg = patch.get("target_bkg")
    if isinstance(raw_bkg, (int, float)) and not isinstance(raw_bkg, bool):
        target_bkg = min(max(float(raw_bkg), _TARGET_BKG_BOUNDS[0]), _TARGET_BKG_BOUNDS[1])

    raw_shadows = patch.get("shadows_clip")
    if isinstance(raw_shadows, (int, float)) and not isinstance(raw_shadows, bool):
        shadows_clip = min(
            max(float(raw_shadows), _SHADOWS_CLIP_BOUNDS[0]), _SHADOWS_CLIP_BOUNDS[1]
        )

    return target_bkg, shadows_clip


async def evaluate(
    *,
    critic: VisionCritic,
    preview_jpeg_path: Path,
    stats: HistogramStats,
    state: LiveStackState,
    default_target_bkg: float = 0.25,
    default_shadows_clip: float = -2.8,
) -> LiveAdaptiveDecision:
    """Run one live-critic evaluation and return the decision.

    A critic that is unreachable or returns a malformed response never
    raises: it degrades to "satisfied" with the current parameters
    unchanged, so a broken vLLM endpoint can never break live-stacking
    (which must keep working for novices with the feature off or
    misconfigured).

    Args:
        critic: Vision critic client (see :class:`VisionCritic`).
        preview_jpeg_path: Path to the just-rendered preview JPEG.
        stats: Linear-accumulator histogram statistics for this frame.
        state: Current live-stack state (read-only here; the caller persists
            the returned decision).
        default_target_bkg: Fallback when no override has been set yet.
        default_shadows_clip: Fallback when no override has been set yet.

    Returns:
        The :class:`LiveAdaptiveDecision` to apply from now on.
    """
    current = LiveAdaptiveDecision(
        target_bkg=state.adaptive_target_bkg or default_target_bkg,
        shadows_clip=state.adaptive_shadows_clip or default_shadows_clip,
        satisfied=False,
        reasoning="",
        confidence=0.0,
    )

    current_values = {
        "target_bkg": current.target_bkg,
        "shadows_clip": current.shadows_clip,
    }
    live_stats = {
        "median_r": stats.median_r,
        "median_g": stats.median_g,
        "median_b": stats.median_b,
        "clip_low_pct": stats.clip_low_pct,
        "clip_high_pct": stats.clip_high_pct,
        "fwhm": stats.last_fwhm,
        "frame_count": state.frame_count,
    }

    try:
        verdict = await critic.critique(
            step_name="livestack_autostretch",
            iteration=state.adaptive_attempts,
            max_iterations=state.adaptive_attempts + 1,
            capability_context=_CAPABILITY_CONTEXT,
            current_values=current_values,
            stats=live_stats,
            preview_jpeg_path=preview_jpeg_path,
            history=state.adaptive_history,
        )
    except Exception:  # noqa: BLE001
        logger.warning("live_adaptive_critic_unavailable", session_id=state.session_id, exc_info=True)
        current.satisfied = True
        current.reasoning = "Vision critic unavailable; keeping current parameters."
        return current

    target_bkg, shadows_clip = _sanitize_live_patch(verdict.patch, current)
    return LiveAdaptiveDecision(
        target_bkg=target_bkg,
        shadows_clip=shadows_clip,
        satisfied=verdict.satisfied,
        reasoning=verdict.reasoning,
        confidence=verdict.confidence,
    )


def apply_decision(state: LiveStackState, decision: LiveAdaptiveDecision) -> None:
    """Persist a :class:`LiveAdaptiveDecision` into ``state`` in place.

    Freezes future evaluations (``adaptive_converged=True``) once the
    critic is satisfied or the attempt budget is exhausted — matching the
    "find a good treatment once, then reuse it" behaviour.

    Args:
        state: Live-stack state to update (mutated in place).
        decision: Outcome of :func:`evaluate`.
    """
    state.adaptive_target_bkg = decision.target_bkg
    state.adaptive_shadows_clip = decision.shadows_clip
    state.adaptive_attempts += 1
    state.adaptive_last_evaluated_frame_count = state.frame_count
    state.adaptive_history.append(
        {
            "attempt": state.adaptive_attempts,
            "frame_count": state.frame_count,
            "satisfied": decision.satisfied,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning,
            "target_bkg": decision.target_bkg,
            "shadows_clip": decision.shadows_clip,
        }
    )
    # Kept small: max_attempts is a hard, low cap (default 5), so this list
    # never grows unbounded even without an explicit trim.
    if decision.satisfied:
        state.adaptive_converged = True
