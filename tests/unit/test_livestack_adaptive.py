"""Unit tests for :mod:`app.livestack.adaptive` (live-stack autostretch critic)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.livestack.adaptive import (
    LiveAdaptiveDecision,
    apply_decision,
    evaluate,
    should_evaluate,
)
from app.livestack.recommender import HistogramStats
from app.livestack.state import LiveStackState
from app.pipeline.adaptive.critic import CriticVerdict


class _StubCritic:
    def __init__(self, verdict: CriticVerdict | None = None, error: Exception | None = None) -> None:
        self.verdict = verdict
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def critique(self, **kwargs: Any) -> CriticVerdict:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        assert self.verdict is not None
        return self.verdict


def _preview_path(tmp_path: Path) -> Path:
    path = tmp_path / "preview.jpg"
    path.write_bytes(b"\xff\xd8\xff\xe0fake")
    return path


def _stats() -> HistogramStats:
    return HistogramStats(
        median_r=0.01, median_g=0.012, median_b=0.009,
        clip_low_pct=0.1, clip_high_pct=0.0, last_fwhm=3.2,
    )


class TestShouldEvaluate:
    def test_below_warmup_is_not_due(self) -> None:
        state = LiveStackState(session_id="s1", frame_count=3)
        assert should_evaluate(state, warmup_frames=5, recheck_every=3, max_attempts=5) is False

    def test_at_warmup_is_due(self) -> None:
        state = LiveStackState(session_id="s1", frame_count=5)
        assert should_evaluate(state, warmup_frames=5, recheck_every=3, max_attempts=5) is True

    def test_not_enough_spacing_since_last_eval(self) -> None:
        state = LiveStackState(
            session_id="s1", frame_count=6, adaptive_last_evaluated_frame_count=5
        )
        assert should_evaluate(state, warmup_frames=5, recheck_every=3, max_attempts=5) is False

    def test_enough_spacing_since_last_eval(self) -> None:
        state = LiveStackState(
            session_id="s1", frame_count=8, adaptive_last_evaluated_frame_count=5
        )
        assert should_evaluate(state, warmup_frames=5, recheck_every=3, max_attempts=5) is True

    def test_converged_is_never_due_again(self) -> None:
        state = LiveStackState(session_id="s1", frame_count=20, adaptive_converged=True)
        assert should_evaluate(state, warmup_frames=5, recheck_every=3, max_attempts=5) is False

    def test_max_attempts_reached_is_never_due_again(self) -> None:
        state = LiveStackState(session_id="s1", frame_count=20, adaptive_attempts=5)
        assert should_evaluate(state, warmup_frames=5, recheck_every=3, max_attempts=5) is False


class TestEvaluate:
    @pytest.mark.asyncio
    async def test_satisfied_verdict_keeps_current_values(self, tmp_path: Path) -> None:
        state = LiveStackState(session_id="s1", frame_count=5)
        critic = _StubCritic(
            verdict=CriticVerdict(satisfied=True, confidence=0.9, reasoning="Looks good.")
        )

        decision = await evaluate(
            critic=critic,  # type: ignore[arg-type]
            preview_jpeg_path=_preview_path(tmp_path),
            stats=_stats(),
            state=state,
        )

        assert decision.satisfied is True
        assert decision.target_bkg == 0.25
        assert decision.shadows_clip == -2.8
        assert critic.calls[0]["step_name"] == "livestack_autostretch"

    @pytest.mark.asyncio
    async def test_unsatisfied_verdict_applies_clamped_patch(self, tmp_path: Path) -> None:
        state = LiveStackState(session_id="s1", frame_count=5)
        critic = _StubCritic(
            verdict=CriticVerdict(
                satisfied=False, confidence=0.4, reasoning="Too dim.",
                patch={"target_bkg": 0.35, "shadows_clip": -3.5},
            )
        )

        decision = await evaluate(
            critic=critic,  # type: ignore[arg-type]
            preview_jpeg_path=_preview_path(tmp_path),
            stats=_stats(),
            state=state,
        )

        assert decision.satisfied is False
        assert decision.target_bkg == 0.35
        assert decision.shadows_clip == -3.5

    @pytest.mark.asyncio
    async def test_patch_out_of_bounds_is_clamped(self, tmp_path: Path) -> None:
        state = LiveStackState(session_id="s1", frame_count=5)
        critic = _StubCritic(
            verdict=CriticVerdict(
                satisfied=False, confidence=0.4, reasoning="extreme",
                patch={"target_bkg": 5.0, "shadows_clip": -50.0},
            )
        )

        decision = await evaluate(
            critic=critic,  # type: ignore[arg-type]
            preview_jpeg_path=_preview_path(tmp_path),
            stats=_stats(),
            state=state,
        )

        assert decision.target_bkg == 0.45  # upper bound
        assert decision.shadows_clip == -6.0  # lower bound

    @pytest.mark.asyncio
    async def test_reuses_previously_tuned_values_as_baseline(self, tmp_path: Path) -> None:
        state = LiveStackState(
            session_id="s1", frame_count=8,
            adaptive_target_bkg=0.30, adaptive_shadows_clip=-3.0,
        )
        critic = _StubCritic(
            verdict=CriticVerdict(satisfied=True, confidence=0.8, reasoning="fine")
        )

        decision = await evaluate(
            critic=critic,  # type: ignore[arg-type]
            preview_jpeg_path=_preview_path(tmp_path),
            stats=_stats(),
            state=state,
        )

        assert decision.target_bkg == 0.30
        assert decision.shadows_clip == -3.0

    @pytest.mark.asyncio
    async def test_unreachable_critic_degrades_to_satisfied(self, tmp_path: Path) -> None:
        state = LiveStackState(session_id="s1", frame_count=5)
        critic = _StubCritic(error=RuntimeError("network down"))

        decision = await evaluate(
            critic=critic,  # type: ignore[arg-type]
            preview_jpeg_path=_preview_path(tmp_path),
            stats=_stats(),
            state=state,
        )

        assert decision.satisfied is True
        assert decision.target_bkg == 0.25
        assert decision.shadows_clip == -2.8


class TestApplyDecision:
    def test_updates_state_fields(self) -> None:
        state = LiveStackState(session_id="s1", frame_count=5)
        decision = LiveAdaptiveDecision(
            target_bkg=0.30, shadows_clip=-3.0, satisfied=False,
            reasoning="not yet", confidence=0.5,
        )

        apply_decision(state, decision)

        assert state.adaptive_target_bkg == 0.30
        assert state.adaptive_shadows_clip == -3.0
        assert state.adaptive_attempts == 1
        assert state.adaptive_last_evaluated_frame_count == 5
        assert state.adaptive_converged is False
        assert len(state.adaptive_history) == 1

    def test_satisfied_decision_freezes_state(self) -> None:
        state = LiveStackState(session_id="s1", frame_count=8)
        decision = LiveAdaptiveDecision(
            target_bkg=0.28, shadows_clip=-2.9, satisfied=True,
            reasoning="good", confidence=0.9,
        )

        apply_decision(state, decision)

        assert state.adaptive_converged is True

    def test_multiple_applications_accumulate_history(self) -> None:
        state = LiveStackState(session_id="s1", frame_count=5)
        for i in range(3):
            apply_decision(
                state,
                LiveAdaptiveDecision(
                    target_bkg=0.25 + i * 0.01, shadows_clip=-2.8,
                    satisfied=False, reasoning=f"iter {i}", confidence=0.5,
                ),
            )
        assert state.adaptive_attempts == 3
        assert len(state.adaptive_history) == 3
