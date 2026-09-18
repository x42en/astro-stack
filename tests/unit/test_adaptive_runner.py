"""Unit tests for the adaptive critic loop (:mod:`app.pipeline.adaptive.runner`
and :mod:`app.pipeline.adaptive.graph`).

Uses a stub critic (no real HTTP) and a stub ``run_step`` callback so the
full LangGraph loop is exercised deterministically, without any network or
FITS/Siril dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.pipeline.adaptive.critic import CriticVerdict
from app.pipeline.adaptive.runner import run_adaptive_loop
from app.pipeline.adaptive.types import AdaptiveIterationRecord

_ALLOWED = ("stretch_method", "stretch_strength", "color_calibration_enabled")


class _StubCritic:
    """Returns pre-scripted verdicts in order, one per ``critique()`` call."""

    def __init__(self, verdicts: list[CriticVerdict]) -> None:
        self._verdicts = list(verdicts)
        self.calls: list[dict[str, Any]] = []

    async def critique(self, **kwargs: Any) -> CriticVerdict:
        self.calls.append(kwargs)
        return self._verdicts.pop(0)


class _RaisingCritic:
    async def critique(self, **kwargs: Any) -> CriticVerdict:
        raise RuntimeError("simulated network failure")


def _make_run_step(calls: list[dict[str, Any]]):
    async def _run_step(new_config: dict[str, Any]) -> tuple[dict[str, Any], str]:
        calls.append(new_config)
        return {"mean": 0.5}, "/tmp/preview.jpg"

    return _run_step


def _base_config() -> dict[str, Any]:
    return {
        "stretch_method": "asinh",
        "stretch_strength": 150.0,
        "color_calibration_enabled": True,
    }


class TestRunAdaptiveLoopAutonomousDefault:
    @pytest.mark.asyncio
    async def test_immediately_satisfied_stops_after_zero_reruns(self) -> None:
        critic = _StubCritic(
            [CriticVerdict(satisfied=True, confidence=0.9, reasoning="Looks great.", patch={})]
        )
        run_step_calls: list[dict[str, Any]] = []

        result = await run_adaptive_loop(
            step_name="stretch_color",
            allowed_fields=_ALLOWED,
            config_dict=_base_config(),
            initial_stats={"mean": 0.4},
            initial_preview_path=Path("/tmp/preview.jpg"),
            max_iterations=3,
            require_human_approval=False,
            run_step=_make_run_step(run_step_calls),
            critic=critic,
        )

        assert result.converged is True
        assert len(result.iterations) == 1
        assert run_step_calls == []  # step never re-run
        assert result.final_config_patch == {
            "stretch_method": "asinh",
            "stretch_strength": 150.0,
            "color_calibration_enabled": True,
        }

    @pytest.mark.asyncio
    async def test_applies_patch_and_converges_second_iteration(self) -> None:
        critic = _StubCritic(
            [
                CriticVerdict(
                    satisfied=False,
                    confidence=0.3,
                    reasoning="Too dim.",
                    patch={"stretch_strength": 180.0},
                ),
                CriticVerdict(satisfied=True, confidence=0.9, reasoning="Better."),
            ]
        )
        run_step_calls: list[dict[str, Any]] = []

        result = await run_adaptive_loop(
            step_name="stretch_color",
            allowed_fields=_ALLOWED,
            config_dict=_base_config(),
            initial_stats={},
            initial_preview_path=Path("/tmp/preview.jpg"),
            max_iterations=3,
            require_human_approval=False,
            run_step=_make_run_step(run_step_calls),
            critic=critic,
        )

        assert result.converged is True
        assert len(result.iterations) == 2
        assert result.iterations[0].satisfied is False
        assert result.iterations[0].patch_applied == {"stretch_strength": 180.0}
        assert result.iterations[1].satisfied is True
        assert len(run_step_calls) == 1
        assert run_step_calls[0]["stretch_strength"] == 180.0
        assert result.final_config_patch["stretch_strength"] == 180.0

    @pytest.mark.asyncio
    async def test_no_human_reviewer_called_when_approval_not_required(self) -> None:
        reviewer_calls: list[AdaptiveIterationRecord] = []

        async def reviewer(record: AdaptiveIterationRecord) -> bool:
            reviewer_calls.append(record)
            return True

        critic = _StubCritic(
            [
                CriticVerdict(
                    satisfied=False, confidence=0.3, reasoning="x",
                    patch={"stretch_strength": 180.0},
                ),
                CriticVerdict(satisfied=True, confidence=0.9, reasoning="ok"),
            ]
        )

        await run_adaptive_loop(
            step_name="stretch_color",
            allowed_fields=_ALLOWED,
            config_dict=_base_config(),
            initial_stats={},
            initial_preview_path=Path("/tmp/preview.jpg"),
            max_iterations=3,
            require_human_approval=False,
            run_step=_make_run_step([]),
            critic=critic,
            human_reviewer=reviewer,
        )

        # Off by default: the reviewer hook must NOT be invoked unless the
        # profile explicitly requires human approval.
        assert reviewer_calls == []

    @pytest.mark.asyncio
    async def test_iteration_budget_exhausted_does_not_converge(self) -> None:
        critic = _StubCritic(
            [
                CriticVerdict(
                    satisfied=False, confidence=0.2, reasoning="still off",
                    patch={"stretch_strength": 160.0},
                ),
                CriticVerdict(
                    satisfied=False, confidence=0.2, reasoning="still off",
                    patch={"stretch_strength": 170.0},
                ),
            ]
        )
        run_step_calls: list[dict[str, Any]] = []

        result = await run_adaptive_loop(
            step_name="stretch_color",
            allowed_fields=_ALLOWED,
            config_dict=_base_config(),
            initial_stats={},
            initial_preview_path=Path("/tmp/preview.jpg"),
            max_iterations=2,
            require_human_approval=False,
            run_step=_make_run_step(run_step_calls),
            critic=critic,
        )

        assert result.converged is False
        # First iteration applies + re-runs; second iteration hits the budget
        # cap and finalizes without re-running the step again.
        assert len(run_step_calls) == 1
        assert len(result.iterations) == 2

    @pytest.mark.asyncio
    async def test_patch_fully_dropped_by_sanitizer_finalizes_immediately(self) -> None:
        critic = _StubCritic(
            [
                CriticVerdict(
                    satisfied=False,
                    confidence=0.2,
                    reasoning="proposes disallowed field",
                    patch={"denoise_strength": 0.9},
                ),
            ]
        )
        run_step_calls: list[dict[str, Any]] = []

        result = await run_adaptive_loop(
            step_name="stretch_color",
            allowed_fields=_ALLOWED,
            config_dict=_base_config(),
            initial_stats={},
            initial_preview_path=Path("/tmp/preview.jpg"),
            max_iterations=5,
            require_human_approval=False,
            run_step=_make_run_step(run_step_calls),
            critic=critic,
        )

        assert result.converged is False
        assert run_step_calls == []
        assert len(result.iterations) == 1
        assert result.iterations[0].patch_applied == {}

    @pytest.mark.asyncio
    async def test_unreachable_critic_degrades_to_satisfied(self) -> None:
        result = await run_adaptive_loop(
            step_name="stretch_color",
            allowed_fields=_ALLOWED,
            config_dict=_base_config(),
            initial_stats={},
            initial_preview_path=Path("/tmp/preview.jpg"),
            max_iterations=3,
            require_human_approval=False,
            run_step=_make_run_step([]),
            critic=_RaisingCritic(),
        )

        assert result.converged is True
        assert len(result.iterations) == 1
        assert "unavailable" in result.iterations[0].reasoning.lower()

    @pytest.mark.asyncio
    async def test_on_iteration_callback_invoked_and_failures_swallowed(self) -> None:
        seen: list[AdaptiveIterationRecord] = []

        async def on_iteration(record: AdaptiveIterationRecord) -> None:
            seen.append(record)
            raise RuntimeError("UI publish failed")

        critic = _StubCritic(
            [CriticVerdict(satisfied=True, confidence=1.0, reasoning="fine")]
        )

        result = await run_adaptive_loop(
            step_name="stretch_color",
            allowed_fields=_ALLOWED,
            config_dict=_base_config(),
            initial_stats={},
            initial_preview_path=Path("/tmp/preview.jpg"),
            max_iterations=3,
            require_human_approval=False,
            run_step=_make_run_step([]),
            critic=critic,
            on_iteration=on_iteration,
        )

        assert result.converged is True
        assert len(seen) == 1  # callback failure did not break the loop

    @pytest.mark.asyncio
    async def test_rejects_max_iterations_below_one(self) -> None:
        with pytest.raises(ValueError):
            await run_adaptive_loop(
                step_name="stretch_color",
                allowed_fields=_ALLOWED,
                config_dict=_base_config(),
                initial_stats={},
                initial_preview_path=Path("/tmp/preview.jpg"),
                max_iterations=0,
                require_human_approval=False,
                run_step=_make_run_step([]),
                critic=_StubCritic([]),
            )


class TestRunAdaptiveLoopHumanApproval:
    @pytest.mark.asyncio
    async def test_reviewer_rejection_stops_loop_without_applying_patch(self) -> None:
        async def reviewer(record: AdaptiveIterationRecord) -> bool:
            return False

        critic = _StubCritic(
            [
                CriticVerdict(
                    satisfied=False, confidence=0.3, reasoning="x",
                    patch={"stretch_strength": 180.0},
                ),
            ]
        )
        run_step_calls: list[dict[str, Any]] = []

        result = await run_adaptive_loop(
            step_name="stretch_color",
            allowed_fields=_ALLOWED,
            config_dict=_base_config(),
            initial_stats={},
            initial_preview_path=Path("/tmp/preview.jpg"),
            max_iterations=3,
            require_human_approval=True,
            run_step=_make_run_step(run_step_calls),
            critic=critic,
            human_reviewer=reviewer,
        )

        assert result.converged is False
        assert run_step_calls == []
        assert result.iterations[0].human_approved is False
        assert result.final_config_patch["stretch_strength"] == 150.0  # unchanged

    @pytest.mark.asyncio
    async def test_reviewer_approval_applies_patch(self) -> None:
        async def reviewer(record: AdaptiveIterationRecord) -> bool:
            return True

        critic = _StubCritic(
            [
                CriticVerdict(
                    satisfied=False, confidence=0.3, reasoning="x",
                    patch={"stretch_strength": 180.0},
                ),
                CriticVerdict(satisfied=True, confidence=0.9, reasoning="ok"),
            ]
        )
        run_step_calls: list[dict[str, Any]] = []

        result = await run_adaptive_loop(
            step_name="stretch_color",
            allowed_fields=_ALLOWED,
            config_dict=_base_config(),
            initial_stats={},
            initial_preview_path=Path("/tmp/preview.jpg"),
            max_iterations=3,
            require_human_approval=True,
            run_step=_make_run_step(run_step_calls),
            critic=critic,
            human_reviewer=reviewer,
        )

        assert result.converged is True
        assert result.iterations[0].human_approved is True
        assert len(run_step_calls) == 1

    @pytest.mark.asyncio
    async def test_no_reviewer_configured_auto_approves_with_warning(self) -> None:
        critic = _StubCritic(
            [
                CriticVerdict(
                    satisfied=False, confidence=0.3, reasoning="x",
                    patch={"stretch_strength": 180.0},
                ),
                CriticVerdict(satisfied=True, confidence=0.9, reasoning="ok"),
            ]
        )
        run_step_calls: list[dict[str, Any]] = []

        result = await run_adaptive_loop(
            step_name="stretch_color",
            allowed_fields=_ALLOWED,
            config_dict=_base_config(),
            initial_stats={},
            initial_preview_path=Path("/tmp/preview.jpg"),
            max_iterations=3,
            require_human_approval=True,
            run_step=_make_run_step(run_step_calls),
            critic=critic,
            human_reviewer=None,
        )

        assert result.converged is True
        assert result.iterations[0].human_approved is True
        assert len(run_step_calls) == 1
