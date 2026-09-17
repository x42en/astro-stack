"""Entry point wiring the adaptive critic graph to a concrete pipeline step.

This is the module the pipeline orchestrator calls; it hides the LangGraph
plumbing behind a single async function.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from app.core.logging import get_logger
from app.pipeline.adaptive.critic import VisionCritic
from app.pipeline.adaptive.graph import (
    HumanReviewerFn,
    OnIterationFn,
    RunStepFn,
    build_adaptive_graph,
)
from app.pipeline.adaptive.types import AdaptiveIterationRecord, AdaptiveLoopResult

logger = get_logger(__name__)


async def run_adaptive_loop(
    *,
    step_name: str,
    allowed_fields: tuple[str, ...],
    config_dict: dict,
    initial_stats: dict,
    initial_preview_path: Path,
    max_iterations: int,
    require_human_approval: bool,
    run_step: RunStepFn,
    on_iteration: Optional[OnIterationFn] = None,
    human_reviewer: Optional[HumanReviewerFn] = None,
    critic: Optional[VisionCritic] = None,
) -> AdaptiveLoopResult:
    """Run the adaptive vision-critic loop for a single pipeline step.

    Fully autonomous by default: with ``require_human_approval=False`` (the
    profile default), every critic-proposed patch that survives catalog
    validation is applied automatically — no pause, no external input
    required, matching AstroStack's novice-friendly, hands-off product goal.

    A critic that is unreachable or returns a malformed response never fails
    this call or the underlying job: it is treated as an implicit
    "satisfied", and the loop stops after 0 iterations (see
    :mod:`app.pipeline.adaptive.graph`).

    Args:
        step_name: Machine name of the step being refined (e.g. ``"stretch_color"``).
        allowed_fields: Profile field names the critic may adjust.
        config_dict: Full profile config dict, as already used for the
            step's first (pre-loop) execution. Not mutated in place; the
            final merged values are returned in the result instead.
        initial_stats: Numeric stats for the step's first-execution output.
        initial_preview_path: JPEG preview for the step's first-execution output.
        max_iterations: Hard cap on critic iterations (must be >= 1).
        require_human_approval: When True, proposed patches are only applied
            once ``human_reviewer`` approves them (or, if no reviewer is
            configured, a warning is logged and the patch auto-approves).
        run_step: Async callback that re-executes the step with a patched
            config dict and returns ``(new_stats, new_preview_path)``.
        on_iteration: Optional callback fired after every completed
            iteration (e.g. to publish a WebSocket event).
        human_reviewer: Optional async callback approving/rejecting a
            proposed patch.
        critic: Vision critic client; a default instance built from
            application settings is used when omitted.

    Returns:
        :class:`~app.pipeline.adaptive.types.AdaptiveLoopResult` summarising
        every iteration and whether the loop converged.

    Raises:
        ValueError: If ``max_iterations`` is less than 1.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")

    own_critic = critic is None
    active_critic = critic or VisionCritic()
    try:
        graph = build_adaptive_graph(
            critic=active_critic,
            run_step=run_step,
            on_iteration=on_iteration,
            human_reviewer=human_reviewer,
        )
        initial_state = {
            "step_name": step_name,
            "iteration": 0,
            "max_iterations": max_iterations,
            "require_human_approval": require_human_approval,
            "allowed_fields": allowed_fields,
            "config_dict": dict(config_dict),
            "stats": initial_stats,
            "preview_path": str(initial_preview_path),
            "history": [],
            "pending_record": None,
            "approved": True,
            "done": False,
            "converged": False,
        }
        final_state = await graph.ainvoke(initial_state)
    finally:
        if own_critic:
            await active_critic.aclose()

    history = [AdaptiveIterationRecord(**record) for record in final_state["history"]]
    final_patch = {name: final_state["config_dict"].get(name) for name in allowed_fields}

    logger.info(
        "adaptive_loop_finished",
        step=step_name,
        converged=final_state["converged"],
        iterations=len(history),
    )

    return AdaptiveLoopResult(
        converged=bool(final_state["converged"]),
        iterations=history,
        final_config_patch=final_patch,
    )
