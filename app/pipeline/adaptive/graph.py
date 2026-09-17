"""LangGraph state machine for the adaptive vision-critic loop (Phase 2).

Runs entirely in-process with no checkpointer/persistence layer: a single
pipeline step is repeatedly re-executed with a critic-adjusted config until
the vision critic reports satisfaction, the iteration budget runs out, or an
optional human reviewer rejects a proposed patch. See
:func:`app.pipeline.adaptive.runner.run_adaptive_loop` for the entry point
that wires this graph to a concrete
:class:`~app.pipeline.base_step.PipelineStep`.

Fully autonomous by default: the human-approval branch only ever gates a
patch when the caller explicitly sets ``require_human_approval=True`` *and*
supplies a ``human_reviewer`` callback; otherwise every catalog-validated
patch is applied automatically, matching AstroStack's novice-friendly,
hands-off product goal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from app.core.logging import get_logger
from app.pipeline.adaptive.critic import CriticVerdict, VisionCritic
from app.pipeline.adaptive.patch import sanitize_patch
from app.pipeline.adaptive.tool_catalog import render_capabilities_for_prompt
from app.pipeline.adaptive.types import AdaptiveIterationRecord

logger = get_logger(__name__)

# Re-executes the target step with a patched config; returns fresh
# ``(stats, preview_path)`` for the newly produced output.
RunStepFn = Callable[[dict[str, Any]], Awaitable[tuple[dict[str, Any], str]]]

# Invoked after every completed iteration (e.g. to publish a WebSocket event).
OnIterationFn = Callable[[AdaptiveIterationRecord], Awaitable[None]]

# Invoked to approve/reject a proposed patch when human approval is required.
HumanReviewerFn = Callable[[AdaptiveIterationRecord], Awaitable[bool]]


class AdaptiveLoopState(TypedDict, total=False):
    """Mutable state threaded through the adaptive loop graph.

    Attributes:
        step_name: Machine name of the pipeline step being refined.
        iteration: 0-based index of the iteration about to run.
        max_iterations: Hard cap on iterations for this run.
        require_human_approval: Whether patches need external approval.
        allowed_fields: Field names the critic may adjust for this step.
        config_dict: Current full profile config; only ``allowed_fields`` are
            ever changed across iterations.
        stats: Latest numeric image statistics for the step's output.
        preview_path: Filesystem path (as a string) of the latest JPEG preview.
        history: Completed iteration records, as plain dicts.
        pending_record: The just-critiqued iteration record, awaiting a
            routing decision (human gate / apply / finalize).
        approved: Result of the human-gate decision for ``pending_record``
            (defaults to True — see :func:`build_adaptive_graph`). Only
            meaningful between the ``human_gate`` and routing steps; every
            key must be declared here because LangGraph only persists state
            fields that are part of this schema.
        done: Set once the loop has stopped.
        converged: True only when the critic itself reported satisfaction.
    """

    step_name: str
    iteration: int
    max_iterations: int
    require_human_approval: bool
    allowed_fields: tuple[str, ...]
    config_dict: dict[str, Any]
    stats: dict[str, Any]
    preview_path: str
    history: list[dict[str, Any]]
    pending_record: Optional[dict[str, Any]]
    approved: bool
    done: bool
    converged: bool


def build_adaptive_graph(
    *,
    critic: VisionCritic,
    run_step: RunStepFn,
    on_iteration: Optional[OnIterationFn] = None,
    human_reviewer: Optional[HumanReviewerFn] = None,
) -> Any:
    """Build and compile the adaptive critic loop graph.

    Args:
        critic: Vision-language critic client.
        run_step: Async callback that re-executes the target pipeline step
            with a patched config dict and returns fresh
            ``(stats, preview_path)``.
        on_iteration: Optional callback invoked after every completed
            iteration. Never allowed to break the loop — failures are
            logged and swallowed so a UI-notification bug cannot break the
            automated critic loop itself.
        human_reviewer: Optional callback asked to approve a proposed patch
            before it is applied. Only consulted when
            ``state["require_human_approval"]`` is True. When ``None`` (the
            default, matching AstroStack's fully-automated goal), a patch is
            auto-approved even if ``require_human_approval`` is True (with a
            logged warning) rather than stalling the job.

    Returns:
        A compiled LangGraph graph exposing an ``ainvoke(initial_state)`` coroutine.
    """

    async def critique_node(state: AdaptiveLoopState) -> dict[str, Any]:
        capability_context = render_capabilities_for_prompt(state["allowed_fields"])
        current_values = {f: state["config_dict"].get(f) for f in state["allowed_fields"]}
        try:
            verdict = await critic.critique(
                step_name=state["step_name"],
                iteration=state["iteration"],
                max_iterations=state["max_iterations"],
                capability_context=capability_context,
                current_values=current_values,
                stats=state["stats"],
                preview_jpeg_path=Path(state["preview_path"]),
                history=state["history"],
            )
        except Exception:  # noqa: BLE001
            # An unreachable/misbehaving critic must never break the fully
            # automated pipeline — accept the current result and stop.
            logger.warning(
                "adaptive_critic_unavailable", step=state["step_name"], exc_info=True
            )
            verdict = CriticVerdict(
                satisfied=True,
                confidence=0.0,
                reasoning="Vision critic unavailable; accepting current result.",
            )

        sanitized = sanitize_patch(verdict.patch, allowed_fields=state["allowed_fields"])
        record = AdaptiveIterationRecord(
            iteration=state["iteration"],
            satisfied=verdict.satisfied,
            confidence=verdict.confidence,
            reasoning=verdict.reasoning,
            patch_proposed=verdict.patch,
            patch_applied=sanitized,
        )
        return {"pending_record": record.to_dict()}

    def route_after_critique(state: AdaptiveLoopState) -> str:
        record = state["pending_record"]
        assert record is not None
        if record["satisfied"] or not record["patch_applied"]:
            return "finalize"
        if state["iteration"] + 1 >= state["max_iterations"]:
            return "finalize"
        return "human_gate"

    async def human_gate_node(state: AdaptiveLoopState) -> dict[str, Any]:
        record = dict(state["pending_record"] or {})
        approved = True
        if state["require_human_approval"]:
            if human_reviewer is not None:
                approved = await human_reviewer(AdaptiveIterationRecord(**record))
            else:
                logger.warning(
                    "adaptive_human_approval_requested_no_reviewer",
                    step=state["step_name"],
                    iteration=state["iteration"],
                )
            record["human_approved"] = approved
        return {"pending_record": record, "approved": approved}

    def route_after_human_gate(state: AdaptiveLoopState) -> str:
        return "apply_patch" if state.get("approved", True) else "finalize"

    async def apply_patch_node(state: AdaptiveLoopState) -> dict[str, Any]:
        record = state["pending_record"]
        assert record is not None
        new_config = dict(state["config_dict"])
        new_config.update(record["patch_applied"])
        stats, preview_path = await run_step(new_config)
        history = [*state["history"], record]
        await _safe_on_iteration(on_iteration, record)
        return {
            "config_dict": new_config,
            "stats": stats,
            "preview_path": preview_path,
            "history": history,
            "iteration": state["iteration"] + 1,
            "pending_record": None,
        }

    async def finalize_node(state: AdaptiveLoopState) -> dict[str, Any]:
        record = state.get("pending_record")
        history = state["history"]
        converged = False
        if record is not None:
            history = [*history, record]
            converged = bool(record["satisfied"])
            await _safe_on_iteration(on_iteration, record)
        return {"done": True, "converged": converged, "history": history}

    graph: StateGraph = StateGraph(AdaptiveLoopState)
    graph.add_node("critique", critique_node)
    graph.add_node("human_gate", human_gate_node)
    graph.add_node("apply_patch", apply_patch_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "critique")
    graph.add_conditional_edges(
        "critique",
        route_after_critique,
        {"finalize": "finalize", "human_gate": "human_gate"},
    )
    graph.add_conditional_edges(
        "human_gate",
        route_after_human_gate,
        {"apply_patch": "apply_patch", "finalize": "finalize"},
    )
    graph.add_edge("apply_patch", "critique")
    graph.add_edge("finalize", END)

    return graph.compile()


async def _safe_on_iteration(
    callback: Optional[OnIterationFn], record_dict: dict[str, Any]
) -> None:
    """Invoke ``callback`` with a reconstructed record, swallowing any error.

    Args:
        callback: Optional iteration callback.
        record_dict: Plain-dict form of an :class:`AdaptiveIterationRecord`.
    """
    if callback is None:
        return
    try:
        await callback(AdaptiveIterationRecord(**record_dict))
    except Exception:  # noqa: BLE001
        logger.warning("adaptive_on_iteration_callback_failed", exc_info=True)
