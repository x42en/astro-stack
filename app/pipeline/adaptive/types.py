"""Shared result/record types for the adaptive vision-critic loop (Phase 2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class AdaptiveIterationRecord:
    """One completed iteration of the adaptive critic loop.

    Attributes:
        iteration: 0-based iteration index.
        satisfied: Whether the critic accepted the result at this iteration.
        confidence: Critic's self-reported confidence (0.0-1.0).
        reasoning: Critic's short English explanation.
        patch_proposed: Raw patch proposed by the critic, before catalog
            validation/clamping.
        patch_applied: Patch actually applied after catalog validation and
            clamping (may drop or clamp values from ``patch_proposed``).
        human_approved: ``None`` when no human-approval gate was involved for
            this iteration; otherwise whether the reviewer approved it.
    """

    iteration: int
    satisfied: bool
    confidence: float
    reasoning: str
    patch_proposed: dict[str, Any] = field(default_factory=dict)
    patch_applied: dict[str, Any] = field(default_factory=dict)
    human_approved: Optional[bool] = None

    def to_dict(self) -> dict[str, Any]:
        """Render as a plain JSON-serialisable dict.

        Returns:
            Dict with every field of this record.
        """
        return {
            "iteration": self.iteration,
            "satisfied": self.satisfied,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "patch_proposed": self.patch_proposed,
            "patch_applied": self.patch_applied,
            "human_approved": self.human_approved,
        }


@dataclass
class AdaptiveLoopResult:
    """Final outcome of an adaptive critic loop run for one pipeline step.

    Attributes:
        converged: True if the critic reported ``satisfied=True`` before the
            iteration budget was exhausted or a reviewer rejected a patch.
        iterations: Ordered history of every iteration performed.
        final_config_patch: Cumulative value of every field the loop was
            allowed to touch, after all applied iterations (suitable for
            merging back into the caller's full profile config dict).
    """

    converged: bool
    iterations: list[AdaptiveIterationRecord]
    final_config_patch: dict[str, Any]

    def to_metadata(self) -> dict[str, Any]:
        """Render as a JSON-serialisable dict for ``JobStep.output_metadata``.

        Returns:
            Dict with ``adaptive_converged``, ``adaptive_iterations``, and
            ``adaptive_final_patch`` keys.
        """
        return {
            "adaptive_converged": self.converged,
            "adaptive_iterations": [it.to_dict() for it in self.iterations],
            "adaptive_final_patch": self.final_config_patch,
        }
