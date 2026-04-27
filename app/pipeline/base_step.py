"""Abstract base class for all pipeline processing steps.

Every concrete step must inherit from :class:`PipelineStep` and implement
:meth:`execute`. The orchestrator calls ``execute()`` and catches
:class:`~app.core.errors.PipelineStepException` for retry logic.

Example:
    >>> class MyStep(PipelineStep):
    ...     name = "my_step"
    ...     display_name = "My Custom Step"
    ...
    ...     async def execute(self, context: PipelineContext) -> StepResult:
    ...         # do work
    ...         return StepResult(success=True, metadata={"key": "value"})
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class PipelineContext:
    """Shared mutable state passed between pipeline steps.

    The orchestrator creates one context per job run and passes it through
    each step in sequence. Steps append results to ``metadata`` so later
    steps can consume them (e.g. plate-solve coordinates used in colour cal).

    Attributes:
        job_id: UUID of the running pipeline job.
        session_id: UUID of the associated session.
        work_dir: Intermediate working directory for this session.
        output_dir: Final output directory for this session.
        gpu_device: CUDA device string (e.g. ``"cuda:0"``).
        metadata: Accumulated step metadata from previous steps.
        cancelled: Set to ``True`` to request job cancellation.
    """

    job_id: uuid.UUID
    session_id: uuid.UUID
    work_dir: Path
    output_dir: Path
    gpu_device: str = "cuda:0"
    metadata: dict[str, Any] = field(default_factory=dict)
    cancelled: bool = False

    # Paths set by individual steps and consumed by later steps
    stacked_fits_path: Optional[Path] = None
    background_removed_path: Optional[Path] = None
    stretched_fits_path: Optional[Path] = None
    denoised_path: Optional[Path] = None
    sharpened_path: Optional[Path] = None
    superres_path: Optional[Path] = None
    nebula_only_path: Optional[Path] = None
    final_fits_path: Optional[Path] = None


@dataclass
class StepResult:
    """Result returned by a pipeline step after execution.

    Attributes:
        success: ``True`` if the step completed without errors.
        metadata: Step-specific output data to be stored in :class:`PipelineContext`.
        skipped: ``True`` if the step was disabled by the active profile.
        message: Human-readable summary of the step outcome.
    """

    success: bool
    metadata: dict[str, Any] = field(default_factory=dict)
    skipped: bool = False
    message: str = ""


class PipelineStep(ABC):
    """Abstract base class for a single pipeline processing step.

    Subclasses must define :attr:`name` (machine identifier) and
    :attr:`display_name` (human-readable), and implement :meth:`execute`.

    Attributes:
        name: Short snake_case identifier used in DB records and WS events.
        display_name: Human-readable step name shown in the UI.
    """

    name: str = "base_step"
    display_name: str = "Base Step"

    @abstractmethod
    async def execute(
        self,
        context: PipelineContext,
        config: dict[str, Any],
    ) -> StepResult:
        """Execute the processing step.

        Args:
            context: Shared pipeline context containing paths and accumulated metadata.
            config: Step-specific parameters extracted from the active profile config.

        Returns:
            A :class:`StepResult` describing the outcome.

        Raises:
            PipelineStepException: On unrecoverable or retryable step failure.
        """
        ...

    def __repr__(self) -> str:
        """Return a developer-friendly representation.

        Returns:
            String with class name and step name.
        """
        return f"<{self.__class__.__name__} name={self.name!r}>"
