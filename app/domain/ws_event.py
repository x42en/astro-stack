"""WebSocket event schemas for real-time pipeline progress updates.

All events emitted by the pipeline workers through Redis pub/sub and
forwarded to connected WebSocket clients conform to one of these schemas.
The ``type`` field acts as a discriminator for client-side deserialisation.

Event flow::

    PipelineStep  -->  EventBus.publish()  -->  Redis PubSub
                                                     |
                                      WebSocketManager.broadcast()
                                                     |
                                        Connected WS clients

Example:
    >>> event = ProgressEvent(
    ...     job_id=job_id,
    ...     session_id=session_id,
    ...     step="stacking",
    ...     step_index=3,
    ...     total_steps=9,
    ...     percent=67.5,
    ...     message="Stacking 42/63 frames...",
    ... )
    >>> payload = event.model_dump_json()
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    """Return the current UTC datetime.

    Returns:
        Current time in UTC with timezone info.
    """
    return datetime.now(tz=timezone.utc)


class EventType(str, Enum):
    """Discriminator values for WebSocket event payloads.

    Attributes:
        PROGRESS: Incremental progress update within a step.
        LOG: A log message emitted by a processing tool.
        STEP_STATUS: A step has started, succeeded, failed, or was skipped.
        ERROR: A retryable or definitive error occurred.
        COMPLETED: The pipeline job has finished successfully.
        CANCELLED: The pipeline job was cancelled.
        SESSION_DETECTED: Watchdog has detected a new session.
        SESSION_READY: Session frame inventory is complete.
    """

    PROGRESS = "progress"
    LOG = "log"
    STEP_STATUS = "step_status"
    ERROR = "error"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    SESSION_DETECTED = "session_detected"
    SESSION_READY = "session_ready"


class LogLevel(str, Enum):
    """Severity levels for log events forwarded from processing tools.

    Attributes:
        DEBUG: Verbose diagnostic output.
        INFO: Informational messages.
        WARNING: Non-fatal anomalies.
        ERROR: Recoverable errors.
    """

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class LogSource(str, Enum):
    """Origin of a log message within the pipeline.

    Attributes:
        SIRIL: Siril headless output.
        ASTAP: ASTAP plate solver output.
        GRAXPERT: GraXpert output.
        COSMIC: Cosmic Clarity scripts output.
        SYSTEM: Internal application logs.
    """

    SIRIL = "siril"
    ASTAP = "astap"
    GRAXPERT = "graxpert"
    COSMIC = "cosmic"
    SYSTEM = "system"


class StepStatusValue(str, Enum):
    """Possible outcomes reported in a :class:`StepStatusEvent`.

    Attributes:
        STARTING: Step execution has begun.
        SUCCESS: Step completed without errors.
        ERROR: Step failed (may trigger retry).
        SKIPPED: Step was disabled by the active profile.
    """

    STARTING = "starting"
    SUCCESS = "success"
    ERROR = "error"
    SKIPPED = "skipped"


# ── Base ──────────────────────────────────────────────────────────────────────


class BaseEvent(BaseModel):
    """Common fields shared by all WebSocket events.

    Attributes:
        type: Discriminator string for client-side routing.
        job_id: Associated pipeline job UUID (may be None for session events).
        session_id: Associated session UUID.
        timestamp: UTC timestamp when the event was generated.
    """

    type: EventType
    job_id: Optional[uuid.UUID] = None
    session_id: Optional[uuid.UUID] = None
    timestamp: datetime = Field(default_factory=_utcnow)


# ── Concrete event types ──────────────────────────────────────────────────────


class ProgressEvent(BaseEvent):
    """Incremental progress update within a pipeline step.

    Attributes:
        type: Always ``EventType.PROGRESS``.
        step: Machine-readable step identifier.
        step_index: Zero-based step ordinal.
        total_steps: Total number of steps in the pipeline.
        percent: Completion percentage of the current step (0.0–100.0).
        message: Human-readable progress description.
    """

    type: Literal[EventType.PROGRESS] = EventType.PROGRESS
    step: str
    step_index: int
    total_steps: int
    percent: float = Field(ge=0.0, le=100.0)
    message: str


class LogEvent(BaseEvent):
    """A log line emitted by a processing tool and forwarded to the client.

    Attributes:
        type: Always ``EventType.LOG``.
        level: Severity level.
        source: Tool that produced the message.
        message: Raw log message text.
    """

    type: Literal[EventType.LOG] = EventType.LOG
    level: LogLevel = LogLevel.INFO
    source: LogSource = LogSource.SYSTEM
    message: str


class StepStatusEvent(BaseEvent):
    """Notification that a pipeline step has changed state.

    Attributes:
        type: Always ``EventType.STEP_STATUS``.
        step: Machine-readable step identifier.
        step_index: Zero-based step ordinal.
        status: New step status value.
        result: Optional step-specific output data (e.g. plate-solve coords).
    """

    type: Literal[EventType.STEP_STATUS] = EventType.STEP_STATUS
    step: str
    step_index: int
    status: StepStatusValue
    result: Optional[dict[str, Any]] = None


class ErrorEvent(BaseEvent):
    """An error has occurred, potentially triggering a retry.

    Attributes:
        type: Always ``EventType.ERROR``.
        error_code: Machine-readable :class:`~app.core.errors.ErrorCode` value.
        message: English error description.
        step: Name of the step that failed.
        retryable: Whether the error class permits automatic retry.
        attempt: Current attempt number (1-based).
        max_attempts: Maximum allowed attempts.
        details: Optional extra context data.
    """

    type: Literal[EventType.ERROR] = EventType.ERROR
    error_code: str
    message: str
    step: str
    retryable: bool = False
    attempt: int = 1
    max_attempts: int = 3
    details: Optional[dict[str, Any]] = None


class CompletedEvent(BaseEvent):
    """The pipeline has finished successfully.

    Attributes:
        type: Always ``EventType.COMPLETED``.
        duration_seconds: Total wall-clock time for the pipeline run.
        outputs: Dict mapping output type to file path or URL.
    """

    type: Literal[EventType.COMPLETED] = EventType.COMPLETED
    duration_seconds: float
    outputs: dict[str, str] = Field(default_factory=dict)


class CancelledEvent(BaseEvent):
    """The pipeline job was cancelled (manually or due to shutdown).

    Attributes:
        type: Always ``EventType.CANCELLED``.
        reason: Short human-readable reason for cancellation.
    """

    type: Literal[EventType.CANCELLED] = EventType.CANCELLED
    reason: str = "User requested cancellation"


class SessionDetectedEvent(BaseEvent):
    """Watchdog has found a new potential session in the inbox.

    Attributes:
        type: Always ``EventType.SESSION_DETECTED``.
        inbox_path: Detected session directory path.
        name: Auto-derived session name.
    """

    type: Literal[EventType.SESSION_DETECTED] = EventType.SESSION_DETECTED
    inbox_path: str
    name: str


class SessionReadyEvent(BaseEvent):
    """Session frame inventory is complete and ready for processing.

    Attributes:
        type: Always ``EventType.SESSION_READY``.
        frame_count_lights: Number of light frames found.
        frame_count_darks: Number of dark frames found.
        frame_count_flats: Number of flat frames found.
        frame_count_bias: Number of bias frames found.
        input_format: Detected source file format.
    """

    type: Literal[EventType.SESSION_READY] = EventType.SESSION_READY
    frame_count_lights: int
    frame_count_darks: int
    frame_count_flats: int
    frame_count_bias: int
    input_format: str


# ── Union type for client-side deserialisation ────────────────────────────────

AnyEvent = Union[
    ProgressEvent,
    LogEvent,
    StepStatusEvent,
    ErrorEvent,
    CompletedEvent,
    CancelledEvent,
    SessionDetectedEvent,
    SessionReadyEvent,
]
