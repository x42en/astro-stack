"""Application error codes and exception hierarchy.

All errors expose a machine-readable ``error_code`` string that the frontend
can use for i18n message lookup. Human-readable messages are in English and
are intended as developer/support aids, not end-user facing text.

Example:
    >>> raise AstroStackException(
    ...     ErrorCode.SESS_NOT_FOUND,
    ...     "Session abc123 does not exist",
    ...     status_code=404,
    ... )
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    """Enumeration of all application error codes for i18n support.

    Code prefixes:
        - ``SESS_``: Session lifecycle errors.
        - ``JOB_``:  Pipeline job errors.
        - ``PIPE_``: Individual pipeline step errors.
        - ``PROF_``: Processing profile errors.
        - ``AUTH_``: Authentication / authorisation errors.
        - ``MODEL_``: AI model loading / validation errors.
        - ``SYS_``:  Internal system / infrastructure errors.
    """

    # ── Session ───────────────────────────────────────────────────────────────
    SESS_NOT_FOUND = "SESS_NOT_FOUND"
    SESS_ALREADY_PROCESSING = "SESS_ALREADY_PROCESSING"
    SESS_INVALID_STRUCTURE = "SESS_INVALID_STRUCTURE"
    SESS_NO_LIGHTS_FOUND = "SESS_NO_LIGHTS_FOUND"
    SESS_DETECTION_FAILED = "SESS_DETECTION_FAILED"

    # ── Job ───────────────────────────────────────────────────────────────────
    JOB_NOT_FOUND = "JOB_NOT_FOUND"
    JOB_CANCEL_FAILED = "JOB_CANCEL_FAILED"
    JOB_MAX_RETRIES_EXCEEDED = "JOB_MAX_RETRIES_EXCEEDED"
    JOB_ALREADY_COMPLETED = "JOB_ALREADY_COMPLETED"
    JOB_INVALID_STATE_TRANSITION = "JOB_INVALID_STATE_TRANSITION"

    # ── Pipeline steps ────────────────────────────────────────────────────────
    PIPE_RAW_CONVERSION_FAILED = "PIPE_RAW_CONVERSION_FAILED"
    PIPE_SIRIL_INIT_FAILED = "PIPE_SIRIL_INIT_FAILED"
    PIPE_SIRIL_COMMAND_ERROR = "PIPE_SIRIL_COMMAND_ERROR"
    PIPE_SIRIL_TIMEOUT = "PIPE_SIRIL_TIMEOUT"
    PIPE_REGISTRATION_FAILED = "PIPE_REGISTRATION_FAILED"
    PIPE_DISK_FULL = "PIPE_DISK_FULL"
    PIPE_PLATE_SOLVE_FAILED = "PIPE_PLATE_SOLVE_FAILED"
    PIPE_GRADIENT_REMOVAL_FAILED = "PIPE_GRADIENT_REMOVAL_FAILED"
    PIPE_STRETCH_FAILED = "PIPE_STRETCH_FAILED"
    PIPE_COSMIC_DENOISE_FAILED = "PIPE_COSMIC_DENOISE_FAILED"
    PIPE_COSMIC_SHARPEN_FAILED = "PIPE_COSMIC_SHARPEN_FAILED"
    PIPE_COSMIC_SUPERRES_FAILED = "PIPE_COSMIC_SUPERRES_FAILED"
    PIPE_GRAXPERT_DENOISE_FAILED = "PIPE_GRAXPERT_DENOISE_FAILED"
    PIPE_STAR_SEPARATION_FAILED = "PIPE_STAR_SEPARATION_FAILED"
    PIPE_EXPORT_FAILED = "PIPE_EXPORT_FAILED"
    PIPE_STEP_NOT_FOUND = "PIPE_STEP_NOT_FOUND"

    # ── Profile ───────────────────────────────────────────────────────────────
    PROF_NOT_FOUND = "PROF_NOT_FOUND"
    PROF_VALIDATION_ERROR = "PROF_VALIDATION_ERROR"
    PROF_NAME_CONFLICT = "PROF_NAME_CONFLICT"
    PROF_PRESET_NOT_EDITABLE = "PROF_PRESET_NOT_EDITABLE"

    # ── Auth ──────────────────────────────────────────────────────────────────
    AUTH_TOKEN_INVALID = "AUTH_TOKEN_INVALID"
    AUTH_TOKEN_EXPIRED = "AUTH_TOKEN_EXPIRED"
    AUTH_REQUIRED = "AUTH_REQUIRED"
    AUTH_INSUFFICIENT_PERMISSIONS = "AUTH_INSUFFICIENT_PERMISSIONS"

    # ── AI Models ─────────────────────────────────────────────────────────────
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    MODEL_CHECKSUM_MISMATCH = "MODEL_CHECKSUM_MISMATCH"
    MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"

    # ── System ────────────────────────────────────────────────────────────────
    SYS_INTERNAL_ERROR = "SYS_INTERNAL_ERROR"
    SYS_EXTERNAL_TOOL_MISSING = "SYS_EXTERNAL_TOOL_MISSING"
    SYS_GPU_UNAVAILABLE = "SYS_GPU_UNAVAILABLE"
    SYS_REDIS_UNAVAILABLE = "SYS_REDIS_UNAVAILABLE"
    SYS_DATABASE_ERROR = "SYS_DATABASE_ERROR"
    SYS_STORAGE_ERROR = "SYS_STORAGE_ERROR"


class AstroStackException(Exception):
    """Base exception for all application-level errors.

    Wraps an :class:`ErrorCode` and optional structured details so that the
    global error handler can serialise a consistent JSON response.

    Attributes:
        error_code: Machine-readable code for i18n lookup on the client.
        message: English developer message (not displayed directly to users).
        status_code: HTTP status code to return (default 500).
        details: Arbitrary serialisable dict for debugging context.
        retryable: Hint for the orchestrator retry logic.
    """

    def __init__(
        self,
        error_code: ErrorCode,
        message: str,
        status_code: int = 500,
        details: dict[str, Any] | None = None,
        retryable: bool = False,
    ) -> None:
        """Initialise the exception.

        Args:
            error_code: One of the :class:`ErrorCode` members.
            message: Short English description of the error.
            status_code: HTTP status code for REST responses.
            details: Optional dict with additional debugging context.
            retryable: Whether this error class is safe to retry.
        """
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        self.retryable = retryable

    def to_dict(self) -> dict[str, Any]:
        """Serialise the exception to a JSON-compatible dict.

        Returns:
            Dictionary suitable for a structured API error response.
        """
        return {
            "error_code": self.error_code.value,
            "message": self.message,
            "details": self.details,
        }


# ── Convenience subclasses ────────────────────────────────────────────────────


class NotFoundException(AstroStackException):
    """Raised when a requested resource does not exist (HTTP 404).

    Args:
        error_code: The specific not-found error code.
        message: Description of what was not found.
        details: Optional context dict.
    """

    def __init__(
        self,
        error_code: ErrorCode,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(error_code, message, status_code=404, details=details)


class ConflictException(AstroStackException):
    """Raised on state or name conflicts (HTTP 409).

    Args:
        error_code: The specific conflict error code.
        message: Description of the conflict.
        details: Optional context dict.
    """

    def __init__(
        self,
        error_code: ErrorCode,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(error_code, message, status_code=409, details=details)


class ValidationException(AstroStackException):
    """Raised when input data fails domain-level validation (HTTP 422).

    Args:
        error_code: The specific validation error code.
        message: Description of the validation failure.
        details: Optional context dict.
    """

    def __init__(
        self,
        error_code: ErrorCode,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(error_code, message, status_code=422, details=details)


class PipelineStepException(AstroStackException):
    """Raised when a pipeline processing step fails.

    Args:
        error_code: The specific step error code.
        message: Description of the step failure.
        step_name: Name of the failed step.
        retryable: Whether this failure can be retried.
        details: Optional context dict.
    """

    def __init__(
        self,
        error_code: ErrorCode,
        message: str,
        step_name: str,
        retryable: bool = True,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            error_code,
            message,
            status_code=500,
            details={"step_name": step_name, **(details or {})},
            retryable=retryable,
        )
        self.step_name = step_name


class AuthException(AstroStackException):
    """Raised for authentication and authorisation failures (HTTP 401/403).

    Args:
        error_code: The specific auth error code.
        message: Description of the auth failure.
        status_code: 401 for missing/invalid token, 403 for forbidden.
    """

    def __init__(
        self,
        error_code: ErrorCode,
        message: str,
        status_code: int = 401,
    ) -> None:
        super().__init__(error_code, message, status_code=status_code)
