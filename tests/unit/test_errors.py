"""Unit tests for the error hierarchy and ErrorCode enum."""

from __future__ import annotations

import pytest

from app.core.errors import (
    AstroStackException,
    AuthException,
    ConflictException,
    ErrorCode,
    NotFoundException,
    PipelineStepException,
    ValidationException,
)


class TestErrorCode:
    """Tests for the ErrorCode enum."""

    def test_all_session_codes_have_sess_prefix(self) -> None:
        """All session error codes must start with SESS_."""
        session_codes = [c for c in ErrorCode if c.value.startswith("SESS_")]
        assert len(session_codes) > 0

    def test_all_job_codes_have_job_prefix(self) -> None:
        """All job error codes must start with JOB_."""
        job_codes = [c for c in ErrorCode if c.value.startswith("JOB_")]
        assert len(job_codes) > 0

    def test_error_codes_are_unique(self) -> None:
        """All error code values must be unique."""
        values = [c.value for c in ErrorCode]
        assert len(values) == len(set(values))

    def test_error_code_is_string(self) -> None:
        """ErrorCode inherits from str so values can be used as plain strings."""
        assert isinstance(ErrorCode.SESS_NOT_FOUND, str)
        assert ErrorCode.SESS_NOT_FOUND == "SESS_NOT_FOUND"


class TestAstroStackException:
    """Tests for the base exception class."""

    def test_to_dict_contains_error_code(self) -> None:
        """to_dict() must include error_code, message, and details keys."""
        exc = AstroStackException(
            ErrorCode.SYS_INTERNAL_ERROR,
            "Something went wrong",
            details={"key": "value"},
        )
        d = exc.to_dict()
        assert d["error_code"] == "SYS_INTERNAL_ERROR"
        assert d["message"] == "Something went wrong"
        assert d["details"] == {"key": "value"}

    def test_default_status_code_is_500(self) -> None:
        """Default HTTP status code must be 500."""
        exc = AstroStackException(ErrorCode.SYS_INTERNAL_ERROR, "error")
        assert exc.status_code == 500

    def test_not_retryable_by_default(self) -> None:
        """Exceptions are not retryable by default."""
        exc = AstroStackException(ErrorCode.SYS_INTERNAL_ERROR, "error")
        assert exc.retryable is False


class TestSubclassExceptions:
    """Tests for convenience exception subclasses."""

    def test_not_found_has_404_status(self) -> None:
        """NotFoundException must use HTTP 404."""
        exc = NotFoundException(ErrorCode.SESS_NOT_FOUND, "Not found")
        assert exc.status_code == 404

    def test_conflict_has_409_status(self) -> None:
        """ConflictException must use HTTP 409."""
        exc = ConflictException(ErrorCode.SESS_ALREADY_PROCESSING, "Conflict")
        assert exc.status_code == 409

    def test_validation_has_422_status(self) -> None:
        """ValidationException must use HTTP 422."""
        exc = ValidationException(ErrorCode.PROF_VALIDATION_ERROR, "Invalid")
        assert exc.status_code == 422

    def test_pipeline_step_exception_captures_step_name(self) -> None:
        """PipelineStepException must store the step name."""
        exc = PipelineStepException(
            ErrorCode.PIPE_SIRIL_COMMAND_ERROR,
            "Siril failed",
            step_name="preprocessing",
        )
        assert exc.step_name == "preprocessing"
        assert exc.details.get("step_name") == "preprocessing"

    def test_pipeline_step_exception_retryable_by_default(self) -> None:
        """PipelineStepException is retryable by default."""
        exc = PipelineStepException(
            ErrorCode.PIPE_SIRIL_COMMAND_ERROR,
            "err",
            step_name="preprocessing",
        )
        assert exc.retryable is True

    def test_auth_exception_default_401(self) -> None:
        """AuthException uses 401 by default."""
        exc = AuthException(ErrorCode.AUTH_TOKEN_INVALID, "Invalid token")
        assert exc.status_code == 401
