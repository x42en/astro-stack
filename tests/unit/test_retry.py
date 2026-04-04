"""Unit tests for the pipeline retry policy."""

from __future__ import annotations

import pytest

from app.core.errors import ErrorCode
from app.pipeline.retry import RetryPolicy


class TestRetryPolicy:
    """Tests for RetryPolicy behaviour."""

    def test_should_retry_on_retryable_error_within_limit(self) -> None:
        """Returns True for a retryable error within the attempt limit."""
        policy = RetryPolicy(max_attempts=3)
        assert policy.should_retry(ErrorCode.PIPE_SIRIL_COMMAND_ERROR, attempt=1) is True
        assert policy.should_retry(ErrorCode.PIPE_SIRIL_COMMAND_ERROR, attempt=2) is True

    def test_should_not_retry_when_max_attempts_reached(self) -> None:
        """Returns False when attempt == max_attempts."""
        policy = RetryPolicy(max_attempts=3)
        assert policy.should_retry(ErrorCode.PIPE_SIRIL_COMMAND_ERROR, attempt=3) is False

    def test_should_not_retry_non_retryable_error(self) -> None:
        """Returns False for an error not in the retryable set."""
        policy = RetryPolicy(max_attempts=3)
        assert policy.should_retry(ErrorCode.SESS_NOT_FOUND, attempt=1) is False

    def test_wait_time_grows_exponentially(self) -> None:
        """Wait time doubles with each attempt for backoff_factor=2."""
        policy = RetryPolicy(
            max_attempts=5,
            backoff_seconds=5.0,
            backoff_factor=2.0,
        )
        assert policy.wait_time(1) == pytest.approx(5.0)
        assert policy.wait_time(2) == pytest.approx(10.0)
        assert policy.wait_time(3) == pytest.approx(20.0)

    def test_wait_time_capped_at_max_backoff(self) -> None:
        """Wait time is capped at max_backoff_seconds."""
        policy = RetryPolicy(
            max_attempts=10,
            backoff_seconds=10.0,
            backoff_factor=3.0,
            max_backoff_seconds=60.0,
        )
        # 10 * 3^5 = 2430 > 60
        assert policy.wait_time(6) == pytest.approx(60.0)

    def test_custom_retryable_errors(self) -> None:
        """Custom retryable_errors set is respected."""
        policy = RetryPolicy(
            max_attempts=3,
            retryable_errors=frozenset({ErrorCode.SESS_NOT_FOUND}),
        )
        assert policy.should_retry(ErrorCode.SESS_NOT_FOUND, attempt=1) is True
        assert policy.should_retry(ErrorCode.PIPE_SIRIL_COMMAND_ERROR, attempt=1) is False
