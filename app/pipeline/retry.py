"""Retry policy for pipeline step execution.

Defines the :class:`RetryPolicy` dataclass consumed by the orchestrator to
determine whether a failed step should be retried, and how long to wait
between attempts.

Example:
    >>> policy = RetryPolicy(max_attempts=3, backoff_seconds=5.0)
    >>> if policy.should_retry(error_code, attempt=1):
    ...     await asyncio.sleep(policy.wait_time(attempt=1))
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from app.core.errors import ErrorCode
from app.core.logging import get_logger

logger = get_logger(__name__)

# Error codes that may be safely retried (transient failures)
_DEFAULT_RETRYABLE: frozenset[ErrorCode] = frozenset(
    {
        ErrorCode.PIPE_SIRIL_COMMAND_ERROR,
        ErrorCode.PIPE_SIRIL_TIMEOUT,
        ErrorCode.PIPE_GRADIENT_REMOVAL_FAILED,
        ErrorCode.PIPE_COSMIC_DENOISE_FAILED,
        ErrorCode.PIPE_COSMIC_SHARPEN_FAILED,
        ErrorCode.PIPE_COSMIC_SUPERRES_FAILED,
        ErrorCode.SYS_REDIS_UNAVAILABLE,
    }
)


@dataclass
class RetryPolicy:
    """Configuration for step-level retry behaviour.

    Attributes:
        max_attempts: Maximum number of execution attempts (including the first).
            A value of 1 means no retries.
        backoff_seconds: Base wait time (in seconds) between attempts.
        backoff_factor: Multiplier applied to ``backoff_seconds`` on each retry.
            A value of 2.0 doubles the wait after each failure.
        max_backoff_seconds: Upper ceiling on the computed wait time.
        retryable_errors: Set of :class:`~app.core.errors.ErrorCode` values
            for which a retry is permitted. Errors outside this set cause
            immediate failure.
    """

    max_attempts: int = 3
    backoff_seconds: float = 5.0
    backoff_factor: float = 2.0
    max_backoff_seconds: float = 60.0
    retryable_errors: frozenset[ErrorCode] = field(default_factory=lambda: _DEFAULT_RETRYABLE)

    def should_retry(self, error_code: ErrorCode, attempt: int) -> bool:
        """Determine whether a failed attempt should be retried.

        Args:
            error_code: The :class:`~app.core.errors.ErrorCode` of the failure.
            attempt: The number of attempts already made (1-based).

        Returns:
            ``True`` if another attempt should be scheduled.
        """
        if attempt >= self.max_attempts:
            logger.debug(
                "retry_exhausted",
                error_code=error_code.value,
                attempt=attempt,
                max_attempts=self.max_attempts,
            )
            return False
        if error_code not in self.retryable_errors:
            logger.debug(
                "error_not_retryable",
                error_code=error_code.value,
            )
            return False
        return True

    def wait_time(self, attempt: int) -> float:
        """Compute the wait time before a retry attempt.

        Uses exponential backoff: ``backoff_seconds * backoff_factor^(attempt-1)``.

        Args:
            attempt: The 1-based attempt number that just failed.

        Returns:
            Wait duration in seconds (capped at :attr:`max_backoff_seconds`).
        """
        raw = self.backoff_seconds * (self.backoff_factor ** (attempt - 1))
        return min(raw, self.max_backoff_seconds)

    async def wait(self, attempt: int) -> None:
        """Sleep for the computed backoff duration.

        Args:
            attempt: The 1-based attempt number that just failed.
        """
        delay = self.wait_time(attempt)
        logger.info("retry_backoff", delay_seconds=delay, attempt=attempt)
        await asyncio.sleep(delay)
