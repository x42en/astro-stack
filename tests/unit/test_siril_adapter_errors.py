"""Unit tests for Siril adapter error classification logic.

These tests focus on the ``run_command`` ``_collect`` branch that converts
Siril ``status: error`` events into typed ``PipelineStepException`` instances
with the appropriate ``error_code`` and ``retryable`` flag.
"""

from __future__ import annotations

from typing import AsyncGenerator

import pytest

from app.core.errors import ErrorCode, PipelineStepException
from app.pipeline.adapters.siril_adapter import (
    SirilAdapter,
    SirilEvent,
    SirilEventType,
)


def _log(msg: str) -> SirilEvent:
    return SirilEvent(event_type=SirilEventType.LOG, message=msg)


def _error(message: str = "error register", command: str = "register") -> SirilEvent:
    return SirilEvent(
        event_type=SirilEventType.STATUS,
        message=message,
        status_verb="error",
        command_name=command,
    )


class _FakeAdapter(SirilAdapter):
    """Test double that bypasses the subprocess + pipe machinery.

    We only need ``run_command`` to consume an async event stream, so we
    override ``send_command`` (no-op) and ``stream_output`` (yields canned
    events). The constructor avoids parent ``__init__`` so we never touch
    the file system or the Siril binary.
    """

    def __init__(self, events: list[SirilEvent]) -> None:  # noqa: D401 - test helper
        self._events = events

    async def send_command(self, command: str) -> None:  # type: ignore[override]
        return None

    async def stream_output(self) -> AsyncGenerator[SirilEvent, None]:  # type: ignore[override]
        for e in self._events:
            yield e


@pytest.mark.asyncio
async def test_alignment_failure_is_non_retryable() -> None:
    """The two-pass register failure must surface as PIPE_REGISTRATION_FAILED.

    Reproduces the exact log signature observed in production on a wide-field
    Lion-constellation dataset where Siril burned five reference-image trials
    before giving up.
    """
    events = [
        _log("Trial #1: After sequence analysis, we are choosing image 8 ..."),
        _log("Cannot perform star matching: try #3. Image 1 skipped"),
        _log("Trial #5: After sequence alignment, image #12 could not align "
             "more than half of the frames, recomputing"),
        _log("Could not find an image that aligns more than itself, aborting"),
        _error(),
    ]
    adapter = _FakeAdapter(events)

    with pytest.raises(PipelineStepException) as exc_info:
        await adapter.run_command("register pp_light -2pass", timeout=5.0)

    assert exc_info.value.error_code is ErrorCode.PIPE_REGISTRATION_FAILED
    assert exc_info.value.retryable is False


@pytest.mark.asyncio
async def test_could_not_align_more_than_half_alone_triggers_registration_failed() -> None:
    """The intermediate ``could not align more than half`` line must also count.

    Even before the final "aborting" message, every trial logs this string.
    Detecting it ensures the failure is classified consistently regardless of
    which Siril log line lands last in the buffer.
    """
    events = [
        _log("Trial #1: After sequence alignment, image #8 could not align "
             "more than half of the frames, recomputing"),
        _error(),
    ]
    adapter = _FakeAdapter(events)

    with pytest.raises(PipelineStepException) as exc_info:
        await adapter.run_command("register pp_light -2pass", timeout=5.0)

    assert exc_info.value.error_code is ErrorCode.PIPE_REGISTRATION_FAILED
    assert exc_info.value.retryable is False


@pytest.mark.asyncio
async def test_disk_full_still_takes_priority() -> None:
    """Disk-full must remain non-retryable even when alignment-failure-like
    log lines also appear in the buffer (priority-ordering regression guard)."""
    events = [
        _log("Could not find an image that aligns more than itself, aborting"),
        _log("Not enough free disk space"),
        _error(),
    ]
    adapter = _FakeAdapter(events)

    with pytest.raises(PipelineStepException) as exc_info:
        await adapter.run_command("register pp_light -2pass", timeout=5.0)

    assert exc_info.value.error_code is ErrorCode.PIPE_DISK_FULL
    assert exc_info.value.retryable is False


@pytest.mark.asyncio
async def test_generic_siril_error_remains_retryable() -> None:
    """Any unrecognised Siril error must keep the historical retryable behaviour."""
    events = [
        _log("Some unrelated log line"),
        _error(message="error stack", command="stack"),
    ]
    adapter = _FakeAdapter(events)

    with pytest.raises(PipelineStepException) as exc_info:
        await adapter.run_command("stack r_pp_light", timeout=5.0)

    assert exc_info.value.error_code is ErrorCode.PIPE_SIRIL_COMMAND_ERROR
    assert exc_info.value.retryable is True
