"""Unit tests for WebSocket event schemas."""

from __future__ import annotations

import json
import uuid

import pytest

from app.domain.ws_event import (
    CompletedEvent,
    ErrorEvent,
    EventType,
    LogEvent,
    LogLevel,
    LogSource,
    ProgressEvent,
    SessionReadyEvent,
    StepStatusEvent,
    StepStatusValue,
)


@pytest.fixture()
def job_id() -> uuid.UUID:
    """Return a fixed job UUID for tests."""
    return uuid.UUID("12345678-1234-5678-1234-567812345678")


@pytest.fixture()
def session_id() -> uuid.UUID:
    """Return a fixed session UUID for tests."""
    return uuid.UUID("87654321-4321-8765-4321-876543218765")


class TestProgressEvent:
    """Tests for ProgressEvent schema."""

    def test_serialises_to_json(self, job_id: uuid.UUID, session_id: uuid.UUID) -> None:
        """ProgressEvent serialises to valid JSON with required fields."""
        event = ProgressEvent(
            job_id=job_id,
            session_id=session_id,
            step="stacking",
            step_index=3,
            total_steps=9,
            percent=67.5,
            message="Stacking 42/63 frames...",
        )
        data = json.loads(event.model_dump_json())
        assert data["type"] == EventType.PROGRESS
        assert data["percent"] == 67.5
        assert data["step"] == "stacking"

    def test_percent_clamped_to_0_100(self) -> None:
        """Percent field must be within 0–100 range."""
        with pytest.raises(Exception):
            ProgressEvent(
                step="test",
                step_index=0,
                total_steps=1,
                percent=150.0,  # invalid
                message="",
            )


class TestLogEvent:
    """Tests for LogEvent schema."""

    def test_default_level_is_info(self, job_id: uuid.UUID) -> None:
        """Default log level is INFO."""
        event = LogEvent(job_id=job_id, message="test")
        assert event.level == LogLevel.INFO

    def test_default_source_is_system(self, job_id: uuid.UUID) -> None:
        """Default log source is SYSTEM."""
        event = LogEvent(job_id=job_id, message="test")
        assert event.source == LogSource.SYSTEM


class TestErrorEvent:
    """Tests for ErrorEvent schema."""

    def test_error_event_contains_error_code(
        self, job_id: uuid.UUID, session_id: uuid.UUID
    ) -> None:
        """ErrorEvent must carry the error_code string."""
        event = ErrorEvent(
            job_id=job_id,
            session_id=session_id,
            error_code="PIPE_SIRIL_COMMAND_ERROR",
            message="Siril failed",
            step="preprocessing",
            retryable=True,
            attempt=2,
            max_attempts=3,
        )
        data = json.loads(event.model_dump_json())
        assert data["error_code"] == "PIPE_SIRIL_COMMAND_ERROR"
        assert data["retryable"] is True
        assert data["attempt"] == 2


class TestCompletedEvent:
    """Tests for CompletedEvent schema."""

    def test_completed_event_has_duration(self, job_id: uuid.UUID, session_id: uuid.UUID) -> None:
        """CompletedEvent must include duration and outputs."""
        event = CompletedEvent(
            job_id=job_id,
            session_id=session_id,
            duration_seconds=842.0,
            outputs={
                "fits": "/output/final.fits",
                "jpeg": "/output/preview.jpg",
            },
        )
        assert event.type == EventType.COMPLETED
        assert event.duration_seconds == pytest.approx(842.0)
        assert "fits" in event.outputs


class TestStepStatusEvent:
    """Tests for StepStatusEvent schema."""

    def test_step_status_starting(self, job_id: uuid.UUID) -> None:
        """StepStatusEvent correctly captures step name and status."""
        event = StepStatusEvent(
            job_id=job_id,
            step="plate_solving",
            step_index=2,
            status=StepStatusValue.STARTING,
        )
        assert event.status == StepStatusValue.STARTING
        assert event.step == "plate_solving"


class TestSessionReadyEvent:
    """Tests for SessionReadyEvent schema."""

    def test_session_ready_event_frame_counts(self, session_id: uuid.UUID) -> None:
        """SessionReadyEvent carries frame count metadata."""
        event = SessionReadyEvent(
            session_id=session_id,
            frame_count_lights=42,
            frame_count_darks=20,
            frame_count_flats=15,
            frame_count_bias=30,
            input_format="fits",
        )
        assert event.frame_count_lights == 42
        assert event.input_format == "fits"
