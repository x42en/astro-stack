"""Unit tests for the Siril pipe output parser."""

from __future__ import annotations

import pytest

from app.pipeline.adapters.siril_adapter import SirilEventType, _parse_siril_line


class TestSirilLineParser:
    """Tests for _parse_siril_line()."""

    def test_parses_ready(self) -> None:
        """'ready' line should produce a READY event."""
        event = _parse_siril_line("ready")
        assert event.event_type == SirilEventType.READY

    def test_parses_log(self) -> None:
        """'log: ...' lines should produce a LOG event with message."""
        event = _parse_siril_line("log: Detected Bayer CFA RGGB")
        assert event.event_type == SirilEventType.LOG
        assert event.message == "Detected Bayer CFA RGGB"

    def test_parses_progress(self) -> None:
        """'progress: xx%' lines should produce a PROGRESS event with percent."""
        event = _parse_siril_line("progress: 67.5%")
        assert event.event_type == SirilEventType.PROGRESS
        assert event.percent == pytest.approx(67.5)

    def test_parses_progress_100(self) -> None:
        """Progress can reach 100%."""
        event = _parse_siril_line("progress: 100%")
        assert event.percent == pytest.approx(100.0)

    def test_parses_status_success(self) -> None:
        """'status: success stack' produces STATUS event."""
        event = _parse_siril_line("status: success stack")
        assert event.event_type == SirilEventType.STATUS
        assert event.status_verb == "success"
        assert event.command_name == "stack"

    def test_parses_status_error(self) -> None:
        """'status: error calibrate' produces STATUS event with error verb."""
        event = _parse_siril_line("status: error calibrate")
        assert event.event_type == SirilEventType.STATUS
        assert event.status_verb == "error"
        assert event.command_name == "calibrate"

    def test_parses_status_starting(self) -> None:
        """'status: starting register' produces STATUS event."""
        event = _parse_siril_line("status: starting register")
        assert event.event_type == SirilEventType.STATUS
        assert event.status_verb == "starting"

    def test_unknown_line(self) -> None:
        """Unrecognised lines produce UNKNOWN events."""
        event = _parse_siril_line("garbage output xyz")
        assert event.event_type == SirilEventType.UNKNOWN
        assert "garbage" in event.message

    def test_empty_line(self) -> None:
        """Empty lines produce UNKNOWN events."""
        event = _parse_siril_line("")
        assert event.event_type == SirilEventType.UNKNOWN
