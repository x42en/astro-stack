"""Unit tests for SirilPyAdapter (pyscript-based headless Python driving)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.errors import ErrorCode, PipelineStepException
from app.pipeline.adapters.siril_pyadapter import SirilPyAdapter


def _fake_proc(returncode: int, stdout: bytes = b"", stderr: bytes = b"") -> MagicMock:
    proc = MagicMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    return proc


class TestRunScript:
    @pytest.mark.asyncio
    async def test_parses_result_marker(self, tmp_path: Path) -> None:
        adapter = SirilPyAdapter(work_dir=tmp_path)
        stdout = b'log: hello\nlog: SIRILPY_RESULT {"mean": 1.5}\nlog: bye\n'
        with patch(
            "asyncio.create_subprocess_exec",
            AsyncMock(return_value=_fake_proc(0, stdout=stdout)),
        ):
            result = await adapter.run_script("print('hi')")
        assert result == {"mean": 1.5}

    @pytest.mark.asyncio
    async def test_writes_requires_and_pyscript_wrapper(self, tmp_path: Path) -> None:
        adapter = SirilPyAdapter(work_dir=tmp_path, min_version="1.4.2")
        captured_cmd: list[str] = []

        async def fake_exec(*cmd: str, **kwargs: object) -> MagicMock:
            captured_cmd.extend(cmd)
            wrapper_path = Path(cmd[-1])
            wrapper_text = wrapper_path.read_text()
            assert wrapper_text.startswith("requires 1.4.2\n")
            assert "pyscript " in wrapper_text
            return _fake_proc(0, stdout=b'log: SIRILPY_RESULT {}\n')

        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            await adapter.run_script("print('hi')")

        assert captured_cmd[0] == adapter.siril_binary
        assert "-d" in captured_cmd
        assert "-s" in captured_cmd

    @pytest.mark.asyncio
    async def test_missing_binary_raises_external_tool_missing(self, tmp_path: Path) -> None:
        adapter = SirilPyAdapter(work_dir=tmp_path, siril_binary="no-such-siril-cli")
        with patch(
            "asyncio.create_subprocess_exec",
            AsyncMock(side_effect=FileNotFoundError()),
        ):
            with pytest.raises(PipelineStepException) as exc_info:
                await adapter.run_script("print('hi')")
        assert exc_info.value.error_code is ErrorCode.SYS_EXTERNAL_TOOL_MISSING
        assert exc_info.value.retryable is False

    @pytest.mark.asyncio
    async def test_timeout_raises_retryable(self, tmp_path: Path) -> None:
        adapter = SirilPyAdapter(work_dir=tmp_path)

        async def fake_exec(*_args: object, **_kwargs: object) -> MagicMock:
            proc = MagicMock()
            proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
            return proc

        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            with pytest.raises(PipelineStepException) as exc_info:
                await adapter.run_script("print('hi')", timeout=0.01)
        assert exc_info.value.error_code is ErrorCode.PIPE_SIRILPY_SCRIPT_FAILED
        assert exc_info.value.retryable is True

    @pytest.mark.asyncio
    async def test_nonzero_exit_raises_retryable(self, tmp_path: Path) -> None:
        adapter = SirilPyAdapter(work_dir=tmp_path)
        with patch(
            "asyncio.create_subprocess_exec",
            AsyncMock(return_value=_fake_proc(1, stderr=b"Traceback: boom")),
        ):
            with pytest.raises(PipelineStepException) as exc_info:
                await adapter.run_script("raise ValueError('boom')")
        assert exc_info.value.error_code is ErrorCode.PIPE_SIRILPY_SCRIPT_FAILED
        assert exc_info.value.retryable is True
        assert "boom" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_missing_marker_raises_non_retryable(self, tmp_path: Path) -> None:
        adapter = SirilPyAdapter(work_dir=tmp_path)
        with patch(
            "asyncio.create_subprocess_exec",
            AsyncMock(return_value=_fake_proc(0, stdout=b"log: no result here\n")),
        ):
            with pytest.raises(PipelineStepException) as exc_info:
                await adapter.run_script("print('no marker')")
        assert exc_info.value.error_code is ErrorCode.PIPE_SIRILPY_SCRIPT_FAILED
        assert exc_info.value.retryable is False

    @pytest.mark.asyncio
    async def test_malformed_json_raises_non_retryable(self, tmp_path: Path) -> None:
        adapter = SirilPyAdapter(work_dir=tmp_path)
        with patch(
            "asyncio.create_subprocess_exec",
            AsyncMock(return_value=_fake_proc(0, stdout=b"log: SIRILPY_RESULT {not json\n")),
        ):
            with pytest.raises(PipelineStepException) as exc_info:
                await adapter.run_script("print('bad json')")
        assert exc_info.value.error_code is ErrorCode.PIPE_SIRILPY_SCRIPT_FAILED
        assert exc_info.value.retryable is False


class TestGetImageStats:
    @pytest.mark.asyncio
    async def test_builds_load_stat_and_get_stats_calls(self, tmp_path: Path) -> None:
        adapter = SirilPyAdapter(work_dir=tmp_path)
        captured_script: dict[str, str] = {}

        async def fake_run_script(python_source: str, **_kwargs: object) -> dict:
            captured_script["source"] = python_source
            return {"mean": 10.0, "median": 9.0, "sigma": 2.0, "bgnoise": 1.0, "min": 0.0, "max": 20.0}

        with patch.object(adapter, "run_script", side_effect=fake_run_script):
            stats = await adapter.get_image_stats(tmp_path / "dark.fits", channel=1)

        assert stats["mean"] == 10.0
        assert 'siril.cmd("load"' in captured_script["source"]
        assert 'siril.cmd("stat")' in captured_script["source"]
        assert "get_image_stats(1)" in captured_script["source"]


class TestGetImageStars:
    @pytest.mark.asyncio
    async def test_defaults_channel_to_none(self, tmp_path: Path) -> None:
        adapter = SirilPyAdapter(work_dir=tmp_path)
        captured_script: dict[str, str] = {}

        async def fake_run_script(python_source: str, **_kwargs: object) -> list:
            captured_script["source"] = python_source
            return [{"fwhm": 2.5, "snr": 30.0, "roundness": 0.9, "mag": 14.2}]

        with patch.object(adapter, "run_script", side_effect=fake_run_script):
            stars = await adapter.get_image_stars(tmp_path / "light.fits")

        assert stars == [{"fwhm": 2.5, "snr": 30.0, "roundness": 0.9, "mag": 14.2}]
        assert "channel=None" in captured_script["source"]

    @pytest.mark.asyncio
    async def test_non_list_result_returns_empty(self, tmp_path: Path) -> None:
        adapter = SirilPyAdapter(work_dir=tmp_path)
        with patch.object(adapter, "run_script", AsyncMock(return_value={})):
            stars = await adapter.get_image_stars(tmp_path / "light.fits")
        assert stars == []
