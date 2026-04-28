"""Unit tests for the GraXpert adapter (denoise + helpers)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app.core.errors import ErrorCode, PipelineStepException
from app.pipeline.adapters.graxpert_adapter import (
    GraXpertAdapter,
    _clamp,
    _normalize_ai_version,
)


class TestNormalizeAiVersion:
    def test_passes_through_valid_semver(self) -> None:
        assert _normalize_ai_version("1.0.1") == "1.0.1"
        assert _normalize_ai_version("3.0.2") == "3.0.2"

    def test_strips_graxpert_ai_prefix(self) -> None:
        assert _normalize_ai_version("GraXpert-AI-1.0.0") == "1.0.0"

    def test_strips_graxpert_prefix_and_onnx_suffix(self) -> None:
        assert _normalize_ai_version("GraXpert-3.0.2.onnx") == "3.0.2"

    def test_invalid_falls_back_to_safe_default(self) -> None:
        assert _normalize_ai_version("garbage") == "1.0.1"
        assert _normalize_ai_version("") == "1.0.1"


class TestClamp:
    def test_in_range_returns_value(self) -> None:
        assert _clamp(0.5, 0.0, 1.0, name="x") == 0.5

    def test_above_range_clamped_to_high(self) -> None:
        assert _clamp(99.0, 0.0, 1.0, name="strength") == 1.0

    def test_below_range_clamped_to_low(self) -> None:
        assert _clamp(-5.0, 0.0, 1.0, name="strength") == 0.0


class TestGpuFlag:
    def test_cuda_enables_gpu(self) -> None:
        adapter = GraXpertAdapter(source_path="/tmp/g", models_path="/tmp/m", gpu_device="cuda:0")
        assert adapter._gpu_flag() == "true"

    def test_cpu_disables_gpu(self) -> None:
        adapter = GraXpertAdapter(source_path="/tmp/g", models_path="/tmp/m", gpu_device="cpu")
        assert adapter._gpu_flag() == "false"


class TestBuildCmd:
    def _adapter(self) -> GraXpertAdapter:
        # Force the binary code path (no GraXpert.py present in /tmp).
        adapter = GraXpertAdapter(
            source_path="/tmp/no-graxpert",
            models_path="/tmp/models",
            gpu_device="cuda:0",
        )
        return adapter

    def test_includes_cli_and_cmd(self) -> None:
        adapter = self._adapter()
        cmd = adapter._build_cmd(
            cmd_name="denoising",
            input_path=Path("/tmp/in.fits"),
            output_stem="in_GraXpertDenoise",
            extra_flags=["-strength", "0.500"],
        )
        assert "-cli" in cmd
        assert cmd[cmd.index("-cmd") + 1] == "denoising"
        assert cmd[cmd.index("-output") + 1] == "in_GraXpertDenoise"
        assert cmd[cmd.index("-gpu") + 1] == "true"
        # Input is positional (last token).
        assert cmd[-1] == "/tmp/in.fits"
        # Extra flags appear before the positional input.
        assert "-strength" in cmd and "0.500" in cmd

    def test_background_extraction_layout(self) -> None:
        adapter = self._adapter()
        cmd = adapter._build_cmd(
            cmd_name="background-extraction",
            input_path=Path("/tmp/in.fits"),
            output_stem="in_GraXpertBGE",
            extra_flags=["-ai_version", "1.0.1"],
        )
        assert cmd[cmd.index("-cmd") + 1] == "background-extraction"
        assert "-ai_version" in cmd
        assert cmd[cmd.index("-ai_version") + 1] == "1.0.1"


class TestDenoise:
    @pytest.mark.asyncio
    async def test_builds_denoise_flags_and_clamps(self, tmp_path: Path) -> None:
        adapter = GraXpertAdapter(
            source_path="/tmp/no-graxpert",
            models_path="/tmp/models",
            gpu_device="cuda:0",
        )
        captured: dict = {}

        async def fake_run(**kwargs):
            captured.update(kwargs)

        with patch.object(adapter, "_run_command", side_effect=fake_run) as mock_run:
            await adapter.denoise(
                input_path=tmp_path / "in.fits",
                output_path=tmp_path / "out.fits",
                ai_model="GraXpert-AI-3.0.2",  # tests normalization
                strength=2.5,                  # tests clamping (>1)
                batch_size=99,                 # tests clamping (>32)
            )

        assert mock_run.called
        cmd: list[str] = captured["cmd"]
        assert captured["error_code"] is ErrorCode.PIPE_GRAXPERT_DENOISE_FAILED
        assert captured["step_name"] == "denoise"
        assert captured["log_event"] == "graxpert_denoise"
        # Clamped strength=1.0 and batch_size=32; normalized ai_version=3.0.2
        assert "-strength" in cmd and cmd[cmd.index("-strength") + 1] == "1.000"
        assert "-batch_size" in cmd and cmd[cmd.index("-batch_size") + 1] == "32"
        assert cmd[cmd.index("-ai_version") + 1] == "3.0.2"
        assert cmd[cmd.index("-cmd") + 1] == "denoising"

    @pytest.mark.asyncio
    async def test_remove_background_uses_correct_error_code(self, tmp_path: Path) -> None:
        adapter = GraXpertAdapter(
            source_path="/tmp/no-graxpert",
            models_path="/tmp/models",
            gpu_device="cpu",
        )
        captured: dict = {}

        async def fake_run(**kwargs):
            captured.update(kwargs)

        with patch.object(adapter, "_run_command", side_effect=fake_run):
            await adapter.remove_background(
                input_path=tmp_path / "in.fits",
                output_path=tmp_path / "out.fits",
                ai_model="1.0.1",
            )

        assert captured["error_code"] is ErrorCode.PIPE_GRADIENT_REMOVAL_FAILED
        assert captured["step_name"] == "gradient_removal"
        # CPU mode reflected in the built command.
        assert captured["cmd"][captured["cmd"].index("-gpu") + 1] == "false"


class TestRunCommand:
    @pytest.mark.asyncio
    async def test_missing_binary_raises_external_tool_missing(self, tmp_path: Path) -> None:
        adapter = GraXpertAdapter(
            source_path="/tmp/no-graxpert",
            models_path="/tmp/models",
            gpu_device="cpu",
        )

        async def fake_exec(*args, **kwargs):
            raise FileNotFoundError("no such file")

        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            with pytest.raises(PipelineStepException) as exc_info:
                await adapter._run_command(
                    cmd=["graxpert"],
                    input_path=tmp_path / "in.fits",
                    output_stem="in_GraXpertDenoise",
                    output_path=tmp_path / "out.fits",
                    timeout=10.0,
                    error_code=ErrorCode.PIPE_GRAXPERT_DENOISE_FAILED,
                    step_name="denoise",
                    log_event="graxpert_denoise",
                )

        assert exc_info.value.error_code is ErrorCode.SYS_EXTERNAL_TOOL_MISSING

    @pytest.mark.asyncio
    async def test_nonzero_exit_raises_step_failure(self, tmp_path: Path) -> None:
        adapter = GraXpertAdapter(
            source_path="/tmp/no-graxpert",
            models_path="/tmp/models",
            gpu_device="cpu",
        )

        proc = AsyncMock()
        proc.communicate = AsyncMock(return_value=(b"", b"boom"))
        proc.returncode = 2

        async def fake_exec(*args, **kwargs):
            return proc

        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            with pytest.raises(PipelineStepException) as exc_info:
                await adapter._run_command(
                    cmd=["graxpert"],
                    input_path=tmp_path / "in.fits",
                    output_stem="in_GraXpertDenoise",
                    output_path=tmp_path / "out.fits",
                    timeout=10.0,
                    error_code=ErrorCode.PIPE_GRAXPERT_DENOISE_FAILED,
                    step_name="denoise",
                    log_event="graxpert_denoise",
                )

        assert exc_info.value.error_code is ErrorCode.PIPE_GRAXPERT_DENOISE_FAILED

    @pytest.mark.asyncio
    async def test_renames_produced_file_to_output_path(self, tmp_path: Path) -> None:
        adapter = GraXpertAdapter(
            source_path="/tmp/no-graxpert",
            models_path="/tmp/models",
            gpu_device="cpu",
        )
        input_path = tmp_path / "in.fits"
        input_path.write_bytes(b"fake")
        produced = tmp_path / "in_GraXpertDenoise.fits"
        output = tmp_path / "out" / "denoised.fits"

        proc = AsyncMock()
        proc.communicate = AsyncMock(return_value=(b"", b""))
        proc.returncode = 0

        async def fake_exec(*args, **kwargs):
            # Simulate GraXpert writing the output sibling file.
            produced.write_bytes(b"denoised")
            return proc

        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            await adapter._run_command(
                cmd=["graxpert"],
                input_path=input_path,
                output_stem="in_GraXpertDenoise",
                output_path=output,
                timeout=10.0,
                error_code=ErrorCode.PIPE_GRAXPERT_DENOISE_FAILED,
                step_name="denoise",
                log_event="graxpert_denoise",
            )

        assert output.exists()
        assert output.read_bytes() == b"denoised"
        assert not produced.exists()
