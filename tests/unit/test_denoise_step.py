"""Unit tests for the denoise pipeline step engine dispatch."""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from app.pipeline.base_step import PipelineContext
from app.pipeline.steps.denoise import DenoiseStep


def _make_context(tmp_path: Path) -> PipelineContext:
    work = tmp_path / "work"
    out = tmp_path / "out"
    work.mkdir()
    out.mkdir()
    stretched = work / "stretched.fits"
    stretched.write_bytes(b"fake")
    return PipelineContext(
        job_id=uuid.uuid4(),
        session_id=uuid.uuid4(),
        work_dir=work,
        output_dir=out,
        gpu_device="cuda:0",
        stretched_fits_path=stretched,
    )


@pytest.mark.asyncio
async def test_skips_when_disabled(tmp_path: Path) -> None:
    cosmic = AsyncMock()
    graxpert = AsyncMock()
    step = DenoiseStep(adapter=cosmic, graxpert_adapter=graxpert)

    result = await step.execute(_make_context(tmp_path), {"denoise_enabled": False})

    assert result.skipped is True
    cosmic.denoise.assert_not_called()
    graxpert.denoise.assert_not_called()


@pytest.mark.asyncio
async def test_default_engine_is_cosmic_clarity(tmp_path: Path, monkeypatch) -> None:
    cosmic = AsyncMock()
    graxpert = AsyncMock()
    monkeypatch.setattr(
        "app.pipeline.steps.denoise.save_step_preview",
        AsyncMock(),
    )
    step = DenoiseStep(adapter=cosmic, graxpert_adapter=graxpert)

    config = {"denoise_enabled": True, "denoise_strength": 0.4, "denoise_luminance_only": True}
    result = await step.execute(_make_context(tmp_path), config)

    assert result.success is True
    cosmic.denoise.assert_awaited_once()
    graxpert.denoise.assert_not_called()
    kwargs = cosmic.denoise.await_args.kwargs
    assert kwargs["strength"] == 0.4
    assert kwargs["luminance_only"] is True


@pytest.mark.asyncio
async def test_graxpert_engine_dispatches_with_params(tmp_path: Path, monkeypatch) -> None:
    cosmic = AsyncMock()
    graxpert = AsyncMock()
    monkeypatch.setattr(
        "app.pipeline.steps.denoise.save_step_preview",
        AsyncMock(),
    )
    step = DenoiseStep(adapter=cosmic, graxpert_adapter=graxpert)

    config = {
        "denoise_enabled": True,
        "denoise_engine": "graxpert",
        "denoise_strength": 0.6,
        "denoise_graxpert_ai_model": "3.0.2",
        "denoise_graxpert_batch_size": 8,
    }
    result = await step.execute(_make_context(tmp_path), config)

    assert result.success is True
    graxpert.denoise.assert_awaited_once()
    cosmic.denoise.assert_not_called()
    kwargs = graxpert.denoise.await_args.kwargs
    assert kwargs["strength"] == 0.6
    assert kwargs["ai_model"] == "3.0.2"
    assert kwargs["batch_size"] == 8
    assert result.metadata["denoise_engine"] == "GraXpert"


@pytest.mark.asyncio
async def test_unknown_engine_falls_back_to_cosmic(tmp_path: Path, monkeypatch) -> None:
    cosmic = AsyncMock()
    graxpert = AsyncMock()
    monkeypatch.setattr(
        "app.pipeline.steps.denoise.save_step_preview",
        AsyncMock(),
    )
    step = DenoiseStep(adapter=cosmic, graxpert_adapter=graxpert)

    config = {"denoise_enabled": True, "denoise_engine": "mystery"}
    await step.execute(_make_context(tmp_path), config)

    cosmic.denoise.assert_awaited_once()
    graxpert.denoise.assert_not_called()
