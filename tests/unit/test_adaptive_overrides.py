"""Unit tests for the orchestrator's adaptive per-object-type overrides."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from app.pipeline.base_step import PipelineContext
from app.pipeline.orchestrator import PipelineOrchestrator
from app.pipeline.utils.object_type import (
    ADAPTIVE_PROFILE_OVERRIDES_BY_TYPE,
    SKIP_GRADIENT_REMOVAL_TYPES,
    resolve_and_cache_object_type,
)


# ── Static lookup tables ─────────────────────────────────────────────────


def test_galaxy_overrides_contain_all_expected_fields() -> None:
    galaxy = ADAPTIVE_PROFILE_OVERRIDES_BY_TYPE["galaxy"]
    assert galaxy["stretch_strength"] == 30.0
    assert galaxy["denoise_strength"] == 0.40
    assert galaxy["sharpen_stellar_amount"] == 0.20
    assert galaxy["sharpen_nonstellar_amount"] == 0.25


def test_cluster_planetary_supernova_only_soften_stretch() -> None:
    assert ADAPTIVE_PROFILE_OVERRIDES_BY_TYPE["cluster"] == {"stretch_strength": 50.0}
    assert ADAPTIVE_PROFILE_OVERRIDES_BY_TYPE["supernova"] == {"stretch_strength": 60.0}
    assert ADAPTIVE_PROFILE_OVERRIDES_BY_TYPE["planetary"] == {"stretch_strength": 80.0}


def test_skip_gradient_removal_types_is_empty() -> None:
    """Galaxies are no longer hard-skipped — the catalogue routes them to
    chained Object + Stars deconvolution via :data:`STRING_OVERRIDES_BY_TYPE`.
    The legacy skip frozenset is kept as an empty extension point."""
    assert SKIP_GRADIENT_REMOVAL_TYPES == frozenset()


# ── resolve_and_cache_object_type ────────────────────────────────────────


def _make_context() -> PipelineContext:
    return PipelineContext(
        job_id=uuid.uuid4(),
        session_id=uuid.uuid4(),
        work_dir=Path("/tmp/work"),
        output_dir=Path("/tmp/out"),
    )


def test_resolve_and_cache_uses_object_name_hint() -> None:
    ctx = _make_context()
    ctx.metadata["object_name_hint"] = "M81"
    assert resolve_and_cache_object_type(ctx) == "galaxy"
    assert ctx.metadata["object_type"] == "galaxy"


def test_resolve_and_cache_falls_back_to_object_name() -> None:
    ctx = _make_context()
    ctx.metadata["object_name"] = "M42"
    assert resolve_and_cache_object_type(ctx) == "nebula"


def test_resolve_and_cache_returns_none_for_unknown() -> None:
    ctx = _make_context()
    ctx.metadata["object_name_hint"] = "ZZZ-not-a-real-object"
    assert resolve_and_cache_object_type(ctx) is None
    assert "object_type" not in ctx.metadata


def test_resolve_and_cache_is_memoised() -> None:
    ctx = _make_context()
    ctx.metadata["object_type"] = "nebula"  # pre-cached
    # Even with a galaxy hint, the cached value wins.
    ctx.metadata["object_name_hint"] = "M81"
    assert resolve_and_cache_object_type(ctx) == "nebula"


# ── _apply_adaptive_overrides ────────────────────────────────────────────


def _make_orchestrator() -> PipelineOrchestrator:
    """Build an orchestrator without running its real __init__."""
    orch = PipelineOrchestrator.__new__(PipelineOrchestrator)
    orch.job_id = uuid.uuid4()
    orch.session_id = uuid.uuid4()
    orch.event_bus = AsyncMock()
    orch.event_bus.publish_job_event = AsyncMock()
    return orch


def _galaxy_config() -> dict[str, Any]:
    return {
        "stretch_method": "asinh",
        "stretch_strength": 150.0,
        "denoise_strength": 0.85,
        "sharpen_stellar_amount": 0.50,
        "sharpen_nonstellar_amount": 0.50,
        "gradient_removal_enabled": True,
    }


@pytest.mark.asyncio
async def test_galaxy_overrides_are_applied() -> None:
    orch = _make_orchestrator()
    ctx = _make_context()
    ctx.metadata["object_name_hint"] = "M81"
    config = _galaxy_config()
    config["gradient_removal_ai_model"] = "1.0.1"

    await orch._apply_adaptive_overrides(ctx, config)

    assert config["stretch_strength"] == 30.0
    assert config["denoise_strength"] == 0.40
    assert config["sharpen_stellar_amount"] == 0.20
    assert config["sharpen_nonstellar_amount"] == 0.25
    # Galaxy is no longer hard-skipped — it is re-routed to chained
    # Object + Stars deconvolution via STRING_OVERRIDES_BY_TYPE.
    assert config["gradient_removal_enabled"] is True
    assert config["gradient_removal_ai_model"] == "deconv-both-1.0.1"

    applied = ctx.metadata["adaptive_overrides_applied"]
    assert applied["object_type"] == "galaxy"
    assert "stretch_strength" in applied["fields"]
    assert "gradient_removal_ai_model" in applied["fields"]


@pytest.mark.asyncio
async def test_no_overrides_for_unknown_target() -> None:
    orch = _make_orchestrator()
    ctx = _make_context()
    ctx.metadata["object_name_hint"] = "ZZZ-unknown"
    config = _galaxy_config()
    snapshot = dict(config)

    await orch._apply_adaptive_overrides(ctx, config)

    assert config == snapshot
    assert "adaptive_overrides_applied" not in ctx.metadata


@pytest.mark.asyncio
async def test_nebula_does_not_skip_gradient_removal() -> None:
    orch = _make_orchestrator()
    ctx = _make_context()
    ctx.metadata["object_name_hint"] = "M42"
    config = _galaxy_config()

    await orch._apply_adaptive_overrides(ctx, config)

    # Nebula is not in SKIP_GRADIENT_REMOVAL_TYPES.
    assert config["gradient_removal_enabled"] is True


@pytest.mark.asyncio
async def test_overrides_never_increase_existing_lower_value() -> None:
    """A user profile already softer than our table should be left alone."""
    orch = _make_orchestrator()
    ctx = _make_context()
    ctx.metadata["object_name_hint"] = "M81"
    config = _galaxy_config()
    config["stretch_strength"] = 10.0  # already < galaxy override of 30
    config["denoise_strength"] = 0.10  # already < 0.40

    await orch._apply_adaptive_overrides(ctx, config)

    assert config["stretch_strength"] == 10.0
    assert config["denoise_strength"] == 0.10
    # Sharpen fields were higher and must still be lowered.
    assert config["sharpen_stellar_amount"] == 0.20


@pytest.mark.asyncio
async def test_stretch_override_skipped_when_method_not_asinh() -> None:
    orch = _make_orchestrator()
    ctx = _make_context()
    ctx.metadata["object_name_hint"] = "M81"
    config = _galaxy_config()
    config["stretch_method"] = "histogram"

    await orch._apply_adaptive_overrides(ctx, config)

    # stretch_strength is preserved when method ≠ asinh.
    assert config["stretch_strength"] == 150.0
    # Other galaxy overrides still apply.
    assert config["denoise_strength"] == 0.40
    # Galaxy step stays enabled — routed to deconv-both.
    assert config["gradient_removal_enabled"] is True


@pytest.mark.asyncio
async def test_cluster_only_softens_stretch() -> None:
    orch = _make_orchestrator()
    ctx = _make_context()
    ctx.metadata["object_name_hint"] = "M13"  # globular cluster
    config = _galaxy_config()

    await orch._apply_adaptive_overrides(ctx, config)

    assert config["stretch_strength"] == 50.0
    assert config["denoise_strength"] == 0.85  # untouched
    assert config["gradient_removal_enabled"] is True  # not skipped
