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
        "gradient_removal_ai_model": "auto",
        "super_resolution_enabled": False,
        "super_resolution_mode": "auto",
        "star_separation_enabled": False,
        "star_separation_mode": "auto",
    }


@pytest.mark.asyncio
async def test_galaxy_overrides_are_applied() -> None:
    orch = _make_orchestrator()
    ctx = _make_context()
    ctx.metadata["object_name_hint"] = "M81"
    config = _galaxy_config()

    await orch._apply_adaptive_overrides(ctx, config)

    assert config["stretch_strength"] == 30.0
    assert config["denoise_strength"] == 0.40
    assert config["sharpen_stellar_amount"] == 0.20
    assert config["sharpen_nonstellar_amount"] == 0.25
    # Galaxy is no longer hard-skipped — it is re-routed to chained
    # Object + Stars deconvolution via the catalogue's ``"auto"`` resolver.
    assert config["gradient_removal_enabled"] is True
    assert config["gradient_removal_ai_model"] == "deconv-both-1.0.1"

    applied = ctx.metadata["adaptive_overrides_applied"]
    assert applied["object_type"] == "galaxy"
    assert "stretch_strength" in applied["fields"]
    assert "gradient_removal_ai_model" in applied["fields"]
    assert applied["fields"]["gradient_removal_ai_model"]["source"] == "auto_catalogue"
    assert applied["fields"]["stretch_strength"]["source"] == "auto"


@pytest.mark.asyncio
async def test_no_overrides_for_unknown_target() -> None:
    orch = _make_orchestrator()
    ctx = _make_context()
    ctx.metadata["object_name_hint"] = "ZZZ-unknown"
    config = _galaxy_config()
    snapshot = dict(config)

    await orch._apply_adaptive_overrides(ctx, config)

    # ``gradient_removal_ai_model="auto"`` is always resolved (Phase 0bis),
    # falling back to the historical BGE 1.0.1 default for unknown types so
    # downstream steps never see the placeholder.
    assert config["gradient_removal_ai_model"] == "1.0.1"
    applied = ctx.metadata["adaptive_overrides_applied"]
    assert applied["object_type"] is None
    assert applied["fields"]["gradient_removal_ai_model"]["source"] == "auto_default"
    # Numeric / boolean fields are untouched on unknown types.
    for key in (
        "stretch_strength",
        "denoise_strength",
        "sharpen_stellar_amount",
        "sharpen_nonstellar_amount",
        "gradient_removal_enabled",
    ):
        assert config[key] == snapshot[key]


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


# ── Tri-state mode for super_resolution / star_separation ────────────────


@pytest.mark.asyncio
async def test_super_resolution_force_off_overrides_user_enabled() -> None:
    """``super_resolution_mode='off'`` disables the step regardless of
    object type, even when the catalogue would have allowed it."""
    orch = _make_orchestrator()
    ctx = _make_context()
    ctx.metadata["object_name_hint"] = "M81"  # galaxy → catalogue allows super-res
    config = _galaxy_config()
    config["super_resolution_enabled"] = True
    config["super_resolution_mode"] = "off"

    await orch._apply_adaptive_overrides(ctx, config)

    assert config["super_resolution_enabled"] is False
    assert (
        ctx.metadata["adaptive_overrides_applied"]["fields"][
            "super_resolution_enabled"
        ]["source"]
        == "force_off"
    )


@pytest.mark.asyncio
async def test_super_resolution_force_on_overrides_nebula_skip() -> None:
    """``super_resolution_mode='on'`` runs the step even on bright nebulae
    where the catalogue would normally skip it."""
    orch = _make_orchestrator()
    ctx = _make_context()
    ctx.metadata["object_name_hint"] = "M42"
    config = _galaxy_config()
    config["super_resolution_enabled"] = False
    config["super_resolution_mode"] = "on"

    await orch._apply_adaptive_overrides(ctx, config)

    assert config["super_resolution_enabled"] is True
    assert (
        ctx.metadata["adaptive_overrides_applied"]["fields"][
            "super_resolution_enabled"
        ]["source"]
        == "force_on"
    )


@pytest.mark.asyncio
async def test_super_resolution_auto_preserves_catalogue_skip_on_nebula() -> None:
    """When the user keeps ``mode='auto'`` the catalogue still skips
    super-resolution on bright nebulae."""
    orch = _make_orchestrator()
    ctx = _make_context()
    ctx.metadata["object_name_hint"] = "M42"
    config = _galaxy_config()
    config["super_resolution_enabled"] = True
    config["super_resolution_mode"] = "auto"

    await orch._apply_adaptive_overrides(ctx, config)

    assert config["super_resolution_enabled"] is False
    assert (
        ctx.metadata["adaptive_overrides_applied"]["fields"][
            "super_resolution_enabled"
        ]["source"]
        == "auto"
    )


@pytest.mark.asyncio
async def test_star_separation_force_off_overrides_user_enabled() -> None:
    orch = _make_orchestrator()
    ctx = _make_context()
    ctx.metadata["object_name_hint"] = "M42"  # nebula → catalogue allows star-sep
    config = _galaxy_config()
    config["star_separation_enabled"] = True
    config["star_separation_mode"] = "off"

    await orch._apply_adaptive_overrides(ctx, config)

    assert config["star_separation_enabled"] is False
    assert (
        ctx.metadata["adaptive_overrides_applied"]["fields"][
            "star_separation_enabled"
        ]["source"]
        == "force_off"
    )


@pytest.mark.asyncio
async def test_star_separation_force_on_overrides_galaxy_skip() -> None:
    orch = _make_orchestrator()
    ctx = _make_context()
    ctx.metadata["object_name_hint"] = "M81"
    config = _galaxy_config()
    config["star_separation_enabled"] = False
    config["star_separation_mode"] = "on"

    await orch._apply_adaptive_overrides(ctx, config)

    assert config["star_separation_enabled"] is True
    assert (
        ctx.metadata["adaptive_overrides_applied"]["fields"][
            "star_separation_enabled"
        ]["source"]
        == "force_on"
    )


@pytest.mark.asyncio
async def test_star_separation_auto_preserves_catalogue_skip_on_galaxy() -> None:
    orch = _make_orchestrator()
    ctx = _make_context()
    ctx.metadata["object_name_hint"] = "M81"
    config = _galaxy_config()
    config["star_separation_enabled"] = True
    config["star_separation_mode"] = "auto"

    await orch._apply_adaptive_overrides(ctx, config)

    assert config["star_separation_enabled"] is False
    assert (
        ctx.metadata["adaptive_overrides_applied"]["fields"][
            "star_separation_enabled"
        ]["source"]
        == "auto"
    )


@pytest.mark.asyncio
async def test_tri_state_works_without_object_type() -> None:
    """Force ON / OFF must be honoured even when the catalogue cannot
    identify the target (no hint, or unknown name)."""
    orch = _make_orchestrator()
    ctx = _make_context()
    ctx.metadata["object_name_hint"] = "ZZZ-unknown"
    config = _galaxy_config()
    config["super_resolution_enabled"] = True
    config["super_resolution_mode"] = "off"

    await orch._apply_adaptive_overrides(ctx, config)

    assert config["super_resolution_enabled"] is False
    applied = ctx.metadata["adaptive_overrides_applied"]
    assert applied["object_type"] is None
    assert applied["fields"]["super_resolution_enabled"]["source"] == "force_off"
