"""Unit tests for the catalogue-driven object type resolver."""

from __future__ import annotations

import pytest

from app.pipeline.utils.object_type import (
    ADAPTIVE_PROFILE_OVERRIDES_BY_TYPE,
    SKIP_GRADIENT_REMOVAL_TYPES,
    SKIP_STAR_SEPARATION_TYPES,
    SKIP_SUPER_RESOLUTION_TYPES,
    resolve_object_type,
)


@pytest.mark.parametrize("name", ["M81", "m81", "M 81", " M81 "])
def test_resolve_messier_galaxy(name: str) -> None:
    """``M81`` (Bode's Galaxy) resolves to ``galaxy`` in any common form."""
    assert resolve_object_type(name) == "galaxy"


@pytest.mark.parametrize("name", ["M42", "m 42", "M42 — Orion Nebula"])
def test_resolve_messier_nebula(name: str) -> None:
    """``M42`` resolves to ``nebula`` even when followed by descriptive text."""
    assert resolve_object_type(name) == "nebula"


def test_resolve_messier_cluster() -> None:
    """``M45`` (Pleiades) resolves to ``cluster``."""
    assert resolve_object_type("M45") == "cluster"


@pytest.mark.parametrize("name", [None, "", "   ", "NotARealObject"])
def test_unknown_returns_none(name: str | None) -> None:
    """Empty / unknown strings fall back to ``None`` so callers preserve the profile."""
    assert resolve_object_type(name) is None


# ── Object-type adaptation policies ──────────────────────────────────────


def test_nebula_caps_stretch_strength() -> None:
    """The nebula override caps ``stretch_strength`` at 150 to avoid burning the
    M42-class core that the Quality preset's 180 produces."""
    assert ADAPTIVE_PROFILE_OVERRIDES_BY_TYPE["nebula"]["stretch_strength"] == 150.0


def test_galaxy_skips_gradient_removal() -> None:
    """GraXpert (AI + polynomial) destroys low-SNR diffuse targets — verified on M81."""
    assert "galaxy" in SKIP_GRADIENT_REMOVAL_TYPES


def test_nebula_skips_super_resolution() -> None:
    """Cosmic Clarity 2× amplifies clipped pixels into reconstruction artefacts
    on bright nebula cores; opt-in must be auto-disabled."""
    assert "nebula" in SKIP_SUPER_RESOLUTION_TYPES


def test_galaxy_and_cluster_skip_star_separation() -> None:
    """Star separation destroys HII regions on galaxies and the very subject of
    clusters; opt-in must be auto-disabled."""
    assert "galaxy" in SKIP_STAR_SEPARATION_TYPES
    assert "cluster" in SKIP_STAR_SEPARATION_TYPES


def test_nebula_keeps_super_resolution_disabled_by_default() -> None:
    """Ensure no accidental cross-listing: nebulae must not be in the
    star-separation skip list (the default 0.8/0.5 weights with the new
    envelope-preserving recombine work correctly on nebulae)."""
    assert "nebula" not in SKIP_STAR_SEPARATION_TYPES
    assert "nebula" not in SKIP_GRADIENT_REMOVAL_TYPES
