"""Unit tests for the catalogue-driven object type resolver."""

from __future__ import annotations

import pytest

from app.pipeline.utils.object_type import resolve_object_type


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
