"""Unit tests for the GraXpert ``ai_model`` selector parser.

The ``gradient_removal_ai_model`` profile field doubles as a combined
``mode + version`` selector so the existing UI dropdown can expose every
GraXpert AI capability through one control.  Backwards compatibility with
the bare semver form (BGE) is critical — older imported profiles must
continue to run BGE.
"""

from __future__ import annotations

import pytest

from app.pipeline.steps.gradient_removal import _parse_ai_model


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1.0.1", ("bge", "1.0.1")),
        ("3.0.2", ("bge", "3.0.2")),  # legacy semver tolerated
        ("deconv-obj-1.0.1", ("deconv-obj", "1.0.1")),
        ("deconv-stars-1.0.0", ("deconv-stars", "1.0.0")),
        ("deconv-both-1.0.1", ("deconv-both", "1.0.1")),
    ],
)
def test_parse_ai_model_known_values(value: str, expected: tuple[str, str]) -> None:
    """Each documented selector value parses to the right ``(mode, version)`` pair."""
    assert _parse_ai_model(value) == expected


@pytest.mark.parametrize("value", ["", None, 0])
def test_parse_ai_model_falls_back_to_bge_default(value: object) -> None:
    """Empty / non-string values fall back to BGE 1.0.1 (the global default)."""
    assert _parse_ai_model(value) == ("bge", "1.0.1")  # type: ignore[arg-type]
