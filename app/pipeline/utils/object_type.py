"""Resolve a celestial object name to a high-level catalogue type.

Used by the processing pipeline to lightly adapt parameters (e.g. stretch
strength) when the bundled Messier / NGC catalogue can identify the target.
The resolver intentionally returns ``None`` on any miss so that callers fall
back to the user-supplied profile values without surprises.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Literal

from app.infrastructure.catalog.messier import get_by_id, search

if TYPE_CHECKING:
    from app.pipeline.base_step import PipelineContext

ObjectType = Literal[
    "galaxy",
    "nebula",
    "cluster",
    "planetary",
    "supernova",
    "other",
]

# Match a leading catalogue token in display strings such as
# ``"M81"``, ``"M 81"``, ``"NGC 3031"``, ``"IC 1396"``, ``"M81 — Bode"``.
_CATALOG_TOKEN = re.compile(
    r"^\s*((?:M|NGC|IC|C)\s*\d+[A-Za-z]?)",
    re.IGNORECASE,
)


def resolve_object_type(name: str | None) -> ObjectType | None:
    """Return the catalogue type for ``name``, or ``None`` if unknown.

    The lookup is forgiving: it strips whitespace, decoration suffixes such
    as ``"M81 — Bode's Galaxy"`` and tolerates either ``"M81"`` or
    ``"M 81"``.  When no exact id match is found a substring search is
    attempted before giving up.

    Args:
        name: Free-form object name, typically ``session.object_name``.

    Returns:
        One of the :data:`ObjectType` literals on a hit; ``None`` otherwise.
    """
    if not name:
        return None
    stripped = name.strip()
    if not stripped:
        return None

    # 1. Try the leading catalogue token (handles "M81 — Bode's Galaxy").
    match = _CATALOG_TOKEN.match(stripped)
    if match:
        token = match.group(1)
        obj = get_by_id(token)
        if obj is not None:
            return obj.type  # type: ignore[return-value]

    # 2. Try the whole string as an id (handles bare "M81", "NGC3031").
    obj = get_by_id(stripped)
    if obj is not None:
        return obj.type  # type: ignore[return-value]

    # 3. Fallback to a substring search and accept the first hit.
    hits = search(stripped, limit=1)
    if hits:
        return hits[0].type  # type: ignore[return-value]

    return None


def resolve_and_cache_object_type(context: "PipelineContext") -> ObjectType | None:
    """Resolve the object type from ``context.metadata`` and cache it.

    Looks up ``object_name_hint`` (user-supplied) first, then ``object_name``
    (best-effort plate-solver result).  The result is memoised under
    ``context.metadata["object_type"]`` so the lookup runs at most once per
    pipeline execution even when several steps need it.

    Args:
        context: Shared pipeline context.

    Returns:
        Resolved :data:`ObjectType` or ``None`` when no catalogue match.
    """
    cached = context.metadata.get("object_type")
    if cached is not None:
        return cached  # type: ignore[return-value]
    name = (
        context.metadata.get("object_name_hint")
        or context.metadata.get("object_name")
        or None
    )
    object_type = resolve_object_type(name)
    if object_type is not None:
        context.metadata["object_type"] = object_type
    return object_type


# ── Per-object-type profile overrides ────────────────────────────────────
#
# When the bundled catalogue identifies the target, the pipeline can soften
# parameters that are tuned for emission nebulae (the default in
# ``PRESET_STANDARD``).  Each numeric override is applied **only when the
# new value is strictly less than the profile value**, so a user-customised
# profile that already lowers a setting is never overridden upward.
#
# Galaxies: stretch is reduced to avoid clipping the mid-tones, and the
# GraXpert AI selector is switched from background extraction to the
# ``deconv-both`` chained pass (object → stars deconvolution).  BGE
# sur-soustracts on low-SNR diffuse galaxy targets (verified on M81); the
# deconvolution path instead recovers detail in the disk and tightens the
# field stars without removing diffuse signal.
#
# Nebulae: stretch is capped at 150 — the Quality preset's 180 burns the
# core on bright HII regions like M42 (verified visually, regression
# investigation 2026-04-29).  Cosmic Clarity 2× super-resolution is
# disabled because the model amplifies clipped pixels into reconstruction
# artefacts on saturated cores.  Star-separation weights are widened to
# preserve the wings: the default ``0.8 / 0.5`` recombine crushes the
# faint outer voluptes once renormalised against a saturated core.
ADAPTIVE_PROFILE_OVERRIDES_BY_TYPE: dict[ObjectType, dict[str, Any]] = {
    "galaxy": {
        "stretch_strength": 30.0,
        "denoise_strength": 0.40,
        "sharpen_stellar_amount": 0.20,
        "sharpen_nonstellar_amount": 0.25,
    },
    "nebula": {
        "stretch_strength": 150.0,
    },
    "cluster": {
        "stretch_strength": 50.0,
    },
    "supernova": {
        "stretch_strength": 60.0,
    },
    "planetary": {
        "stretch_strength": 80.0,
    },
}


# ── Per-object-type string-field overrides ───────────────────────────────
#
# String fields cannot be ranked "stricter / looser", so these overrides
# are applied only when the current value is the **default** for that
# field — a user that picked a non-default value is never overridden.
# Each entry maps a field name to ``(default_sentinel, new_value)``.
STRING_OVERRIDES_BY_TYPE: dict[ObjectType, dict[str, tuple[str, str]]] = {
    "galaxy": {
        # Replace BGE with chained object+stars deconvolution.  GraXpert
        # BGE destroys low-SNR galaxy disks; deconvolution instead lifts
        # the diffuse detail and tightens stars without subtracting signal.
        "gradient_removal_ai_model": ("1.0.1", "deconv-both-1.0.1"),
    },
    "cluster": {
        # Clusters benefit from PSF tightening on the dense star field.
        "gradient_removal_ai_model": ("1.0.1", "deconv-both-1.0.1"),
    },
}


# ── Per-object-type GraXpert BGE policy (legacy hard-skip) ───────────────
#
# Kept as an extension point: object types listed here cause the
# gradient_removal step to be skipped entirely (independent of the
# AI-model selector).  Galaxies are no longer in this set — they now
# default to the ``deconv-both`` chained deconvolution via
# :data:`STRING_OVERRIDES_BY_TYPE`.
SKIP_GRADIENT_REMOVAL_TYPES: frozenset[ObjectType] = frozenset()


# ── Per-object-type AI super-resolution policy ───────────────────────────
#
# Cosmic Clarity 2× upsampling is destructive when the input has clipped
# bright regions: the model hallucinates artefacts in place of the
# saturated cores.  Disable on emission nebulae (M42-class burns).
SKIP_SUPER_RESOLUTION_TYPES: frozenset[ObjectType] = frozenset({"nebula"})


# ── Per-object-type star-separation policy ───────────────────────────────
#
# Star separation is meaningful when there is a separable nebula layer:
# galaxies' HII regions get destroyed alongside the stellar layer, and
# clusters *are* the stellar subject.  Skip both.
SKIP_STAR_SEPARATION_TYPES: frozenset[ObjectType] = frozenset({"galaxy", "cluster"})
