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
# ``PRESET_STANDARD``).  Each override is applied **only when the new value
# is strictly less than the profile value**, so a user-customised profile
# that already lowers a setting is never overridden upward.
#
# Galaxies: GraXpert is disabled because both AI and polynomial modes
# sur-soustract on low-SNR diffuse targets (verified on M81).  The stretch
# is also reduced to avoid clipping the mid-tones.
ADAPTIVE_PROFILE_OVERRIDES_BY_TYPE: dict[ObjectType, dict[str, Any]] = {
    "galaxy": {
        "stretch_strength": 30.0,
        "denoise_strength": 0.40,
        "sharpen_stellar_amount": 0.20,
        "sharpen_nonstellar_amount": 0.25,
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


# ── Per-object-type GraXpert policy ──────────────────────────────────────
#
# Names listed here will skip the gradient_removal step entirely.  Tested
# manually: on M81 both AI and polynomial modes produce all-NaN FITS.
SKIP_GRADIENT_REMOVAL_TYPES: frozenset[ObjectType] = frozenset({"galaxy"})
