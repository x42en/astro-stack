"""Resolve a celestial object name to a high-level catalogue type.

Used by the processing pipeline to lightly adapt parameters (e.g. stretch
strength) when the bundled Messier / NGC catalogue can identify the target.
The resolver intentionally returns ``None`` on any miss so that callers fall
back to the user-supplied profile values without surprises.
"""

from __future__ import annotations

import re
from typing import Literal

from app.infrastructure.catalog.messier import get_by_id, search

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
