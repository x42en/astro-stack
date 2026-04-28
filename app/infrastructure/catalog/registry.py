"""Unified deep-sky catalog registry.

Aggregates the bundled Messier and OpenNGC catalogues into a single,
deduplicated lookup pool. Messier wins on identifier conflicts because its
coordinates are higher-quality for the M-numbered objects.
"""

from __future__ import annotations

from functools import lru_cache

from app.infrastructure.catalog.messier import CatalogObject
from app.infrastructure.catalog.messier import all_objects as _messier_all
from app.infrastructure.catalog.openngc import all_openngc
from app.infrastructure.catalog.popular import POPULAR, PopularEntry


def _normalize(catalog_id: str) -> str:
    """Normalize a catalog identifier (uppercase, strip spaces)."""
    return catalog_id.strip().upper().replace(" ", "")


@lru_cache(maxsize=1)
def _build_deduped() -> tuple[CatalogObject, ...]:
    """Build the deduplicated catalog tuple, sorted by identifier."""
    seen: dict[str, CatalogObject] = {}
    for obj in _messier_all():
        seen[_normalize(obj.id)] = obj
    for obj in all_openngc():
        key = _normalize(obj.id)
        if key not in seen:
            seen[key] = obj
    return tuple(sorted(seen.values(), key=lambda o: (o.id.upper(), o.name)))


@lru_cache(maxsize=1)
def _build_id_index() -> dict[str, CatalogObject]:
    """Build a normalised id → object lookup table."""
    return {_normalize(obj.id): obj for obj in _build_deduped()}


def lookup_object(catalog_id: str) -> CatalogObject | None:
    """Look up a catalog object by id, case- and space-insensitive."""
    return _build_id_index().get(_normalize(catalog_id))


def search_all(query: str, limit: int = 200) -> list[CatalogObject]:
    """Search the unified catalog by id, name and constellation."""
    q = query.strip().lower().replace(" ", "")
    if not q:
        return list(_build_deduped()[:limit])

    exact: list[CatalogObject] = []
    id_prefix: list[CatalogObject] = []
    name_prefix: list[CatalogObject] = []
    other: list[CatalogObject] = []

    for obj in _build_deduped():
        oid = obj.id.lower().replace(" ", "")
        name = obj.name.lower()
        name_nospace = name.replace(" ", "")
        const = obj.constellation.lower()

        if oid == q:
            exact.append(obj)
        elif oid.startswith(q):
            id_prefix.append(obj)
        elif name.startswith(q) or name_nospace.startswith(q):
            name_prefix.append(obj)
        elif q in oid or q in name_nospace or q == const:
            other.append(obj)

    return (exact + id_prefix + name_prefix + other)[:limit]


def all_objects() -> tuple[CatalogObject, ...]:
    """Return the deduplicated full catalog (Messier + OpenNGC)."""
    return _build_deduped()


def popular_entries() -> tuple[PopularEntry, ...]:
    """Return the curated popular-targets list."""
    return POPULAR
