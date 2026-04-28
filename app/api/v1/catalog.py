"""REST endpoints for deep-sky object catalogue lookup and resolution.

Two complementary endpoints back the front-end target picker:

* ``GET /catalog/objects`` — search the bundled offline table (Messier +
  popular NGC/IC). Always available, zero network access.
* ``GET /catalog/resolve`` — best-effort name resolution via SIMBAD using
  ``astroquery``. Used as a fallback when the user types an object that is
  not in the bundled catalogue (e.g. obscure NGC, IC, or proper names).
  Returns 404 gracefully when SIMBAD is unreachable or the name is unknown.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.api.middleware.auth import get_current_user
from app.core.logging import get_logger
from app.infrastructure.catalog.messier import CatalogObject
from app.infrastructure.catalog.registry import (
    lookup_object as get_by_id,
    search_all as search,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/catalog", tags=["catalog"])


class CatalogObjectResponse(BaseModel):
    """API representation of a catalogue entry."""

    id: str
    name: str
    type: str
    constellation: str
    ra_deg: float
    dec_deg: float
    magnitude: Optional[float] = None
    source: str = "bundled"

    @classmethod
    def from_domain(cls, obj: CatalogObject, source: str = "bundled") -> "CatalogObjectResponse":
        return cls(
            id=obj.id,
            name=obj.name,
            type=obj.type,
            constellation=obj.constellation,
            ra_deg=obj.ra_deg,
            dec_deg=obj.dec_deg,
            magnitude=obj.magnitude,
            source=source,
        )


class CatalogObjectListResponse(BaseModel):
    """Paginated-style envelope for catalogue search results."""

    items: list[CatalogObjectResponse]
    total: int


@router.get(
    "/objects",
    response_model=CatalogObjectListResponse,
    summary="Search the offline deep-sky object catalogue",
    description=(
        "Returns matching entries from the bundled Messier + NGC/IC table. "
        "An empty query returns the full catalogue (capped by ``limit``)."
    ),
)
async def list_objects(
    q: str = Query(default="", description="Free-text search across id and name"),
    limit: int = Query(default=200, ge=1, le=500),
    _user: Optional[dict] = Depends(get_current_user),
) -> CatalogObjectListResponse:
    """Search the static catalogue.

    Args:
        q: Free-text query (e.g. ``"M42"``, ``"orion"``, ``"NGC"``).
        limit: Maximum number of results to return.
        _user: Injected auth user (unused but enforces auth when enabled).

    Returns:
        :class:`CatalogObjectListResponse` with matching entries.
    """
    matches = search(q, limit=limit)
    return CatalogObjectListResponse(
        items=[CatalogObjectResponse.from_domain(m) for m in matches],
        total=len(matches),
    )


@router.get(
    "/resolve",
    response_model=CatalogObjectResponse,
    summary="Resolve an object name to coordinates (SIMBAD fallback)",
    description=(
        "Looks up ``name`` first in the bundled catalogue, then falls back to "
        "SIMBAD via ``astroquery``. Returns 404 when the object cannot be "
        "resolved or SIMBAD is unreachable."
    ),
)
async def resolve_object(
    name: str = Query(..., min_length=1, description="Object identifier or common name"),
    _user: Optional[dict] = Depends(get_current_user),
) -> CatalogObjectResponse:
    """Resolve an arbitrary object name to J2000 coordinates.

    Tries the bundled catalogue first (instant, offline). On miss, attempts
    a SIMBAD query if ``astroquery`` is installed and the network is reachable.

    Args:
        name: Object identifier or common name (e.g. ``"M42"``,
            ``"NGC 7000"``, ``"Andromeda"``).
        _user: Injected auth user.

    Returns:
        :class:`CatalogObjectResponse` describing the resolved object.

    Raises:
        HTTPException: 404 when the object cannot be resolved.
    """
    # 1. Bundled catalogue (instant)
    bundled = get_by_id(name)
    if bundled is not None:
        return CatalogObjectResponse.from_domain(bundled)

    # 2. SIMBAD fallback — best effort, never fails the request loudly
    try:
        from astroquery.simbad import Simbad  # type: ignore[import-not-found]  # noqa: PLC0415

        simbad = Simbad()
        simbad.add_votable_fields("flux(V)", "otype")
        result = simbad.query_object(name)
    except ImportError:
        logger.info("simbad_unavailable", reason="astroquery not installed")
        result = None
    except Exception as exc:  # noqa: BLE001 — network or remote failure
        logger.info("simbad_query_failed", name=name, error=str(exc))
        result = None

    if result is None or len(result) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Object '{name}' not found in bundled catalogue or SIMBAD.",
        )

    # SIMBAD returns RA/Dec as sexagesimal strings; parse via astropy
    try:
        from astropy.coordinates import SkyCoord  # noqa: PLC0415
        import astropy.units as u  # noqa: PLC0415

        row = result[0]
        ra_str = str(row["RA"])
        dec_str = str(row["DEC"])
        coord = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg), frame="icrs")
        ra_deg = float(coord.ra.deg)
        dec_deg = float(coord.dec.deg)
        otype = str(row["OTYPE"]) if "OTYPE" in row.colnames else ""
        try:
            magnitude: Optional[float] = float(row["FLUX_V"])
        except (KeyError, TypeError, ValueError):
            magnitude = None
    except Exception as exc:  # noqa: BLE001
        logger.warning("simbad_parse_failed", name=name, error=str(exc))
        raise HTTPException(
            status_code=404, detail=f"SIMBAD result for '{name}' could not be parsed."
        ) from exc

    return CatalogObjectResponse(
        id=name.upper().replace(" ", ""),
        name=name,
        type=_simbad_otype_to_kind(otype),
        constellation="",
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        magnitude=magnitude,
        source="simbad",
    )


def _simbad_otype_to_kind(otype: str) -> str:
    """Map a SIMBAD OTYPE code to our short ``CatalogObject.type`` vocabulary.

    Args:
        otype: SIMBAD object type string (e.g. ``"GlC"``, ``"PN"``, ``"G"``).

    Returns:
        One of ``galaxy``, ``cluster``, ``nebula``, ``planetary``,
        ``supernova`` or ``other``.
    """
    o = (otype or "").lower()
    if "gxy" in o or o == "g" or "galaxy" in o:
        return "galaxy"
    if "glc" in o or "opc" in o or "cluster" in o:
        return "cluster"
    if "pn" in o or "planetary" in o:
        return "planetary"
    if "snr" in o or "supernova" in o:
        return "supernova"
    if "neb" in o or "hii" in o:
        return "nebula"
    return "other"
