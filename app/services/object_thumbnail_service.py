"""Thumbnail fetching for catalog objects.

Provides a small, async-friendly service that resolves a preview image for
a given deep-sky object. The lookup strategy is a fail-soft cascade:

1. **Wikipedia REST** — best-quality color photo when an article exists.
2. **HiPS2FITS (CDS)** — DSS2 color cutout from RA/Dec; always works for
   any sky position and stays consistent visually.
3. **None** — caller renders a placeholder; the route replies 404.

Results are kept in a process-wide bounded LRU cache so repeated visits to
the planning page do not re-hit the external services. The cache is lost on
restart (intentional: simple, predictable, and the upstream APIs are cheap).
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import OrderedDict
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────


_WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
_HIPS2FITS_URL = "https://alasky.cds.unistra.fr/hips-image-services/hips2fits"
_HIPS_SURVEY = "CDS/P/DSS2/color"
_USER_AGENT = (
    "AstroStack/1.0 (+https://github.com/circle-cyber/astro-stack; "
    "object-thumbnails)"
)

# Hard limits to protect against pathological upstream responses.
_HTTP_TIMEOUT_S = 8.0
_MAX_IMAGE_BYTES = 2 * 1024 * 1024  # 2 MiB
_ALLOWED_CONTENT_PREFIX = "image/"

# Permitted catalog id characters (defence in depth — the registry lookup
# already constrains the lookup, but we never want to forward arbitrary
# user input as path/title to a remote service).
_CATALOG_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _\-+/]{0,31}$")


@dataclass(frozen=True)
class ThumbnailResult:
    """A fetched (or cached) thumbnail."""

    data: bytes
    content_type: str
    source: str  # "wikipedia" | "hips2fits"


def _is_safe_catalog_id(catalog_id: str) -> bool:
    """Return True if ``catalog_id`` is well-formed for outbound queries."""
    return bool(_CATALOG_ID_RE.match(catalog_id))


# ── Service ───────────────────────────────────────────────────────────────────


class ObjectThumbnailService:
    """Fetch and cache thumbnails for deep-sky objects.

    Args:
        http_client: Optional injected client (mainly for tests).
        cache_size: Maximum number of entries kept in memory.
    """

    def __init__(
        self,
        http_client: httpx.AsyncClient | None = None,
        cache_size: int = 512,
    ) -> None:
        self._http: httpx.AsyncClient = http_client or httpx.AsyncClient(
            timeout=_HTTP_TIMEOUT_S,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT},
        )
        self._owns_http = http_client is None
        self._cache_size = max(1, cache_size)
        self._cache: OrderedDict[str, ThumbnailResult | None] = OrderedDict()
        self._locks: dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

    async def aclose(self) -> None:
        """Close the underlying HTTP client when owned."""
        if self._owns_http:
            await self._http.aclose()

    # ── Public API ──────────────────────────────────────────────────────────

    async def fetch(
        self,
        catalog_id: str,
        name: str,
        ra_deg: float,
        dec_deg: float,
    ) -> ThumbnailResult | None:
        """Return a thumbnail or ``None`` when no source could deliver one.

        The result is cached (including misses) until the process restarts.
        Concurrent calls for the same ``catalog_id`` share a single fetch.
        """
        if not _is_safe_catalog_id(catalog_id):
            return None

        key = catalog_id.strip().upper()
        cached = self._cache_get(key)
        if cached is not None:
            return cached[0]  # value (may itself be None)

        lock = await self._acquire_key_lock(key)
        async with lock:
            cached = self._cache_get(key)
            if cached is not None:
                return cached[0]

            result = await self._resolve(catalog_id, name, ra_deg, dec_deg)
            self._cache_put(key, result)
            return result

    # ── Cache helpers ───────────────────────────────────────────────────────

    def _cache_get(
        self, key: str
    ) -> tuple[ThumbnailResult | None] | None:
        if key in self._cache:
            value = self._cache.pop(key)
            self._cache[key] = value  # touch (LRU)
            return (value,)
        return None

    def _cache_put(self, key: str, value: ThumbnailResult | None) -> None:
        self._cache[key] = value
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    async def _acquire_key_lock(self, key: str) -> asyncio.Lock:
        async with self._global_lock:
            lock = self._locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[key] = lock
            return lock

    # ── Resolution cascade ──────────────────────────────────────────────────

    async def _resolve(
        self,
        catalog_id: str,
        name: str,
        ra_deg: float,
        dec_deg: float,
    ) -> ThumbnailResult | None:
        wiki = await self._try_wikipedia(catalog_id, name)
        if wiki is not None:
            return wiki
        hips = await self._try_hips2fits(ra_deg, dec_deg)
        if hips is not None:
            return hips
        return None

    async def _try_wikipedia(
        self, catalog_id: str, name: str
    ) -> ThumbnailResult | None:
        # Try identifier first (often a redirect to the named article),
        # then the common name, then a disambiguated form.
        candidates: list[str] = []
        cid = catalog_id.strip()
        nm = name.strip()
        if cid:
            candidates.append(cid)
        if nm and nm.lower() != cid.lower():
            candidates.append(nm)
            if "(" not in nm:
                candidates.append(f"{nm} (astronomy)")

        for title in candidates:
            try:
                resp = await self._http.get(
                    _WIKI_SUMMARY_URL.format(title=_wiki_quote(title)),
                    headers={"Accept": "application/json"},
                )
            except httpx.HTTPError as exc:
                logger.debug("wikipedia summary failed for %r: %s", title, exc)
                continue
            if resp.status_code != 200:
                continue
            try:
                payload = resp.json()
            except ValueError:
                continue
            # Skip disambiguation pages.
            if payload.get("type") == "disambiguation":
                continue
            thumb = (payload.get("originalimage") or payload.get("thumbnail") or {}).get(
                "source"
            )
            if not thumb:
                continue
            image = await self._download_image(thumb)
            if image is not None:
                data, content_type = image
                return ThumbnailResult(
                    data=data, content_type=content_type, source="wikipedia"
                )
        return None

    async def _try_hips2fits(
        self, ra_deg: float, dec_deg: float
    ) -> ThumbnailResult | None:
        if not (-90.0 <= dec_deg <= 90.0) or not (0.0 <= ra_deg <= 360.0):
            return None
        params = {
            "hips": _HIPS_SURVEY,
            "width": 512,
            "height": 288,  # 16:9
            "fov": 0.5,  # degrees; reasonable for most DSOs
            "projection": "TAN",
            "coordsys": "icrs",
            "ra": f"{ra_deg:.6f}",
            "dec": f"{dec_deg:.6f}",
            "format": "jpg",
        }
        try:
            resp = await self._http.get(_HIPS2FITS_URL, params=params)
        except httpx.HTTPError as exc:
            logger.debug("hips2fits failed at (%.3f, %.3f): %s", ra_deg, dec_deg, exc)
            return None
        if resp.status_code != 200:
            return None
        content_type = resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
        if not content_type.startswith(_ALLOWED_CONTENT_PREFIX):
            return None
        data = resp.content
        if not data or len(data) > _MAX_IMAGE_BYTES:
            return None
        return ThumbnailResult(data=data, content_type=content_type, source="hips2fits")

    async def _download_image(self, url: str) -> tuple[bytes, str] | None:
        try:
            resp = await self._http.get(url)
        except httpx.HTTPError as exc:
            logger.debug("image download failed for %s: %s", url, exc)
            return None
        if resp.status_code != 200:
            return None
        content_type = resp.headers.get("content-type", "").split(";")[0].strip()
        if not content_type.startswith(_ALLOWED_CONTENT_PREFIX):
            return None
        data = resp.content
        if not data or len(data) > _MAX_IMAGE_BYTES:
            return None
        return data, content_type


def _wiki_quote(title: str) -> str:
    """Percent-encode a Wikipedia article title for the REST URL.

    Spaces become underscores (Wikipedia convention), then every other
    reserved character is percent-encoded.
    """
    from urllib.parse import quote

    return quote(title.strip().replace(" ", "_"), safe="")
