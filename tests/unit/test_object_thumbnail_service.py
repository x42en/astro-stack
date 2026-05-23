"""Unit tests for :mod:`app.services.object_thumbnail_service`."""

from __future__ import annotations

import httpx
import pytest

from app.services.object_thumbnail_service import (
    ObjectThumbnailService,
    _is_safe_catalog_id,
)

# ── Static helpers ────────────────────────────────────────────────────────────


class TestSanitization:
    @pytest.mark.parametrize(
        "value",
        ["M42", "NGC7000", "IC1318", "Mel-25", "M 31", "Caldwell+14"],
    )
    def test_accepts_well_formed_ids(self, value: str) -> None:
        assert _is_safe_catalog_id(value)

    @pytest.mark.parametrize(
        "value",
        [
            "",
            " M42",
            "../etc/passwd",
            "<script>",
            "M42;DROP",
            "x" * 64,
            "%2e%2e",
        ],
    )
    def test_rejects_malformed_ids(self, value: str) -> None:
        assert not _is_safe_catalog_id(value)


# ── Fetch fixtures ────────────────────────────────────────────────────────────


_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
_JPG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 64


def _wiki_summary(thumb_url: str | None = "https://upload.wikimedia.org/m42.jpg") -> dict:
    payload: dict = {"type": "standard", "title": "Orion Nebula"}
    if thumb_url is not None:
        payload["originalimage"] = {"source": thumb_url, "width": 800, "height": 600}
    return payload


def _make_service(handler) -> ObjectThumbnailService:
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport, follow_redirects=True)
    return ObjectThumbnailService(http_client=http)


@pytest.mark.asyncio
class TestFetch:
    async def test_returns_wikipedia_thumbnail(self) -> None:
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            host = request.url.host
            if host == "en.wikipedia.org":
                return httpx.Response(200, json=_wiki_summary())
            if host == "upload.wikimedia.org":
                return httpx.Response(
                    200, content=_JPG_BYTES, headers={"content-type": "image/jpeg"}
                )
            return httpx.Response(404)

        svc = _make_service(handler)
        try:
            result = await svc.fetch("M42", "Orion Nebula", 83.822, -5.391)
            assert result is not None
            assert result.source == "wikipedia"
            assert result.data == _JPG_BYTES
            assert result.content_type == "image/jpeg"
            # First request hits Wikipedia summary endpoint.
            assert "en.wikipedia.org" in calls[0]
        finally:
            await svc.aclose()

    async def test_falls_back_to_hips2fits_when_wikipedia_misses(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            host = request.url.host
            if host == "en.wikipedia.org":
                return httpx.Response(404)
            if host == "alasky.cds.unistra.fr":
                return httpx.Response(
                    200, content=_JPG_BYTES, headers={"content-type": "image/jpeg"}
                )
            return httpx.Response(500)

        svc = _make_service(handler)
        try:
            result = await svc.fetch("NGC7000", "North America Nebula", 314.75, 44.31)
            assert result is not None
            assert result.source == "hips2fits"
            assert result.content_type == "image/jpeg"
        finally:
            await svc.aclose()

    async def test_returns_none_when_all_sources_fail(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500)

        svc = _make_service(handler)
        try:
            result = await svc.fetch("XYZ1", "Mystery", 10.0, 10.0)
            assert result is None
        finally:
            await svc.aclose()

    async def test_skips_disambiguation_pages(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            host = request.url.host
            if host == "en.wikipedia.org":
                return httpx.Response(
                    200, json={"type": "disambiguation", "title": "M1"}
                )
            if host == "alasky.cds.unistra.fr":
                return httpx.Response(
                    200, content=_PNG_BYTES, headers={"content-type": "image/png"}
                )
            return httpx.Response(404)

        svc = _make_service(handler)
        try:
            result = await svc.fetch("M1", "Crab Nebula", 83.633, 22.014)
            assert result is not None
            assert result.source == "hips2fits"
        finally:
            await svc.aclose()

    async def test_rejects_oversize_image(self) -> None:
        big = b"\xff\xd8" + b"\x00" * (3 * 1024 * 1024)  # > 2 MiB

        def handler(request: httpx.Request) -> httpx.Response:
            host = request.url.host
            if host == "en.wikipedia.org":
                return httpx.Response(200, json=_wiki_summary())
            if host == "upload.wikimedia.org":
                return httpx.Response(
                    200, content=big, headers={"content-type": "image/jpeg"}
                )
            if host == "alasky.cds.unistra.fr":
                return httpx.Response(
                    200, content=big, headers={"content-type": "image/jpeg"}
                )
            return httpx.Response(404)

        svc = _make_service(handler)
        try:
            result = await svc.fetch("M42", "Orion Nebula", 83.8, -5.4)
            assert result is None
        finally:
            await svc.aclose()

    async def test_rejects_non_image_content_type(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            host = request.url.host
            if host == "en.wikipedia.org":
                return httpx.Response(200, json=_wiki_summary())
            if host == "upload.wikimedia.org":
                return httpx.Response(
                    200, content=b"<html/>", headers={"content-type": "text/html"}
                )
            if host == "alasky.cds.unistra.fr":
                return httpx.Response(
                    200, content=b"<html/>", headers={"content-type": "text/html"}
                )
            return httpx.Response(404)

        svc = _make_service(handler)
        try:
            result = await svc.fetch("M42", "Orion Nebula", 83.8, -5.4)
            assert result is None
        finally:
            await svc.aclose()

    async def test_cache_hit_skips_network(self) -> None:
        counter = {"calls": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            counter["calls"] += 1
            host = request.url.host
            if host == "en.wikipedia.org":
                return httpx.Response(200, json=_wiki_summary())
            if host == "upload.wikimedia.org":
                return httpx.Response(
                    200, content=_JPG_BYTES, headers={"content-type": "image/jpeg"}
                )
            return httpx.Response(404)

        svc = _make_service(handler)
        try:
            r1 = await svc.fetch("M42", "Orion Nebula", 83.8, -5.4)
            calls_after_first = counter["calls"]
            r2 = await svc.fetch("M42", "Orion Nebula", 83.8, -5.4)
            assert r1 is not None and r2 is not None
            assert r2.data == r1.data
            # No additional HTTP calls on cache hit.
            assert counter["calls"] == calls_after_first
        finally:
            await svc.aclose()

    async def test_cache_remembers_misses(self) -> None:
        counter = {"calls": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            counter["calls"] += 1
            return httpx.Response(500)

        svc = _make_service(handler)
        try:
            r1 = await svc.fetch("ZZ9", "Mystery", 10.0, 10.0)
            after_first = counter["calls"]
            r2 = await svc.fetch("ZZ9", "Mystery", 10.0, 10.0)
            assert r1 is None
            assert r2 is None
            assert counter["calls"] == after_first
        finally:
            await svc.aclose()

    async def test_rejects_unsafe_catalog_id_without_network(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            pytest.fail("Network must not be hit for unsafe ids")

        svc = _make_service(handler)
        try:
            assert await svc.fetch("../etc/passwd", "x", 0, 0) is None
        finally:
            await svc.aclose()
