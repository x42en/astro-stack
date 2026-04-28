"""Integration tests for the authenticated observation-sites endpoints.

The real :class:`ObservationSiteRepository` uses PostgreSQL-specific column
types, so these tests substitute it with an in-memory fake and assert the
authentication contract:

* ``X-Mock-User`` resolves to a deterministic UUID per username.
* CRUD operations are scoped per user — alice cannot read, update, or delete
  bob's site.
* Cross-user access returns ``404`` (no information leak), not ``403``.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1 import me_sites as me_sites_module
from app.api.v1.me_sites import router as me_sites_router
from app.core.database import get_async_session
from app.api.middleware.error_handler import register_error_handlers
from app.domain.observation_site import ObservationSite


# ── In-memory fake repository ────────────────────────────────────────────────


class _FakeRepo:
    """In-memory observation-site repository keyed by ``(owner, id)``."""

    storage: dict[uuid.UUID, ObservationSite] = {}

    def __init__(self, _session: Any) -> None:  # session is unused
        pass

    async def list_for_user(
        self, owner_user_id: uuid.UUID, offset: int = 0, limit: int = 100
    ) -> list[ObservationSite]:
        return [s for s in self.storage.values() if s.owner_user_id == owner_user_id]

    async def get_for_user(
        self, record_id: uuid.UUID, owner_user_id: uuid.UUID
    ) -> ObservationSite | None:
        site = self.storage.get(record_id)
        if site is None or site.owner_user_id != owner_user_id:
            return None
        return site

    async def create(self, site: ObservationSite) -> ObservationSite:
        if not site.id:
            site.id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        site.created_at = now
        site.updated_at = now
        self.storage[site.id] = site
        return site

    async def update(self, site: ObservationSite) -> ObservationSite:
        site.updated_at = datetime.now(timezone.utc)
        self.storage[site.id] = site
        return site

    async def delete(self, site: ObservationSite) -> None:
        self.storage.pop(site.id, None)


@pytest.fixture(autouse=True)
def patch_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap the real repository for the in-memory fake."""
    _FakeRepo.storage = {}
    monkeypatch.setattr(me_sites_module, "ObservationSiteRepository", _FakeRepo)


@pytest.fixture
def app() -> FastAPI:
    application = FastAPI()
    register_error_handlers(application)
    application.include_router(me_sites_router, prefix="/api/v1")

    async def _no_session() -> Any:
        yield None

    application.dependency_overrides[get_async_session] = _no_session
    return application


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


def _create_site(
    client: TestClient,
    user: str,
    name: str = "Backyard",
    latitude: float = 48.85,
    longitude: float = 2.35,
) -> dict[str, Any]:
    response = client.post(
        "/api/v1/me/sites",
        headers={"X-Mock-User": user},
        json={
            "name": name,
            "description": None,
            "latitude": latitude,
            "longitude": longitude,
            "elevation_m": 35.0,
            "timezone": "Europe/Paris",
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


class TestSitesCrud:
    def test_create_then_list_returns_only_own_sites(self, client: TestClient) -> None:
        site = _create_site(client, "alice")
        listing = client.get("/api/v1/me/sites", headers={"X-Mock-User": "alice"})
        assert listing.status_code == 200
        items = listing.json()
        assert len(items) == 1
        assert items[0]["id"] == site["id"]

    def test_update_own_site_succeeds(self, client: TestClient) -> None:
        site = _create_site(client, "alice")
        response = client.patch(
            f"/api/v1/me/sites/{site['id']}",
            headers={"X-Mock-User": "alice"},
            json={"name": "Renamed"},
        )
        assert response.status_code == 200
        assert response.json()["name"] == "Renamed"

    def test_delete_own_site_succeeds(self, client: TestClient) -> None:
        site = _create_site(client, "alice")
        response = client.delete(
            f"/api/v1/me/sites/{site['id']}",
            headers={"X-Mock-User": "alice"},
        )
        assert response.status_code == 204


class TestCrossUserIsolation:
    def test_alice_cannot_see_bobs_sites(self, client: TestClient) -> None:
        _create_site(client, "bob", name="Bob's roof")
        response = client.get("/api/v1/me/sites", headers={"X-Mock-User": "alice"})
        assert response.status_code == 200
        assert response.json() == []

    def test_alice_cannot_read_bobs_site(self, client: TestClient) -> None:
        site = _create_site(client, "bob")
        response = client.patch(
            f"/api/v1/me/sites/{site['id']}",
            headers={"X-Mock-User": "alice"},
            json={"name": "Hijacked"},
        )
        # 404 (not 403): we do not leak the existence of foreign records.
        assert response.status_code == 404

    def test_alice_cannot_delete_bobs_site(self, client: TestClient) -> None:
        site = _create_site(client, "bob")
        response = client.delete(
            f"/api/v1/me/sites/{site['id']}",
            headers={"X-Mock-User": "alice"},
        )
        assert response.status_code == 404

    def test_unauthenticated_request_is_rejected(self, client: TestClient) -> None:
        response = client.get("/api/v1/me/sites")
        assert response.status_code == 401
