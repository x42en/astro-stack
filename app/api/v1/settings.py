"""REST API endpoints for global application settings.

Endpoints
---------
* ``GET  /settings``  — read the current settings (public, no auth required)
* ``PUT  /settings``  — update settings (admin role required)
* ``GET  /settings/llm`` — LLM provider overview (public, keys masked)
* ``GET  /settings/llm/models`` — list models advertised by a provider
* ``POST /settings/llm/test`` — connectivity check for a provider (admin)

The settings object is a singleton row in the database.  Only administrators
(``require_role("admin")``) may write; any authenticated or anonymous client
may read so the frontend can display operational defaults before the user logs
in.
"""

from __future__ import annotations

from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.middleware.auth import get_current_user, require_role
from app.core.database import get_async_session
from app.domain.app_settings import (
    AppSettingsRead,
    AppSettingsUpdate,
    LlmModelEntry,
    LlmSettingsRead,
)
from app.services.app_settings_service import (
    get_app_settings,
    get_llm_settings,
    upsert_app_settings,
)

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("", response_model=AppSettingsRead, summary="Get global settings")
async def read_settings(
    db: AsyncSession = Depends(get_async_session),
) -> AppSettingsRead:
    """Return the current global operational settings.

    No authentication required — the frontend needs these before the user
    has signed in in order to display correct defaults.
    """
    row = await get_app_settings(db)
    return AppSettingsRead.model_validate(row, from_attributes=True)


@router.put(
    "",
    response_model=AppSettingsRead,
    summary="Update global settings (admin only)",
    dependencies=[Depends(require_role("admin"))],
)
async def update_settings(
    body: AppSettingsUpdate,
    db: AsyncSession = Depends(get_async_session),
    # get_current_user returns the raw JWT claims dict (or None in disabled mode).
    claims: Optional[dict] = Depends(get_current_user),
) -> AppSettingsRead:
    """Partially update global settings.

    Only users with the ``admin`` role may call this endpoint.  In
    ``disabled`` auth mode the role check is bypassed and ``user_id`` will
    be ``None`` in the audit column.
    """
    user_id: Optional[str] = claims.get("sub") if claims else None
    body = _validate_llm_update(body)
    row = await upsert_app_settings(db, body, user_id)
    return AppSettingsRead.model_validate(row, from_attributes=True)


def _validate_llm_update(body: AppSettingsUpdate) -> AppSettingsUpdate:
    """Validate the LLM provider field of a settings update.

    Args:
        body: Incoming partial update.

    Returns:
        The unchanged body.

    Raises:
        HTTPException: If ``llm_active_provider`` is not a known provider.
    """
    if body.llm_active_provider not in (None, "ollama", "vllm", "kilo"):
        raise HTTPException(
            status_code=422,
            detail="llm_active_provider must be one of: ollama, vllm, kilo.",
        )
    return body


def _model_supports_vision(item: dict) -> Optional[bool]:
    """Best-effort vision-capability flag from a Kilo ``/models`` entry."""
    arch = item.get("architecture")
    if isinstance(arch, dict):
        modalities = arch.get("input_modalities")
        if isinstance(modalities, list):
            return "image" in modalities
    return None


@router.get("/llm", response_model=LlmSettingsRead, summary="Get LLM provider overview")
async def read_llm_settings(
    db: AsyncSession = Depends(get_async_session),
) -> LlmSettingsRead:
    """Return the LLM provider overview (active provider + per-profile model).

    No authentication required; API keys are never exposed (``has_api_key``
    flag only).
    """
    row = await get_app_settings(db)
    return get_llm_settings(row)


@router.get(
    "/llm/models",
    response_model=list[LlmModelEntry],
    summary="List models advertised by an LLM provider",
)
async def list_llm_models(
    provider: str = Query(default="kilo", description="ollama | vllm | kilo"),
    db: AsyncSession = Depends(get_async_session),
) -> list[LlmModelEntry]:
    """Proxy the provider's ``/models`` endpoint (Ollama/vLLM ``/v1/models``).

    Args:
        provider: Provider key.
        db: Injected database session.

    Returns:
        Advertised model entries (best effort — empty list when unreachable).
    """
    from app.llm.factory import normalize_provider, resolve_llm_profile  # noqa: PLC0415

    row = await get_app_settings(db)
    name = normalize_provider(provider)
    if name == "custom":
        raise HTTPException(status_code=400, detail="Provider 'custom' has no model catalogue.")
    profile = resolve_llm_profile(name, app_settings=row)
    url = f"{profile.base_url.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {profile.api_key}"} if profile.api_key else {}
    try:
        async with httpx.AsyncClient(timeout=15.0) as http:
            response = await http.get(url, headers=headers)
            response.raise_for_status()
            body = response.json()
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502, detail=f"Provider '{name}' unreachable: {exc}"
        ) from exc
    entries: list[LlmModelEntry] = []
    data = body.get("data") if isinstance(body, dict) else None
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict) or not item.get("id"):
                continue
            item_id = str(item["id"])
            entries.append(
                LlmModelEntry(
                    id=item_id,
                    name=str(item.get("name") or item_id),
                    free=bool(item.get("isFree", False)) or item_id.endswith(":free"),
                    vision=_model_supports_vision(item),
                )
            )
    elif isinstance(body, dict) and isinstance(body.get("models"), list):
        for item in body["models"]:
            if isinstance(item, dict) and item.get("name"):
                model_id = str(item["name"])
                entries.append(LlmModelEntry(id=model_id, name=model_id))
            elif isinstance(item, str):
                entries.append(LlmModelEntry(id=item, name=item))
    return entries


@router.post(
    "/llm/test",
    summary="Test LLM provider connectivity (admin only)",
    dependencies=[Depends(require_role("admin"))],
)
async def test_llm_provider(
    provider: str = Query(default="kilo", description="ollama | vllm | kilo"),
    model: Optional[str] = Query(default=None),
    db: AsyncSession = Depends(get_async_session),
) -> dict:
    """Ping a provider's ``/models`` (or ``/chat/completions`` fallback).

    Args:
        provider: Provider key.
        model: Optional model override for the chat fallback probe.
        db: Injected database session.

    Returns:
        Dict with ``ok``, ``provider``, ``model`` and ``latency_ms``.
    """
    import time  # noqa: PLC0415

    from app.llm.factory import normalize_provider, resolve_llm_profile  # noqa: PLC0415

    row = await get_app_settings(db)
    name = normalize_provider(provider)
    profile = resolve_llm_profile(name, model, app_settings=row)
    headers = {"Authorization": f"Bearer {profile.api_key}"} if profile.api_key else {}
    started = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=15.0) as http:
            probe = await http.get(f"{profile.base_url.rstrip('/')}/models", headers=headers)
            probe.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502, detail=f"Provider '{name}' unreachable: {exc}"
        ) from exc
    latency_ms = round((time.monotonic() - started) * 1000, 1)
    return {"ok": True, "provider": name, "model": profile.model, "latency_ms": latency_ms}
