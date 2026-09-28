"""Resolution of the active LLM profile from env, DB settings and overrides.

Precedence (highest first):

1. Explicit per-call override (job / session / profile).
2. Database ``AppSettings`` singleton (operator-configured default).
3. Environment variables (``LLM_*`` with ``VLLM_*`` legacy fallback).

Secrets (``KILO_API_KEY``, ``VLLM_API_KEY``) are only ever read from the
environment — they are never stored in the database nor exposed via the API.
"""

from __future__ import annotations

from app.core.config import get_settings
from app.llm.types import PROVIDER_NAMES, LLMProfile

KILO_DEFAULT_BASE_URL = "https://api.kilo.ai/api/gateway"
KILO_DEFAULT_MODEL = "qwen/qwen3.8-27b:free"

_VALID_PROVIDERS = frozenset({"ollama", "vllm", "kilo", "custom"})


def list_provider_names() -> tuple[str, ...]:
    """Return the user-facing provider names (``ollama``, ``vllm``, ``kilo``)."""
    return PROVIDER_NAMES


def normalize_provider(value: str | None) -> str:
    """Normalise a provider key, falling back to ``vllm`` when unknown.

    Args:
        value: Raw provider key (may be ``None`` or ``"default"``).

    Returns:
        A valid provider key.
    """
    if not value or value == "default":
        return "vllm"
    lowered = value.strip().lower()
    return lowered if lowered in _VALID_PROVIDERS else "vllm"


def _ollama_base_url(raw: str) -> str:
    """Return the OpenAI-compatible base URL for Ollama (ensures ``/v1``)."""
    stripped = (raw or "").rstrip("/")
    if stripped.endswith("/v1"):
        return stripped
    return f"{stripped}/v1"


def resolve_llm_profile(
    provider: str | None = None,
    model: str | None = None,
    *,
    app_settings: object | None = None,
) -> LLMProfile:
    """Resolve the effective LLM profile.

    Args:
        provider: Explicit override (``None``/``"default"`` = use the
            configured active provider).
        model: Explicit model override (``None`` = provider default).
        app_settings: Optional ``AppSettings`` row; when omitted the active
            provider and per-provider URLs/models fall back to env vars.

    Returns:
        The resolved :class:`LLMProfile`.
    """
    settings = get_settings()
    active = normalize_provider(provider)
    if provider in (None, "default"):
        db_active = getattr(app_settings, "llm_active_provider", None) if app_settings else None
        active = normalize_provider(db_active or settings.llm_active_provider)

    if active == "ollama":
        base_url = _ollama_base_url(_db_or(app_settings, "llm_ollama_url", settings.ollama_url))
        default_model = _db_or(app_settings, "llm_ollama_model", settings.ollama_model)
        api_key = ""
        timeout = settings.ollama_timeout_seconds
    elif active == "kilo":
        base_url = settings.kilo_base_url
        default_model = _db_or(app_settings, "llm_kilo_model", settings.kilo_model)
        api_key = settings.kilo_api_key
        timeout = settings.kilo_timeout_seconds
    elif active == "custom":
        base_url = settings.llm_base_url or settings.vllm_base_url
        default_model = settings.llm_model or settings.vllm_model
        api_key = settings.vllm_api_key
        timeout = settings.llm_timeout_seconds or settings.vllm_timeout_seconds
    else:  # vllm
        active = "vllm"
        base_url = _db_or(app_settings, "llm_vllm_base_url", settings.vllm_base_url)
        default_model = _db_or(app_settings, "llm_vllm_model", settings.vllm_model)
        api_key = settings.vllm_api_key
        timeout = settings.vllm_timeout_seconds

    return LLMProfile(
        provider=active,
        base_url=base_url,
        model=model or default_model,
        api_key=api_key,
        timeout_seconds=timeout,
    )


def build_critic_kwargs(
    provider: str | None = None,
    model: str | None = None,
    *,
    app_settings: object | None = None,
) -> dict[str, object]:
    """Build ``VisionCritic`` kwargs from an optional provider/model override.

    Args:
        provider: ``None``/``"default"`` = configured active provider.
        model: Optional model override.
        app_settings: Optional ``AppSettings`` row for DB-configured defaults.

    Returns:
        Dict with ``provider``, ``base_url``, ``model``, ``api_key`` and
        ``timeout`` keys suitable for :class:`VisionCritic`.
    """
    profile = resolve_llm_profile(provider, model, app_settings=app_settings)
    return {
        "provider": profile.provider,
        "base_url": profile.base_url,
        "model": profile.model,
        "api_key": profile.api_key,
        "timeout": profile.timeout_seconds,
    }


def _db_or(app_settings: object | None, field: str, fallback: str) -> str:
    """Return the DB field value when set, else the env fallback."""
    if app_settings is not None:
        value = getattr(app_settings, field, None)
        if value:
            return str(value)
    return fallback


def validate_override(provider: str | None, model: str | None) -> tuple[str | None, str | None]:
    """Validate per-job LLM override query params.

    Args:
        provider: Raw ``llm_provider`` query value.
        model: Raw ``llm_model`` query value.

    Returns:
        Normalised ``(provider, model)`` where ``"default"`` is collapsed to
        ``None`` (meaning: use the configured active provider).

    Raises:
        ValueError: If the provider key is unknown.
    """
    norm_provider: str | None = None
    if provider not in (None, "", "default"):
        lowered = provider.strip().lower()
        if lowered not in _VALID_PROVIDERS:
            raise ValueError(
                f"Unknown LLM provider '{provider}'. "
                "Expected one of: default, ollama, vllm, kilo, custom."
            )
        norm_provider = lowered
    norm_model = model.strip() if model and model.strip() else None
    return norm_provider, norm_model
