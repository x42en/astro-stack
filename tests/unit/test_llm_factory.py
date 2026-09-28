"""Unit tests for :mod:`app.llm` (provider-agnostic LLM factory and client)."""

from __future__ import annotations

import json

import httpx
import pytest

from app.llm.client import LLMClient
from app.llm.factory import (
    KILO_DEFAULT_MODEL,
    normalize_provider,
    resolve_llm_profile,
    validate_override,
)
from app.llm.types import LLMProfile


class TestNormalizeProvider:
    def test_none_and_default_fall_back_to_vllm(self) -> None:
        assert normalize_provider(None) == "vllm"
        assert normalize_provider("default") == "vllm"

    def test_known_providers_pass_through(self) -> None:
        assert normalize_provider("ollama") == "ollama"
        assert normalize_provider("KILO") == "kilo"
        assert normalize_provider("custom") == "custom"

    def test_unknown_provider_falls_back_to_vllm(self) -> None:
        assert normalize_provider("nope") == "vllm"


class TestValidateOverride:
    def test_default_collapses_to_none(self) -> None:
        assert validate_override(None, None) == (None, None)
        assert validate_override("default", "") == (None, None)

    def test_valid_override_is_normalised(self) -> None:
        assert validate_override("Kilo", "qwen/qwen3.8-27b:free") == (
            "kilo",
            "qwen/qwen3.8-27b:free",
        )

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            validate_override("nope", None)


class TestResolveProfile:
    def test_default_is_vllm_from_env(self) -> None:
        profile = resolve_llm_profile()
        assert profile.provider == "vllm"
        assert profile.base_url.endswith("/v1")

    def test_ollama_appends_v1(self) -> None:
        profile = resolve_llm_profile("ollama")
        assert profile.provider == "ollama"
        assert profile.base_url.endswith("/v1")
        assert profile.api_key == ""

    def test_kilo_defaults(self) -> None:
        profile = resolve_llm_profile("kilo")
        assert profile.provider == "kilo"
        assert profile.base_url == "https://api.kilo.ai/api/gateway"
        assert profile.model == KILO_DEFAULT_MODEL

    def test_explicit_model_override_wins(self) -> None:
        profile = resolve_llm_profile("kilo", model="openai/gpt-5.4-mini")
        assert profile.model == "openai/gpt-5.4-mini"

    def test_db_settings_override_env(self) -> None:
        from types import SimpleNamespace

        row = SimpleNamespace(
            llm_active_provider="kilo",
            llm_ollama_url="",
            llm_ollama_model="",
            llm_vllm_base_url="",
            llm_vllm_model="",
            llm_kilo_model="anthropic/claude-sonnet-4.6",
        )
        profile = resolve_llm_profile(app_settings=row)
        assert profile.provider == "kilo"
        assert profile.model == "anthropic/claude-sonnet-4.6"


class TestLLMClient:
    @pytest.mark.asyncio
    async def test_posts_to_chat_completions_with_bearer(self) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["auth"] = request.headers.get("authorization")
            body = json.loads(request.content)
            captured["model"] = body["model"]
            return httpx.Response(200, json={"choices": [{"message": {"content": "hello"}}]})

        transport = httpx.MockTransport(handler)
        profile = LLMProfile(
            provider="kilo",
            base_url="https://api.kilo.ai/api/gateway",
            model="qwen/qwen3.8-27b:free",
            api_key="secret",
        )
        async with httpx.AsyncClient(transport=transport) as http:
            client = LLMClient(profile, http_client=http)
            body = await client.chat_completions([{"role": "user", "content": "hi"}])

        assert captured["url"].endswith("/chat/completions")
        assert captured["auth"] == "Bearer secret"
        assert captured["model"] == "qwen/qwen3.8-27b:free"
        assert LLMClient.extract_content(body) == "hello"

    @pytest.mark.asyncio
    async def test_no_auth_header_when_key_empty(self) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["auth"] = request.headers.get("authorization")
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        transport = httpx.MockTransport(handler)
        profile = LLMProfile(
            provider="ollama", base_url="http://ollama:11434/v1", model="qwen3-vl:8b"
        )
        async with httpx.AsyncClient(transport=transport) as http:
            client = LLMClient(profile, http_client=http)
            await client.chat_completions([{"role": "user", "content": "hi"}])

        assert captured["auth"] is None

    def test_extract_content_returns_none_on_bad_shape(self) -> None:
        assert LLMClient.extract_content({}) is None
        assert LLMClient.extract_content({"choices": []}) is None
