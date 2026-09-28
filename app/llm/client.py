"""Provider-agnostic OpenAI-compatible chat client.

Thin wrapper over ``httpx`` used by every LLM consumer (vision critic,
future recommenders / chat). Handles the ``/chat/completions`` POST,
bearer auth, and response content extraction. Parsing of the content into
a domain verdict stays in the caller (see ``app.pipeline.adaptive.critic``).
"""

from __future__ import annotations

from typing import Any

import httpx

from app.core.logging import get_logger
from app.llm.types import LLMProfile

logger = get_logger(__name__)


class LLMClient:
    """Minimal OpenAI-compatible chat client bound to one :class:`LLMProfile`.

    Attributes:
        profile: Resolved connection parameters.
    """

    def __init__(
        self,
        profile: LLMProfile,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Bind the client to a resolved profile.

        Args:
            profile: Resolved connection parameters.
            http_client: Optional pre-configured client (tests/DI); a new one
                is created and owned by this instance otherwise.
        """
        self.profile = profile
        self._http = http_client or httpx.AsyncClient(timeout=profile.timeout_seconds)
        self._owns_http = http_client is None

    @property
    def base_url(self) -> str:
        """Base URL of the OpenAI-compatible API (no trailing slash)."""
        return self.profile.base_url.rstrip("/")

    @property
    def model(self) -> str:
        """Model id requested from the provider."""
        return self.profile.model

    @property
    def provider(self) -> str:
        """Provider key (``ollama`` | ``vllm`` | ``kilo`` | ``custom``)."""
        return self.profile.provider

    async def aclose(self) -> None:
        """Close the underlying HTTP client if it was created internally."""
        if self._owns_http:
            await self._http.aclose()

    async def chat_completions(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> dict[str, Any]:
        """POST ``/chat/completions`` and return the raw response body.

        Args:
            messages: OpenAI-style chat messages.
            temperature: Sampling temperature.
            max_tokens: Maximum completion tokens.

        Returns:
            The decoded JSON response body.

        Raises:
            httpx.HTTPError: On network failure or non-2xx status.
        """
        payload: dict[str, Any] = {
            "model": self.profile.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {}
        if self.profile.api_key:
            headers = {"Authorization": f"Bearer {self.profile.api_key}"}
        response = await self._http.post(
            f"{self.base_url}/chat/completions", json=payload, headers=headers
        )
        response.raise_for_status()
        body: dict[str, Any] = response.json()
        return body

    @staticmethod
    def extract_content(body: dict[str, Any]) -> str | None:
        """Extract the assistant message content from a chat response body.

        Args:
            body: Decoded ``/chat/completions`` response body.

        Returns:
            The message content, or ``None`` when the shape is unexpected.
        """
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return None
        return content if isinstance(content, str) else None
