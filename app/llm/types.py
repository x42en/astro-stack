"""Shared LLM provider types.

Three named providers are supported:

* ``ollama`` — local stack, many models, ``/v1`` OpenAI-compatible endpoint.
* ``vllm`` — self-hosted server, serves one model very fast.
* ``kilo`` — external Kilo AI Gateway, hundreds of switchable models
  (``provider/model`` ids, e.g. ``qwen/qwen3.8-27b:free``) at
  ``https://api.kilo.ai/api/gateway``.
* ``custom`` — any other OpenAI-compatible endpoint (escape hatch).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

LLMProvider = Literal["ollama", "vllm", "kilo", "custom"]

PROVIDER_NAMES: tuple[str, ...] = ("ollama", "vllm", "kilo")


class LLMProfile(BaseModel):
    """Resolved connection parameters for one LLM provider.

    Attributes:
        provider: Provider key (``ollama`` | ``vllm`` | ``kilo`` | ``custom``).
        base_url: Base URL of the OpenAI-compatible API (including ``/v1``
            for Ollama).
        model: Model id to request (``provider/model`` for Kilo).
        api_key: Bearer token; empty string means no ``Authorization`` header.
        timeout_seconds: Request timeout in seconds.
    """

    provider: str = "vllm"
    base_url: str = ""
    model: str = ""
    api_key: str = ""
    timeout_seconds: float = Field(default=120.0, gt=0.0)

    @property
    def has_api_key(self) -> bool:
        """Whether a non-empty API key is configured."""
        return bool(self.api_key)
