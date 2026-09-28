"""Provider-agnostic LLM access for AstroStack.

Single entry point for every interaction with a large-language model
(adaptive vision critic today, narrative recommenders / chat tomorrow).
All providers speak the OpenAI-compatible ``POST {base_url}/chat/completions``
contract — Ollama via ``/v1``, vLLM natively, Kilo Gateway natively.
"""

from app.llm.client import LLMClient
from app.llm.factory import (
    KILO_DEFAULT_MODEL,
    build_critic_kwargs,
    list_provider_names,
    resolve_llm_profile,
)
from app.llm.types import LLMProfile, LLMProvider

__all__ = [
    "KILO_DEFAULT_MODEL",
    "LLMClient",
    "LLMProfile",
    "LLMProvider",
    "build_critic_kwargs",
    "list_provider_names",
    "resolve_llm_profile",
]
