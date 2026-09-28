"""Vision-language critic client for the Phase 2 adaptive processing loop.

Talks to an OpenAI-compatible ``/chat/completions`` endpoint (the verified
target is a self-hosted vLLM server, see repository memory notes) with a
JPEG preview image plus numeric stats, and asks it to judge whether a
pipeline step's output is acceptable or should be refined further.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field, ValidationError

from app.core.config import get_settings
from app.core.errors import ErrorCode, PipelineStepException
from app.core.logging import get_logger
from app.core.observability import lf_client, lf_mark_error, lf_observation, lf_update

logger = get_logger(__name__)


class CriticVerdict(BaseModel):
    """Structured verdict returned by the vision critic for one iteration.

    Attributes:
        satisfied: True when the critic judges the current result acceptable
            and the loop should stop.
        confidence: Critic's self-reported confidence in ``satisfied`` (0-1).
        reasoning: One or two sentence English explanation, shown in the UI
            reasoning trace.
        patch: Proposed changes to apply before the next iteration. Expected
            to be empty when ``satisfied`` is True. Keys are validated and
            clamped by :mod:`app.pipeline.adaptive.patch` before being
            applied — this model does not itself restrict which fields may
            appear.
    """

    satisfied: bool
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    reasoning: str = ""
    patch: dict[str, Any] = Field(default_factory=dict)


_SYSTEM_PROMPT = (
    "You are an expert astrophotography image-processing critic embedded in "
    "a fully automated pipeline used by novices with no manual editing "
    "skills. You are shown a JPEG preview of the current output of one "
    "processing step, its numeric statistics, and the list of parameters "
    "you are allowed to adjust for this step (with their role, effect, and "
    "risk). Decide whether the result is acceptable. If not, propose a "
    "small, targeted patch using ONLY the listed parameter names. Prefer no "
    "change over a risky change: these images are processed unattended, so "
    "an over-corrected image is worse than a slightly conservative one. "
    "Respond with STRICT JSON only, no markdown fences and no extra text, "
    "matching exactly this schema: "
    '{"satisfied": bool, "confidence": number between 0 and 1, '
    '"reasoning": "short string", "patch": {"field_name": value, ...}}. '
    'Leave "patch" empty ({}) when satisfied is true.'
)


class VisionCritic:
    """Calls an OpenAI-compatible vision-language model to critique a step output.

    Attributes:
        base_url: Base URL of the OpenAI-compatible API (including ``/v1``).
        model: Model name to request (vLLM's ``--served-model-name``).
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialise the critic client.

        Args:
            base_url: Base URL of the OpenAI-compatible API; defaults to
                ``Settings.vllm_base_url``.
            model: Model name; defaults to ``Settings.vllm_model``.
            api_key: Bearer token; defaults to ``Settings.vllm_api_key``
                (empty string means no ``Authorization`` header is sent).
            timeout: Request timeout in seconds; defaults to
                ``Settings.vllm_timeout_seconds``.
            http_client: Optional pre-configured client (for tests/DI); a new
                one is created and owned by this instance otherwise.
        """
        settings = get_settings()
        self.base_url = (base_url or settings.vllm_base_url).rstrip("/")
        self.model = model or settings.vllm_model
        self.api_key = settings.vllm_api_key if api_key is None else api_key
        self.timeout = timeout if timeout is not None else settings.vllm_timeout_seconds
        self._http = http_client or httpx.AsyncClient(timeout=self.timeout)
        self._owns_http = http_client is None

    async def aclose(self) -> None:
        """Close the underlying HTTP client if it was created internally."""
        if self._owns_http:
            await self._http.aclose()

    async def critique(
        self,
        *,
        step_name: str,
        iteration: int,
        max_iterations: int,
        capability_context: str,
        current_values: dict[str, Any],
        stats: dict[str, Any],
        preview_jpeg_path: Path,
        history: list[dict[str, Any]],
    ) -> CriticVerdict:
        """Ask the vision model to critique the current step output.

        Args:
            step_name: Machine name of the pipeline step being refined.
            iteration: Current 0-based iteration number.
            max_iterations: Maximum allowed iterations for this loop.
            capability_context: Rendered tool-catalog text for the fields the
                critic may adjust, e.g. from
                :func:`app.pipeline.adaptive.tool_catalog.render_capabilities_for_prompt`.
            current_values: Current value of every adjustable field.
            stats: Numeric image statistics (e.g. from
                :meth:`~app.pipeline.adapters.siril_pyadapter.SirilPyAdapter.get_image_stats`).
            preview_jpeg_path: Path to the JPEG preview to attach as an image.
            history: Previous iterations' records for this run, so the
                critic can avoid oscillating between two settings.

        Returns:
            The parsed :class:`CriticVerdict`.

        Raises:
            PipelineStepException: On network failure
                (``PIPE_ADAPTIVE_CRITIC_UNAVAILABLE``) or a malformed/
                unparsable model response (``PIPE_ADAPTIVE_CRITIC_INVALID_RESPONSE``).
        """
        image_b64 = base64.b64encode(preview_jpeg_path.read_bytes()).decode("ascii")
        user_text = (
            f"Step: {step_name}\n"
            f"Iteration: {iteration + 1}/{max_iterations}\n"
            f"Adjustable parameters:\n{capability_context}\n\n"
            f"Current values: {json.dumps(current_values)}\n"
            f"Image statistics: {json.dumps(stats)}\n"
            f"Previous iterations this run: {json.dumps(history)}\n"
        )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                    ],
                },
            ],
            "temperature": 0.2,
            "max_tokens": 800,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

        # One Langfuse generation per critic call. The traced input mirrors
        # the exact messages sent to the model EXCEPT the image: full-res
        # base64 (live previews can be several MB) would blow past the
        # LANGFUSE_PREVIEW_MAX_KB budget — the image is visible on the
        # enclosing span instead, already size-capped. No-op cost when
        # Langfuse tracing is disabled.
        lf_input: Any = None
        if lf_client() is not None:
            lf_input = _traced_messages(payload["messages"])
        with lf_observation(
            "vision-critic",
            as_type="generation",
            model=self.model,
            input=lf_input,
            metadata={
                "step": step_name,
                "iteration": iteration,
                "max_iterations": max_iterations,
            },
        ) as lf_gen:
            try:
                response = await self._http.post(
                    f"{self.base_url}/chat/completions", json=payload, headers=headers
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                lf_mark_error(
                    lf_gen,
                    f"Vision critic request failed for step {step_name!r}: {exc}",
                )
                raise PipelineStepException(
                    ErrorCode.PIPE_ADAPTIVE_CRITIC_UNAVAILABLE,
                    f"Vision critic request failed for step {step_name!r}: {exc}",
                    step_name=step_name,
                    retryable=False,
                ) from exc

            body = response.json()
            try:
                content = body["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                lf_mark_error(
                    lf_gen,
                    f"Vision critic response for step {step_name!r} has no message content.",
                )
                raise PipelineStepException(
                    ErrorCode.PIPE_ADAPTIVE_CRITIC_INVALID_RESPONSE,
                    f"Vision critic response for step {step_name!r} has no message content.",
                    step_name=step_name,
                    retryable=False,
                    details={"body": body},
                ) from exc

            verdict_dict = _extract_json_object(content)
            if verdict_dict is None:
                lf_mark_error(
                    lf_gen,
                    f"Vision critic response for step {step_name!r} is not valid JSON.",
                )
                raise PipelineStepException(
                    ErrorCode.PIPE_ADAPTIVE_CRITIC_INVALID_RESPONSE,
                    f"Vision critic response for step {step_name!r} is not valid JSON.",
                    step_name=step_name,
                    retryable=False,
                    details={"content": content[:2000]},
                )

            try:
                verdict = CriticVerdict.model_validate(verdict_dict)
            except ValidationError as exc:
                schema_msg = (
                    f"Vision critic response for step {step_name!r} "
                    f"failed schema validation: {exc}"
                )
                lf_mark_error(lf_gen, schema_msg)
                raise PipelineStepException(
                    ErrorCode.PIPE_ADAPTIVE_CRITIC_INVALID_RESPONSE,
                    schema_msg,
                    step_name=step_name,
                    retryable=False,
                    details={"content": content[:2000]},
                ) from exc

            lf_update(lf_gen, output=verdict.model_dump(), usage_details=_extract_usage(body))
            return verdict


# Bounded stand-in for the preview bytes in traced generation inputs (the
# actual image is attached, size-capped, to the enclosing span).
_IMAGE_PLACEHOLDER = "<image omitted from trace — see span preview>"


def _traced_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a trace-safe copy of chat messages with images placeholder-ed.

    The critic sends the preview JPEG as base64 inside the ``image_url``
    part; that byte string can reach several megabytes for live previews,
    so the traced copy replaces it with a bounded placeholder. The image
    itself is attached (size-capped) to the enclosing Langfuse span.

    Args:
        messages: OpenAI-style chat messages actually sent to the model.

    Returns:
        A shallow copy safe to attach as a Langfuse generation input.
    """
    traced: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            traced.append(msg)
            continue
        parts: list[Any] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": _IMAGE_PLACEHOLDER},
                    }
                )
            else:
                parts.append(part)
        traced.append({**msg, "content": parts})
    return traced


def _extract_usage(body: dict[str, Any]) -> dict[str, int] | None:
    """Map an OpenAI-style ``usage`` object to Langfuse ``usage_details``.

    Args:
        body: Full chat/completions response body.

    Returns:
        ``{"input": …, "output": …}`` when token counts are present, else
        ``None`` (self-hosted vLLM builds may omit usage).
    """
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return None
    details: dict[str, int] = {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if isinstance(prompt_tokens, int):
        details["input"] = prompt_tokens
    if isinstance(completion_tokens, int):
        details["output"] = completion_tokens
    return details or None


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first top-level JSON object from a model response.

    Tolerates common LLM formatting noise (markdown code fences, leading or
    trailing prose) by locating the outermost matching ``{...}`` span.

    Args:
        text: Raw model response content.

    Returns:
        The parsed dict, or ``None`` if no valid JSON object could be found.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
