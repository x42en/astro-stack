"""Unit tests for :mod:`app.pipeline.adaptive.critic` (VisionCritic HTTP client)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from app.core.errors import ErrorCode, PipelineStepException
from app.pipeline.adaptive.critic import CriticVerdict, VisionCritic, _extract_json_object


def _preview_path(tmp_path: Path) -> Path:
    path = tmp_path / "preview.jpg"
    path.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg-bytes")
    return path


def _chat_response(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


class TestExtractJsonObject:
    def test_extracts_plain_json(self) -> None:
        assert _extract_json_object('{"a": 1}') == {"a": 1}

    def test_extracts_json_with_surrounding_prose(self) -> None:
        text = 'Sure, here is my answer:\n```json\n{"a": 1}\n```\nThanks!'
        assert _extract_json_object(text) == {"a": 1}

    def test_returns_none_for_non_json(self) -> None:
        assert _extract_json_object("no json here") is None

    def test_returns_none_for_json_array(self) -> None:
        assert _extract_json_object("[1, 2, 3]") is None


class TestVisionCriticCritique:
    @pytest.mark.asyncio
    async def test_parses_satisfied_verdict(self, tmp_path: Path) -> None:
        payload = {
            "satisfied": True,
            "confidence": 0.9,
            "reasoning": "Looks good.",
            "patch": {},
        }

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["model"] == "test-model"
            assert body["messages"][0]["role"] == "system"
            return httpx.Response(200, json=_chat_response(json.dumps(payload)))

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http:
            critic = VisionCritic(
                base_url="https://vllm.test/v1", model="test-model", http_client=http
            )
            verdict = await critic.critique(
                step_name="stretch_color",
                iteration=0,
                max_iterations=3,
                capability_context="stretch_strength: ...",
                current_values={"stretch_strength": 150.0},
                stats={"mean": 0.2},
                preview_jpeg_path=_preview_path(tmp_path),
                history=[],
            )

        assert isinstance(verdict, CriticVerdict)
        assert verdict.satisfied is True
        assert verdict.patch == {}

    @pytest.mark.asyncio
    async def test_parses_verdict_wrapped_in_markdown_fences(self, tmp_path: Path) -> None:
        payload = {
            "satisfied": False,
            "confidence": 0.4,
            "reasoning": "Too dim.",
            "patch": {"stretch_strength": 180.0},
        }
        content = f"```json\n{json.dumps(payload)}\n```"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_chat_response(content))

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http:
            critic = VisionCritic(base_url="https://vllm.test/v1", model="m", http_client=http)
            verdict = await critic.critique(
                step_name="stretch_color",
                iteration=1,
                max_iterations=3,
                capability_context="",
                current_values={},
                stats={},
                preview_jpeg_path=_preview_path(tmp_path),
                history=[],
            )

        assert verdict.satisfied is False
        assert verdict.patch == {"stretch_strength": 180.0}

    @pytest.mark.asyncio
    async def test_http_error_raises_critic_unavailable(self, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="service unavailable")

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http:
            critic = VisionCritic(base_url="https://vllm.test/v1", model="m", http_client=http)
            with pytest.raises(PipelineStepException) as exc_info:
                await critic.critique(
                    step_name="stretch_color",
                    iteration=0,
                    max_iterations=3,
                    capability_context="",
                    current_values={},
                    stats={},
                    preview_jpeg_path=_preview_path(tmp_path),
                    history=[],
                )

        assert exc_info.value.error_code is ErrorCode.PIPE_ADAPTIVE_CRITIC_UNAVAILABLE
        assert exc_info.value.retryable is False

    @pytest.mark.asyncio
    async def test_non_json_content_raises_invalid_response(self, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_chat_response("I cannot help with that."))

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http:
            critic = VisionCritic(base_url="https://vllm.test/v1", model="m", http_client=http)
            with pytest.raises(PipelineStepException) as exc_info:
                await critic.critique(
                    step_name="stretch_color",
                    iteration=0,
                    max_iterations=3,
                    capability_context="",
                    current_values={},
                    stats={},
                    preview_jpeg_path=_preview_path(tmp_path),
                    history=[],
                )

        assert exc_info.value.error_code is ErrorCode.PIPE_ADAPTIVE_CRITIC_INVALID_RESPONSE

    @pytest.mark.asyncio
    async def test_schema_violation_raises_invalid_response(self, tmp_path: Path) -> None:
        # Missing the required "satisfied" field.
        payload = {"confidence": 0.5, "reasoning": "hmm", "patch": {}}

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_chat_response(json.dumps(payload)))

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http:
            critic = VisionCritic(base_url="https://vllm.test/v1", model="m", http_client=http)
            with pytest.raises(PipelineStepException) as exc_info:
                await critic.critique(
                    step_name="stretch_color",
                    iteration=0,
                    max_iterations=3,
                    capability_context="",
                    current_values={},
                    stats={},
                    preview_jpeg_path=_preview_path(tmp_path),
                    history=[],
                )

        assert exc_info.value.error_code is ErrorCode.PIPE_ADAPTIVE_CRITIC_INVALID_RESPONSE

    @pytest.mark.asyncio
    async def test_sends_image_as_base64_data_uri(self, tmp_path: Path) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            captured["content"] = body["messages"][1]["content"]
            payload = {"satisfied": True, "confidence": 1.0, "reasoning": "ok", "patch": {}}
            return httpx.Response(200, json=_chat_response(json.dumps(payload)))

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http:
            critic = VisionCritic(base_url="https://vllm.test/v1", model="m", http_client=http)
            await critic.critique(
                step_name="stretch_color",
                iteration=0,
                max_iterations=3,
                capability_context="",
                current_values={},
                stats={},
                preview_jpeg_path=_preview_path(tmp_path),
                history=[],
            )

        image_part = next(p for p in captured["content"] if p["type"] == "image_url")
        assert image_part["image_url"]["url"].startswith("data:image/jpeg;base64,")
