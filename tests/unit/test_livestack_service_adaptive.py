"""Unit tests for the live adaptive-critic wiring in :mod:`app.livestack.service`."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import numpy as np
import pytest

from app.livestack import service as service_module
from app.livestack.processors import open_or_create_accumulator
from app.livestack.service import LiveStackService
from app.livestack.state import LiveStackState
from app.pipeline.adaptive.critic import CriticVerdict


class _FakeSettings:
    live_adaptive_critic_warmup_frames = 5
    live_adaptive_critic_recheck_every = 3
    live_adaptive_critic_max_attempts = 5


class _StubCritic:
    def __init__(self, verdict: CriticVerdict) -> None:
        self.verdict = verdict
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    async def critique(self, **kwargs: Any) -> CriticVerdict:
        self.calls.append(kwargs)
        return self.verdict

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def stub_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service_module, "get_settings", lambda: _FakeSettings())


def _build_service(tmp_path: Path, critic: _StubCritic) -> tuple[LiveStackService, AsyncMock]:
    from app.infrastructure.storage.file_store import FileStore

    class _FakeFsSettings:
        inbox_path = str(tmp_path / "inbox")
        sessions_path = str(tmp_path / "sessions")
        output_path = str(tmp_path / "output")

    file_store = FileStore(_FakeFsSettings())  # type: ignore[arg-type]
    state_repo = AsyncMock()
    service = LiveStackService(file_store, state_repo, event_bus=None, critic=critic)
    return service, state_repo


def _seed_accumulator(service: LiveStackService, session_id: uuid.UUID, shape: tuple[int, int]) -> None:
    acc = open_or_create_accumulator(
        service._store.live_accumulator_path(session_id), shape, np.dtype(np.float32)
    )
    acc[:] = 0.01
    acc.flush()


class TestMaybeRunAdaptiveCritic:
    @pytest.mark.asyncio
    async def test_not_due_returns_state_unchanged_and_skips_critic(self, tmp_path: Path) -> None:
        critic = _StubCritic(CriticVerdict(satisfied=True, confidence=0.9, reasoning="ok"))
        service, state_repo = _build_service(tmp_path, critic)
        session_id = uuid.uuid4()
        state = LiveStackState(session_id=str(session_id), frame_count=2)  # below warmup

        result = await service._maybe_run_adaptive_critic(session_id, state)

        assert result is state
        assert critic.calls == []
        state_repo.save.assert_not_called()

    @pytest.mark.asyncio
    async def test_due_evaluation_calls_critic_and_persists_state(self, tmp_path: Path) -> None:
        critic = _StubCritic(
            CriticVerdict(
                satisfied=False, confidence=0.4, reasoning="too dim",
                patch={"target_bkg": 0.30, "shadows_clip": -3.0},
            )
        )
        service, state_repo = _build_service(tmp_path, critic)
        session_id = uuid.uuid4()
        state = LiveStackState(session_id=str(session_id), frame_count=5, shape=(4, 4))
        _seed_accumulator(service, session_id, (4, 4))

        result = await service._maybe_run_adaptive_critic(session_id, state)

        assert len(critic.calls) == 1
        assert result.adaptive_target_bkg == 0.30
        assert result.adaptive_shadows_clip == -3.0
        assert result.adaptive_attempts == 1
        state_repo.save.assert_awaited_once_with(result)

    @pytest.mark.asyncio
    async def test_uses_injected_critic_not_a_new_default_instance(self, tmp_path: Path) -> None:
        # Regression guard: passing critic=... via the constructor must be
        # honoured (and never closed — it's owned by the caller, not us).
        critic = _StubCritic(CriticVerdict(satisfied=True, confidence=1.0, reasoning="ok"))
        service, _ = _build_service(tmp_path, critic)
        session_id = uuid.uuid4()
        state = LiveStackState(session_id=str(session_id), frame_count=5, shape=(4, 4))
        _seed_accumulator(service, session_id, (4, 4))

        await service._maybe_run_adaptive_critic(session_id, state)

        assert len(critic.calls) == 1
        assert critic.closed is False
