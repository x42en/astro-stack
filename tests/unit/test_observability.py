"""Unit tests for the optional Langfuse observability facade.

Covers the fail-safe contract: when tracing is disabled or the SDK misbehaves,
every helper degrades to a no-op and the wrapped code runs unchanged. The
Langfuse SDK is faked — no network, no Langfuse server, no Docker.
"""

from __future__ import annotations

import base64
import io
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from app.core import observability


@pytest.fixture(autouse=True)
def _reset_client() -> Iterator[None]:
    """Isolate the process-wide client singleton between tests."""
    observability.reset_for_tests()
    yield
    observability.reset_for_tests()


class FakeObservation:
    """Records update calls, mimicking langfuse observation objects."""

    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []

    def update(self, **fields: Any) -> None:
        self.updates.append(fields)


class FakeClient:
    """Minimal stand-in for langfuse.Langfuse."""

    def __init__(self, *, fail_start: bool = False) -> None:
        self.started: list[dict[str, Any]] = []
        self.observations: list[FakeObservation] = []
        self.trace_id_seeds: list[str] = []
        self.flushed = 0
        self._fail_start = fail_start

    def start_as_current_observation(self, **kwargs: Any) -> Any:
        self.started.append(kwargs)
        if self._fail_start:
            raise RuntimeError("langfuse down")
        obs = FakeObservation()
        self.observations.append(obs)

        @contextmanager
        def _cm() -> Any:
            yield obs

        return _cm()

    def create_trace_id(self, *, seed: str | None = None) -> str:
        self.trace_id_seeds.append(seed or "")
        return "0123456789abcdef0123456789abcdef"

    def flush(self) -> None:
        self.flushed += 1


def _use_fake_client(fake: FakeClient) -> None:
    observability.reset_for_tests()
    observability._client = fake
    observability._init_attempted = True


# ── Disabled / unconfigured behaviour ─────────────────────────────────────────


class TestDisabled:
    def test_client_is_none_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = _fake_settings(langfuse_enabled=False)
        monkeypatch.setattr(observability, "get_settings", lambda: settings)
        assert observability.lf_client() is None

    def test_client_is_none_when_keys_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = _fake_settings(langfuse_enabled=True, public_key="", secret_key="")
        monkeypatch.setattr(observability, "get_settings", lambda: settings)
        assert observability.lf_client() is None

    def test_observation_noop_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(observability, "get_settings", lambda: _fake_settings(False))
        with observability.lf_observation("step:test") as obs:
            assert obs is None
        assert observability.lf_trace_id("seed") is None

    def test_body_still_runs_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(observability, "get_settings", lambda: _fake_settings(False))
        executed = False
        with observability.lf_observation("step:test") as obs:
            assert obs is None
            executed = True
        assert executed is True

    def test_preview_returns_none_when_disabled(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        path = _write_jpeg(tmp_path / "p.jpg", size=64)
        monkeypatch.setattr(observability, "get_settings", lambda: _fake_settings(False))
        assert observability.lf_preview_data_uri(path) is None

    def test_trace_id_seeds_are_deterministic(self) -> None:
        job_id = uuid.uuid4()
        seed = observability.job_trace_seed(job_id)
        assert seed == f"astrostack-job-{job_id}"
        assert seed == observability.job_trace_seed(uuid.UUID(str(job_id)))
        live = observability.livestack_trace_seed(job_id)
        assert live.startswith("astrostack-livestack-")
        assert live != seed


# ── Happy path ────────────────────────────────────────────────────────────────


class TestObservation:
    def test_yields_observation_and_passes_kwargs(self) -> None:
        fake = FakeClient()
        _use_fake_client(fake)
        with observability.lf_observation(
            "step:denoise",
            trace_id=observability.lf_trace_id("astrostack-job-abc"),
            input={"attempt": 1},
            metadata={"k": "v"},
        ) as obs:
            assert isinstance(obs, FakeObservation)
            observability.lf_update(obs, output={"done": True})
        assert fake.started[0]["name"] == "step:denoise"
        assert fake.started[0]["trace_context"] == {
            "trace_id": "0123456789abcdef0123456789abcdef"
        }
        assert obs.updates == [{"output": {"done": True}}]

    def test_error_in_body_is_recorded_and_reraised(self) -> None:
        fake = FakeClient()
        _use_fake_client(fake)
        with (
            pytest.raises(ValueError, match="siril exploded"),
            observability.lf_observation("step:test") as obs,
        ):
            assert obs is not None
            raise ValueError("siril exploded")
        assert obs.updates[0]["level"] == "ERROR"
        assert "siril exploded" in obs.updates[0]["status_message"]

    def test_base_exception_still_exits_and_propagates(self) -> None:
        """CancelledError/KeyboardInterrupt must not leak the OTel context."""
        fake = FakeClient()
        _use_fake_client(fake)
        with (
            pytest.raises(KeyboardInterrupt),
            observability.lf_observation("step:test") as obs,
        ):
            assert obs is not None
            raise KeyboardInterrupt
        assert all(u.get("level") != "ERROR" for u in obs.updates)

    def test_broken_exit_does_not_replace_pipeline_error(self) -> None:
        class BrokenExitCM:
            def __enter__(self) -> FakeObservation:
                return FakeObservation()

            def __exit__(self, *args: Any) -> None:
                raise RuntimeError("exit boom")

        class BrokenExitClient(FakeClient):
            def start_as_current_observation(self, **kwargs: Any) -> Any:
                self.started.append(kwargs)
                return BrokenExitCM()

        _use_fake_client(BrokenExitClient())
        with (
            pytest.raises(ValueError, match="original"),
            observability.lf_observation("step:test"),
        ):
            raise ValueError("original")
        # Success path: a failing __exit__ must not fail a successful block.
        with observability.lf_observation("step:test"):
            pass

    def test_start_failure_degrades_to_noop(self) -> None:
        fake = FakeClient(fail_start=True)
        _use_fake_client(fake)
        executed = False
        with observability.lf_observation("step:test") as obs:
            assert obs is None
            executed = True
        assert executed is True

    def test_update_failure_never_raises(self) -> None:
        fake = FakeClient()
        _use_fake_client(fake)
        with observability.lf_observation("step:test") as obs:
            assert obs is not None
            obs.update = lambda **kw: (_ for _ in ()).throw(RuntimeError("boom"))
            observability.lf_update(obs, output="x")
            observability.lf_mark_error(obs, "message")


class TestTraceId:
    def test_deterministic_from_seed(self) -> None:
        fake = FakeClient()
        _use_fake_client(fake)
        first = observability.lf_trace_id("astrostack-job-1")
        second = observability.lf_trace_id("astrostack-job-1")
        assert first == second == "0123456789abcdef0123456789abcdef"
        assert fake.trace_id_seeds == ["astrostack-job-1", "astrostack-job-1"]

    def test_none_when_client_unavailable(self) -> None:
        observability.reset_for_tests()
        assert observability.lf_trace_id("x") is None


class TestFlush:
    @pytest.mark.asyncio
    async def test_flush_calls_client_once(self) -> None:
        fake = FakeClient()
        _use_fake_client(fake)
        await observability.lf_flush_async()
        assert fake.flushed == 1

    @pytest.mark.asyncio
    async def test_flush_swallows_client_errors(self) -> None:
        fake = FakeClient()
        _use_fake_client(fake)

        def boom() -> None:
            raise RuntimeError("langfuse down")

        fake.flush = boom  # type: ignore[method-assign]
        await observability.lf_flush_async()

    @pytest.mark.asyncio
    async def test_flush_noop_when_disabled(self) -> None:
        observability.reset_for_tests()
        await observability.lf_flush_async()


# ── Preview budget handling ───────────────────────────────────────────────────


def _fake_settings(
    langfuse_enabled: bool = True,
    public_key: str = "pk-lf-test",
    secret_key: str = "sk-lf-test",
    attach_previews: bool = True,
    max_px: int = 512,
    max_kb: int = 400,
) -> Any:
    """Build a lightweight settings stub with only the fields the facade reads."""
    return SimpleNamespace(
        langfuse_enabled=langfuse_enabled,
        langfuse_public_key=public_key,
        langfuse_secret_key=secret_key,
        langfuse_sample_rate=1.0,
        langfuse_timeout_seconds=10.0,
        langfuse_attach_previews=attach_previews,
        langfuse_preview_max_px=max_px,
        langfuse_preview_max_kb=max_kb,
    )


def _jpeg_bytes(width: int, height: int, color: tuple[int, int, int]) -> bytes:
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _write_jpeg_file(path: Path, width: int, height: int, color: tuple[int, int, int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_jpeg_bytes(width, height, color))
    return path


def _write_jpeg(path: Path, size: int) -> Path:
    return _write_jpeg_file(path, size, size, (128, 64, 32))


class TestPreviewDataUri:
    def test_small_jpeg_passes_through(self, tmp_path: Path) -> None:
        _use_fake_client(FakeClient())
        path = _write_jpeg_file(tmp_path / "preview.jpg", 200, 200, (10, 200, 30))
        uri = observability.lf_preview_data_uri(path)
        assert uri is not None
        assert uri.startswith("data:image/jpeg;base64,")
        payload = base64.b64decode(uri.split(",", 1)[1])
        assert payload[:2] == b"\xff\xd8"  # JPEG SOI

    def test_accepts_str_path(self, tmp_path: Path) -> None:
        _use_fake_client(FakeClient())
        path = _write_jpeg_file(tmp_path / "preview.jpg", 64, 64, (0, 0, 0))
        assert observability.lf_preview_data_uri(str(path)) is not None

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        _use_fake_client(FakeClient())
        assert observability.lf_preview_data_uri(tmp_path / "nope.jpg") is None

    def test_disabled_flag_skips_encoding(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _use_fake_client(FakeClient())
        path = _write_jpeg_file(tmp_path / "preview.jpg", 64, 64, (1, 2, 3))
        monkeypatch.setattr(
            observability, "get_settings", lambda: _fake_settings(attach_previews=False)
        )
        assert observability.lf_preview_data_uri(path) is None

    def test_oversized_jpeg_is_downscaled_into_budget(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _use_fake_client(FakeClient())
        # High-frequency noise JPEG — large on disk, forces re-encode path.
        img = Image.effect_noise((2000, 2000), 64).convert("RGB")
        path = tmp_path / "big.jpg"
        path.write_bytes(_noise_jpeg(img))

        monkeypatch.setattr(
            observability,
            "get_settings",
            lambda: _fake_settings(max_px=512, max_kb=150),
        )
        uri = observability.lf_preview_data_uri(path)
        assert uri is not None
        raw = base64.b64decode(uri.split(",", 1)[1])
        assert len(raw) <= 150 * 1024
        with Image.open(io.BytesIO(raw)) as decoded:
            assert max(decoded.size) <= 512

    def test_undownscalable_jpeg_is_dropped(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _use_fake_client(FakeClient())
        img = Image.effect_noise((512, 512), 100).convert("RGB")
        path = tmp_path / "dense.jpg"
        path.write_bytes(_noise_jpeg(img))
        monkeypatch.setattr(
            observability,
            "get_settings",
            lambda: _fake_settings(max_px=512, max_kb=1),  # 1 KB floor is unreachable
        )
        assert observability.lf_preview_data_uri(path) is None


def _noise_jpeg(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=100, subsampling=0)
    return buf.getvalue()
