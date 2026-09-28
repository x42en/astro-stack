"""Optional Langfuse tracing facade — fail-safe by construction.

Thin wrapper around the Langfuse Python SDK (OpenTelemetry-based) used to
debug processing sessions: one trace per pipeline job, one span per step
attempt, one generation per vision-critic call, plus downscaled JPEG
previews so the huge RAW/FITS masters never leave the server.

Guarantees:

* **No-op when disabled** — every helper short-circuits to ``None``/no-op
  unless ``LANGFUSE_ENABLED=true`` and both API keys are configured, so the
  default deployment never imports or touches the Langfuse SDK.
* **Never raises** — client creation, observation start, updates and flushes
  are all guarded; a broken or unreachable Langfuse instance can only add
  warnings to the application log, never fail a job.
* **Never swallows pipeline errors** — the :func:`lf_observation` context
  manager records the error on the observation, then re-raises unchanged.
"""

from __future__ import annotations

import asyncio
import base64
import io
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from PIL import Image

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Process-wide client state. The SDK client owns background export threads,
# so it is created at most once per process (API server / each ARQ worker).
_client: Any | None = None
_init_attempted: bool = False

_PREVIEW_QUALITY_STEPS: tuple[int, ...] = (85, 70, 55, 40)

# Hard cap on the best-effort per-job flush wait (see lf_flush_async).
_FLUSH_TIMEOUT_S: float = 5.0


def lf_client() -> Any | None:
    """Return the process-wide Langfuse client, or ``None`` when unusable.

    The client is lazily created once per process from
    :class:`~app.core.config.Settings`. Missing keys or an import failure
    degrade to ``None`` (all helpers then no-op).

    Returns:
        The Langfuse client instance, or ``None`` when tracing is disabled,
        unconfigured, or the SDK failed to initialise.
    """
    global _client, _init_attempted
    if _client is not None:
        return _client
    if _init_attempted:
        return None
    _init_attempted = True

    settings = get_settings()
    if not settings.langfuse_enabled:
        return None
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        logger.warning("langfuse_enabled_but_keys_missing")
        return None

    try:
        from langfuse import Langfuse

        _client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            base_url=settings.langfuse_base_url,
            sample_rate=settings.langfuse_sample_rate,
            timeout=int(settings.langfuse_timeout_seconds),
            tracing_enabled=True,
        )
        logger.info("langfuse_ready", base_url=settings.langfuse_base_url)
    except Exception:
        logger.warning("langfuse_init_failed", exc_info=True)
        _client = None
    return _client


def reset_for_tests() -> None:
    """Forget the cached client (unit tests only)."""
    global _client, _init_attempted
    _client = None
    _init_attempted = False


def lf_trace_id(seed: str) -> str | None:
    """Return a deterministic Langfuse trace ID for ``seed``.

    Deterministic IDs let a Langfuse trace be correlated with its AstroStack
    job/session (e.g. deep links, cross-referencing after ARQ retries — the
    same seed always maps to the same trace).

    Args:
        seed: Stable identifier, e.g. ``"job-<uuid>"`` or
            ``"livestack-<uuid>"``.

    Returns:
        A 32-character hex trace ID, or ``None`` when tracing is disabled.
    """
    client = lf_client()
    if client is None:
        return None
    try:
        return str(client.create_trace_id(seed=seed))
    except Exception:
        logger.warning("langfuse_trace_id_failed", exc_info=True)
        return None


@contextmanager
def lf_observation(
    name: str,
    *,
    as_type: str = "span",
    trace_id: str | None = None,
    model: str | None = None,
    input: Any | None = None,
    metadata: Any | None = None,
) -> Iterator[Any | None]:
    """Start a Langfuse observation, active in the OTel context for the block.

    All observations created inside the ``with`` block nest under it
    automatically. When tracing is disabled or the SDK fails, yields ``None``
    and the block still runs exactly as before.

    Args:
        name: Observation name, e.g. ``"step:denoise"`` or ``"vision-critic"``.
        as_type: ``"span"`` for work units, ``"generation"`` for LLM calls.
        trace_id: Pin the trace ID (deterministic, from :func:`lf_trace_id`).
            Only meaningful on the root observation of a trace.
        model: Model name for ``generation`` observations.
        input: Serialisable observation input (kept small: configs, stats,
            downscaled preview data URIs — never raw FITS/RAW bytes).
        metadata: Arbitrary JSON-serialisable metadata.

    Yields:
        A live observation object (``.update(...)``) or ``None`` when
        tracing is off.

    Raises:
        Never raises for its own bookkeeping; exceptions raised by the
        wrapped block propagate unchanged (after being recorded on the
        observation).
    """
    client = lf_client()
    if client is None:
        yield None
        return

    kwargs: dict[str, Any] = {"name": name, "as_type": as_type}
    if trace_id:
        kwargs["trace_context"] = {"trace_id": trace_id}
    if model is not None:
        kwargs["model"] = model
    if input is not None:
        kwargs["input"] = input
    if metadata is not None:
        kwargs["metadata"] = metadata

    try:
        cm = client.start_as_current_observation(**kwargs)
        obs = cm.__enter__()
    except Exception:  # SDK/startup failure — degrade to no-op
        logger.warning("langfuse_observation_start_failed", name=name, exc_info=True)
        yield None
        return

    try:
        yield obs
    except BaseException as exc:  # record then re-raise unchanged
        # BaseException on purpose: asyncio.CancelledError (ARQ job
        # cancellation / worker shutdown) must still close the span and
        # detach the OTel context, or the worker's tracing context leaks.
        if isinstance(exc, Exception):
            lf_mark_error(obs, f"{type(exc).__name__}: {exc}")
        try:
            cm.__exit__(type(exc), exc, exc.__traceback__)
        except Exception:
            logger.warning("langfuse_observation_exit_failed", name=name, exc_info=True)
        raise
    try:
        cm.__exit__(None, None, None)
    except Exception:
        logger.warning("langfuse_observation_exit_failed", name=name, exc_info=True)


@contextmanager
def lf_attributes(
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
    metadata: Any | None = None,
) -> Iterator[None]:
    """Propagate trace attributes (session, tags…) to all nested observations.

    Yields nothing and never raises; when tracing is disabled this is a
    plain no-op passthrough.
    """
    client = lf_client()
    if client is None:
        yield
        return
    try:
        from langfuse import propagate_attributes

        cm = propagate_attributes(
            user_id=user_id,
            session_id=session_id,
            tags=tags,
            metadata=metadata,
        )
        cm.__enter__()
    except Exception:  # SDK/startup failure — degrade to no-op
        logger.warning("langfuse_attributes_start_failed", exc_info=True)
        yield
        return
    try:
        yield
    finally:
        try:
            cm.__exit__(None, None, None)
        except Exception:
            logger.warning("langfuse_attributes_exit_failed", exc_info=True)


def lf_update(obs: Any | None, **fields: Any) -> None:
    """Best-effort update of an observation (input/output/metadata/level…)."""
    if obs is None:
        return
    try:
        obs.update(**fields)
    except Exception:
        logger.warning("langfuse_update_failed", exc_info=True)


def lf_mark_error(obs: Any | None, message: str) -> None:
    """Flag an observation as failed with a truncated status message."""
    lf_update(obs, level="ERROR", status_message=message[:512])


def lf_preview_data_uri(preview_path: Path | str) -> str | None:
    """Encode a JPEG preview as a base64 data URI within the size budget.

    The pipeline already renders 900px JPEG previews for the UI/WebSocket
    (see ``PipelineOrchestrator._fits_to_preview_jpeg``); this helper re-uses
    those files. Oversized files are progressively downscaled to
    ``langfuse_preview_max_px`` and re-compressed; anything still above
    ``langfuse_preview_max_kb`` after the last pass is dropped.

    Never reads RAW/FITS sources — callers pass already-rendered JPEGs.

    Args:
        preview_path: Path to an existing JPEG preview file.

    Returns:
        A ``data:image/jpeg;base64,…`` URI, or ``None`` when tracing is
        disabled, previews are disabled, the file is missing, or it cannot
        fit the budget.
    """
    settings = get_settings()
    if not settings.langfuse_attach_previews:
        return None
    if lf_client() is None:
        return None

    try:
        path = Path(preview_path)
        if not path.exists():
            return None
        budget_bytes = settings.langfuse_preview_max_kb * 1024
        data = path.read_bytes()
        if len(data) <= budget_bytes:
            return _encode_data_uri(data)

        # draft() makes libjpeg decode at a reduced 1/2^n scale when the file
        # is much larger than the target — turns a multi-megapixel decode
        # (~0.4 s) into a few tens of milliseconds for live previews.
        with Image.open(io.BytesIO(data)) as img:
            img.draft("RGB", (settings.langfuse_preview_max_px, settings.langfuse_preview_max_px))
            rgb = img.convert("RGB")
        max_px = settings.langfuse_preview_max_px
        scale = min(1.0, max_px / max(rgb.width, rgb.height))
        if scale < 1.0:
            rgb = rgb.resize(
                (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale))),
                Image.Resampling.LANCZOS,
            )
        for quality in _PREVIEW_QUALITY_STEPS:
            buf = io.BytesIO()
            rgb.save(buf, format="JPEG", quality=quality)
            encoded = buf.getvalue()
            if len(encoded) <= budget_bytes:
                return _encode_data_uri(encoded)
        logger.warning(
            "langfuse_preview_over_budget_dropped",
            path=str(path),
            budget_kb=settings.langfuse_preview_max_kb,
        )
        return None
    except Exception:
        logger.warning(
            "langfuse_preview_encode_failed", path=str(preview_path), exc_info=True
        )
        return None


def _encode_data_uri(jpeg_bytes: bytes) -> str:
    """Return a JPEG byte string as a base64 ``data:`` URI."""
    return "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode("ascii")


async def lf_flush_async() -> None:
    """Flush buffered Langfuse events to the server (best-effort, async-safe).

    Called at the end of every ARQ job so a long-lived worker never loses
    more than the current job's data if it is killed. Bounded (the SDK
    internally caps the export wait at ~30 s) and cancellation-safe: an
    in-flight job return or ARQ ``Retry`` must never be replaced by flush
    bookkeeping.
    """
    client = lf_client()
    if client is None:
        return
    flush = asyncio.to_thread(client.flush)
    try:
        await asyncio.wait_for(asyncio.shield(flush), timeout=_FLUSH_TIMEOUT_S)
    except asyncio.CancelledError:
        # Cancellation (worker shutdown / job cancel) must not replace the
        # caller's in-flight result. The flush thread keeps running in the
        # background; the SDK's atexit shutdown flushes again at exit.
        return
    except Exception:  # timeout, network, SDK — log and move on
        logger.warning("langfuse_flush_failed", exc_info=True)


def job_trace_seed(job_id: uuid.UUID) -> str:
    """Seed string for the deterministic trace ID of a pipeline job."""
    return f"astrostack-job-{job_id}"


def livestack_trace_seed(session_id: uuid.UUID) -> str:
    """Seed string for the shared trace of one live-stacking session."""
    return f"astrostack-livestack-{session_id}"
