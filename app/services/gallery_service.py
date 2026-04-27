"""Public gallery service: publish/unpublish, list, signed-URL downloads.

Acts on top of :class:`AstroSession` rows: the gallery is just a flag
(``is_in_gallery``) that owners flip on completed sessions.  Public reads
go through this service so the schema-level filter and the augmented
payload (last completed job's preview path) live in one place.

Downloads use short-lived HMAC tokens to keep the endpoint stateless and
avoid leaking direct file paths.  Tokens carry ``session_id``, ``format``,
and an expiry timestamp; verification recomputes the digest with
:func:`hmac.compare_digest` to thwart timing attacks.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Optional

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from app.core.config import get_settings
from app.core.errors import ConflictException, ErrorCode, NotFoundException
from app.domain.gallery import GalleryDownload
from app.domain.job import JobStatus, PipelineJob
from app.domain.session import AstroSession, SessionStatus

logger = structlog.get_logger(__name__)

# Allowed download formats and their on-disk attribute on the latest
# completed PipelineJob row.
_FORMAT_ATTR = {
    "tiff": "output_tiff_path",
    "fits": "output_fits_path",
}

# Signed-URL lifetime — long enough for the user to switch tabs and
# acknowledge a "Download" prompt, short enough to be useless if leaked.
TOKEN_TTL_SECONDS = 30 * 60  # 30 minutes

# Naive in-memory rate limiter.  Acceptable for single-instance deploys;
# replace with Redis if the API ever scales horizontally.
_RATE_LIMIT_MAX = 5
_RATE_LIMIT_WINDOW = 3600  # 1 hour
_rate_buckets: dict[str, Deque[float]] = defaultdict(deque)


@dataclass(slots=True)
class GalleryItem:
    """Augmented payload for a public gallery card."""

    session: AstroSession
    job_id: uuid.UUID
    preview_path: str


class RateLimitExceeded(ConflictException):
    """Raised when a single IP requests too many downloads in a window."""

    def __init__(self) -> None:
        super().__init__(
            ErrorCode.SESS_ALREADY_PROCESSING,
            "Too many download requests; please retry in an hour.",
            details={"retry_after_seconds": _RATE_LIMIT_WINDOW},
        )


# ───────────────────────── Token helpers ──────────────────────────────────


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def _sign_payload(payload: dict) -> str:
    secret = get_settings().jwt_secret.encode("utf-8")
    body = _b64url(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    sig = hmac.new(secret, body.encode("ascii"), hashlib.sha256).digest()
    return f"{body}.{_b64url(sig)}"


def _verify_token(token: str) -> dict:
    """Decode and verify a download token.

    Raises :class:`NotFoundException` on any failure (bad format, bad
    signature, expired) — we deliberately collapse all errors to a single
    opaque 404 so attackers cannot probe individual failure modes.
    """
    err = NotFoundException(
        ErrorCode.PIPE_EXPORT_FAILED,
        "Download link is invalid or has expired.",
    )
    try:
        body, sig = token.split(".", 1)
    except ValueError as exc:
        raise err from exc

    secret = get_settings().jwt_secret.encode("utf-8")
    expected = hmac.new(secret, body.encode("ascii"), hashlib.sha256).digest()
    try:
        provided = _b64url_decode(sig)
    except (ValueError, base64.binascii.Error) as exc:  # type: ignore[attr-defined]
        raise err from exc
    if not hmac.compare_digest(expected, provided):
        raise err

    try:
        payload = json.loads(_b64url_decode(body))
    except (ValueError, json.JSONDecodeError) as exc:
        raise err from exc

    if payload.get("exp", 0) < int(time.time()):
        raise err
    return payload


# ───────────────────────── Service ────────────────────────────────────────


class GalleryService:
    """Coordinates gallery publish/unpublish, listing, and downloads."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    # ── Owner actions ───────────────────────────────────────────────

    async def publish(
        self,
        session_id: uuid.UUID,
        author_name: Optional[str] = None,
    ) -> AstroSession:
        """Mark a completed session as visible in the public gallery."""
        from datetime import datetime

        session = await self.db.get(AstroSession, session_id)
        if session is None:
            raise NotFoundException(
                ErrorCode.SESS_NOT_FOUND,
                f"Session '{session_id}' not found.",
            )
        if session.status != SessionStatus.COMPLETED.value:
            raise ConflictException(
                ErrorCode.SESS_ALREADY_PROCESSING,
                "Only completed sessions can be published to the gallery.",
                details={"status": session.status},
            )
        session.is_in_gallery = True
        session.gallery_published_at = datetime.utcnow()
        if author_name:
            session.gallery_author_name = author_name[:120]
        elif not session.gallery_author_name:
            session.gallery_author_name = "Astronomer"
        self.db.add(session)
        await self.db.commit()
        await self.db.refresh(session)
        logger.info("gallery_published", session_id=str(session.id))
        return session

    async def unpublish(self, session_id: uuid.UUID) -> AstroSession:
        """Hide a session from the public gallery (idempotent)."""
        session = await self.db.get(AstroSession, session_id)
        if session is None:
            raise NotFoundException(
                ErrorCode.SESS_NOT_FOUND,
                f"Session '{session_id}' not found.",
            )
        session.is_in_gallery = False
        self.db.add(session)
        await self.db.commit()
        await self.db.refresh(session)
        logger.info("gallery_unpublished", session_id=str(session.id))
        return session

    # ── Public reads ────────────────────────────────────────────────

    async def list_public(
        self,
        offset: int = 0,
        limit: int = 60,
    ) -> list[GalleryItem]:
        """Return published sessions enriched with their preview path."""
        stmt = (
            select(AstroSession)
            .where(AstroSession.is_in_gallery.is_(True))  # type: ignore[union-attr]
            .order_by(AstroSession.gallery_published_at.desc())  # type: ignore[attr-defined]
            .offset(offset)
            .limit(limit)
        )
        sessions = list((await self.db.execute(stmt)).scalars().all())
        items: list[GalleryItem] = []
        for session in sessions:
            job = await self._latest_completed_job(session.id)
            if job is None or not job.output_preview_path:
                continue
            items.append(
                GalleryItem(
                    session=session,
                    job_id=job.id,
                    preview_path=job.output_preview_path,
                )
            )
        return items

    async def _latest_completed_job(
        self, session_id: uuid.UUID
    ) -> Optional[PipelineJob]:
        stmt = (
            select(PipelineJob)
            .where(
                PipelineJob.session_id == session_id,
                PipelineJob.status == JobStatus.COMPLETED.value,
            )
            .order_by(PipelineJob.created_at.desc())  # type: ignore[attr-defined]
            .limit(1)
        )
        return (await self.db.execute(stmt)).scalars().first()

    # ── Download flow ───────────────────────────────────────────────

    async def request_download(
        self,
        session_id: uuid.UUID,
        email: str,
        fmt: str,
        requester_ip: Optional[str],
    ) -> tuple[str, int]:
        """Validate the request, log it, and return ``(token, expires_at)``.

        Raises :class:`RateLimitExceeded` when the requester IP has used
        up its quota for the rolling window.
        """
        if fmt not in _FORMAT_ATTR:
            raise NotFoundException(
                ErrorCode.PIPE_EXPORT_FAILED,
                f"Unsupported format '{fmt}'.",
            )
        session = await self.db.get(AstroSession, session_id)
        if session is None or not session.is_in_gallery:
            raise NotFoundException(
                ErrorCode.SESS_NOT_FOUND,
                "Gallery item not found.",
            )

        # Rate-limit by IP (fall back to email if IP missing)
        bucket_key = requester_ip or f"email:{email.lower()}"
        self._enforce_rate_limit(bucket_key)

        # Audit row (best-effort: never block the download on log errors)
        record = GalleryDownload(
            session_id=session_id,
            email=email[:320],
            format=fmt,
            requester_ip=(requester_ip or None),
        )
        self.db.add(record)
        await self.db.commit()

        exp = int(time.time()) + TOKEN_TTL_SECONDS
        token = _sign_payload(
            {"sid": str(session_id), "fmt": fmt, "exp": exp}
        )
        return token, exp

    async def serve_download(
        self,
        session_id: uuid.UUID,
        fmt: str,
        token: str,
    ) -> Path:
        """Verify ``token`` and return the on-disk path to stream."""
        payload = _verify_token(token)
        if payload.get("sid") != str(session_id) or payload.get("fmt") != fmt:
            raise NotFoundException(
                ErrorCode.PIPE_EXPORT_FAILED,
                "Download link is invalid or has expired.",
            )

        session = await self.db.get(AstroSession, session_id)
        if session is None or not session.is_in_gallery:
            raise NotFoundException(
                ErrorCode.SESS_NOT_FOUND,
                "Gallery item not found.",
            )

        job = await self._latest_completed_job(session_id)
        if job is None:
            raise NotFoundException(
                ErrorCode.PIPE_EXPORT_FAILED,
                "No completed job available for this gallery item.",
            )

        attr = _FORMAT_ATTR[fmt]
        path_str = getattr(job, attr, None)
        if not path_str:
            raise NotFoundException(
                ErrorCode.PIPE_EXPORT_FAILED,
                f"{fmt.upper()} output not available.",
            )
        path = Path(path_str)
        if not path.exists():
            raise NotFoundException(
                ErrorCode.PIPE_EXPORT_FAILED,
                f"{fmt.upper()} file is missing on disk.",
            )

        session.gallery_download_count += 1
        self.db.add(session)
        await self.db.commit()
        return path

    # ── Internals ───────────────────────────────────────────────────

    @staticmethod
    def _enforce_rate_limit(key: str) -> None:
        """Slide the timestamp window for ``key`` and raise if over quota."""
        now = time.monotonic()
        bucket = _rate_buckets[key]
        cutoff = now - _RATE_LIMIT_WINDOW
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= _RATE_LIMIT_MAX:
            raise RateLimitExceeded()
        bucket.append(now)
