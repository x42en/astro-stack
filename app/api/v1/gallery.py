"""Public gallery REST endpoints.

Two audiences:

* **Owner** (authenticated): publish/unpublish a session.
* **Anyone** (anonymous): browse the gallery, request a download link
  with their email, and follow the signed URL to actually fetch the file.

The download flow is two-step on purpose: the request endpoint enforces
rate-limiting and audits the email, while the download endpoint stays
stateless and only validates the HMAC token.  This means a leaked
gallery URL cannot trigger downloads on its own — you must POST your
email first.
"""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.middleware.auth import get_current_user
from app.core.database import get_async_session
from app.services.gallery_service import GalleryService

router = APIRouter(tags=["gallery"])


# ────────────────────────── Schemas ──────────────────────────────────


class GalleryItemRead(BaseModel):
    """Public payload for a gallery card."""

    session_id: uuid.UUID
    job_id: uuid.UUID
    name: str
    object_name: Optional[str]
    author_name: Optional[str]
    published_at: Optional[str]
    acquired_at: Optional[str]
    download_count: int
    preview_url: str


class PublishRequest(BaseModel):
    """Optional author override at publish time."""

    author_name: Optional[str] = Field(default=None, max_length=120)


class DownloadRequestBody(BaseModel):
    """Email + format submitted by the gallery viewer."""

    email: EmailStr
    format: str = Field(pattern="^(tiff|fits)$")


class DownloadRequestResponse(BaseModel):
    download_url: str
    expires_at: int


# ────────────────────────── Helpers ──────────────────────────────────


def _build_item(item, request: Request) -> GalleryItemRead:
    base = str(request.base_url).rstrip("/")
    return GalleryItemRead(
        session_id=item.session.id,
        job_id=item.job_id,
        name=item.session.name,
        object_name=item.session.object_name,
        author_name=item.session.gallery_author_name,
        published_at=(
            item.session.gallery_published_at.isoformat()
            if item.session.gallery_published_at
            else None
        ),
        acquired_at=(
            item.session.acquired_at.isoformat()
            if item.session.acquired_at
            else None
        ),
        download_count=item.session.gallery_download_count,
        preview_url=f"{base}/api/v1/gallery/{item.session.id}/preview",
    )


# ────────────────────────── Public endpoints ─────────────────────────


@router.get(
    "/gallery",
    response_model=list[GalleryItemRead],
    summary="List published gallery sessions (public)",
)
async def list_gallery(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(60, ge=1, le=120),
    db: AsyncSession = Depends(get_async_session),
) -> list[GalleryItemRead]:
    """Anonymous listing of all published sessions, newest first."""
    service = GalleryService(db)
    items = await service.list_public(
        offset=(page - 1) * page_size, limit=page_size
    )
    return [_build_item(it, request) for it in items]


@router.get(
    "/gallery/{session_id}/preview",
    summary="Public preview image for a gallery session",
)
async def gallery_preview(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
) -> FileResponse:
    """Serve the JPEG preview of a published session (no auth)."""
    service = GalleryService(db)
    items = await service.list_public(offset=0, limit=10_000)
    for item in items:
        if item.session.id == session_id:
            return FileResponse(item.preview_path, media_type="image/jpeg")
    from app.core.errors import ErrorCode, NotFoundException

    raise NotFoundException(
        ErrorCode.SESS_NOT_FOUND, "Gallery item not found."
    )


@router.post(
    "/gallery/{session_id}/request-download",
    response_model=DownloadRequestResponse,
    summary="Request a signed download URL for a gallery session",
)
async def request_download(
    session_id: uuid.UUID,
    body: DownloadRequestBody,
    request: Request,
    db: AsyncSession = Depends(get_async_session),
) -> DownloadRequestResponse:
    """Audit the email and return a short-lived signed URL."""
    service = GalleryService(db)
    client_ip = request.client.host if request.client else None
    token, exp = await service.request_download(
        session_id=session_id,
        email=body.email,
        fmt=body.format,
        requester_ip=client_ip,
    )
    base = str(request.base_url).rstrip("/")
    url = (
        f"{base}/api/v1/gallery/{session_id}/download"
        f"?token={token}&format={body.format}"
    )
    return DownloadRequestResponse(download_url=url, expires_at=exp)


@router.get(
    "/gallery/{session_id}/download",
    summary="Stream a gallery file using a signed token",
)
async def download_gallery_file(
    session_id: uuid.UUID,
    token: str = Query(...),
    format: str = Query(..., pattern="^(tiff|fits)$"),
    db: AsyncSession = Depends(get_async_session),
) -> FileResponse:
    """Validate ``token`` and stream the requested file."""
    service = GalleryService(db)
    path = await service.serve_download(session_id, format, token)
    media = "image/tiff" if format == "tiff" else "application/octet-stream"
    return FileResponse(
        path=str(path),
        media_type=media,
        filename=f"astrostack_{session_id}.{format}",
    )


# ────────────────────────── Owner endpoints ──────────────────────────


@router.post(
    "/sessions/{session_id}/publish",
    summary="Publish a completed session to the public gallery",
)
async def publish_session(
    session_id: uuid.UUID,
    body: PublishRequest = PublishRequest(),
    db: AsyncSession = Depends(get_async_session),
    _user: Optional[dict] = Depends(get_current_user),
) -> dict:
    service = GalleryService(db)
    session = await service.publish(session_id, author_name=body.author_name)
    return {
        "session_id": str(session.id),
        "is_in_gallery": session.is_in_gallery,
        "gallery_published_at": (
            session.gallery_published_at.isoformat()
            if session.gallery_published_at
            else None
        ),
    }


@router.delete(
    "/sessions/{session_id}/publish",
    summary="Remove a session from the public gallery",
)
async def unpublish_session(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
    _user: Optional[dict] = Depends(get_current_user),
) -> dict:
    service = GalleryService(db)
    session = await service.unpublish(session_id)
    return {
        "session_id": str(session.id),
        "is_in_gallery": session.is_in_gallery,
    }
