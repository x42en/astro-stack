"""REST API endpoints for astrophotography sessions."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar

import aiofiles
from fastapi import APIRouter, Depends, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.middleware.auth import get_current_user
from app.core.config import get_settings
from app.core.database import get_async_session
from app.domain.job import ProfilePreset
from app.domain.session import AstroSession, InputFormat, SessionRead, SessionStatus
from app.infrastructure.storage.file_store import FileStore
from app.services.job_service import JobService
from app.services.session_service import SessionService

router = APIRouter(prefix="/sessions", tags=["sessions"])

T = TypeVar("T")


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response envelope."""

    items: list[T]
    total: int
    page: int
    page_size: int
    has_more: bool

    model_config = {"arbitrary_types_allowed": True}


@router.get(
    "",
    response_model=PaginatedResponse[SessionRead],
    summary="List all sessions",
    description="Returns a paginated list of sessions, optionally filtered by status or name.",
)
async def list_sessions(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=500),
    status: Optional[SessionStatus] = Query(default=None),
    search: Optional[str] = Query(default=None, description="Filter by name substring"),
    db: AsyncSession = Depends(get_async_session),
    _user: Optional[dict] = Depends(get_current_user),
) -> Any:
    """List sessions with optional status and name filters.

    Args:
        page: Page number (1-based).
        page_size: Number of items per page.
        status: Optional status filter.
        search: Optional name substring filter.
        db: Injected database session.
        _user: Injected auth user (None if auth disabled).

    Returns:
        :class:`PaginatedResponse` containing a page of sessions.
    """
    service = SessionService(db)
    offset = (page - 1) * page_size
    sessions = await service.list_sessions(offset=offset, limit=page_size, status=status, search=search)
    total = await service.count_sessions(status=status, search=search)
    items = [SessionRead.model_validate(s.model_dump()) for s in sessions]
    return PaginatedResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        has_more=(offset + len(sessions)) < total,
    )


@router.post(
    "/upload",
    response_model=dict,
    summary="Upload a file chunk for chunked session ingestion",
    description=(
        "Chunked multipart upload. Send one chunk per request using the custom headers. "
        "The first chunk (Upload-Chunk: 0) creates the session when no X-Session-ID is supplied. "
        "Set Upload-Finalize: true on the last chunk to reassemble the file and update frame counts."
    ),
)
async def upload_chunk(
    file: UploadFile = File(...),
    x_frame_type: str = Header(alias="X-Frame-Type"),
    x_file_name: str = Header(alias="X-File-Name"),
    x_session_id: Optional[str] = Header(default=None, alias="X-Session-ID"),
    x_session_name: Optional[str] = Header(default=None, alias="X-Session-Name"),
    x_object_name: Optional[str] = Header(default=None, alias="X-Object-Name"),
    x_upload_id: Optional[str] = Header(default=None, alias="X-Upload-ID"),
    upload_chunk: int = Header(alias="Upload-Chunk"),
    upload_total_chunks: int = Header(alias="Upload-Total-Chunks"),
    upload_finalize: Optional[str] = Header(default=None, alias="Upload-Finalize"),
    db: AsyncSession = Depends(get_async_session),
    _user: Optional[dict] = Depends(get_current_user),
) -> dict:
    """Accept a single chunk of a multi-part file upload.

    Args:
        file: The chunk data as a multipart file field.
        x_frame_type: Frame category (``lights``, ``darks``, ``flats``, ``bias``).
        x_file_name: Original file name (used for the assembled output file).
        x_session_id: Existing session UUID; omit to create a new session.
        x_session_name: Human-readable session name (required when creating).
        x_object_name: Optional target object name stored on the session.
        x_upload_id: Upload token from the first chunk response.
        upload_chunk: Zero-based index of this chunk.
        upload_total_chunks: Total number of chunks for this file.
        upload_finalize: Set to ``"true"`` on the last chunk to assemble the file.
        db: Injected database session.
        _user: Injected auth user.

    Returns:
        For non-finalising chunks: ``{"upload_id": ..., "session_id": ...}``.
        For the finalising chunk: ``{"session_id": ..., "file_path": ...}``.
    """
    settings = get_settings()
    service = SessionService(db)

    # ── Resolve or create session ─────────────────────────────────────────────
    if x_session_id:
        session = await service.get_or_404(uuid.UUID(x_session_id))
        session_uuid = session.id
        session_inbox_path = Path(session.inbox_path)
    else:
        if not x_session_name:
            raise HTTPException(
                status_code=400,
                detail="X-Session-Name header is required when creating a new session.",
            )
        session_uuid = uuid.uuid4()
        session_inbox_path = Path(settings.inbox_path) / str(session_uuid)
        new_session = AstroSession(
            id=session_uuid,
            name=x_session_name,
            inbox_path=str(session_inbox_path),
            status=SessionStatus.PENDING,
            input_format=None,
            object_name=x_object_name or None,
        )
        await service._session_repo.create(new_session)

    # ── Persist chunk to temp directory ──────────────────────────────────────
    upload_id = x_upload_id or str(uuid.uuid4())
    uploads_root = Path(settings.sessions_path) / "uploads" / upload_id
    uploads_root.mkdir(parents=True, exist_ok=True)

    chunk_data = await file.read()
    chunk_path = uploads_root / f"chunk_{upload_chunk:05d}"
    async with aiofiles.open(chunk_path, "wb") as fh:
        await fh.write(chunk_data)

    # ── Finalise: reassemble file and update session ──────────────────────────
    if upload_finalize == "true":
        frame_dir = session_inbox_path / x_frame_type
        frame_dir.mkdir(parents=True, exist_ok=True)

        safe_filename = Path(x_file_name).name  # strip any path traversal
        target_path = frame_dir / safe_filename
        async with aiofiles.open(target_path, "wb") as out:
            for i in range(upload_total_chunks):
                part_path = uploads_root / f"chunk_{i:05d}"
                async with aiofiles.open(part_path, "rb") as part:
                    await out.write(await part.read())

        # Clean up temp chunks (best effort)
        shutil.rmtree(str(uploads_root), ignore_errors=True)

        # Refresh frame counts from the session directory
        file_store = service._file_store
        frames = file_store.discover_frames(session_inbox_path)
        detected_format = file_store.detect_input_format(frames)
        await service._session_repo.update(
            session_uuid,
            {
                "frame_count_lights": len(frames["lights"]),
                "frame_count_darks": len(frames["darks"]),
                "frame_count_flats": len(frames["flats"]),
                "frame_count_bias": len(frames["bias"]),
                "input_format": InputFormat(detected_format) if frames["lights"] else None,
                "status": SessionStatus.READY if len(frames["lights"]) > 0 else SessionStatus.PENDING,
            },
        )

        return {"session_id": str(session_uuid), "file_path": str(target_path)}

    return {"upload_id": upload_id, "session_id": str(session_uuid)}


@router.get(
    "/{session_id}",
    response_model=SessionRead,
    summary="Get session details",
)
async def get_session(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
    _user: Optional[dict] = Depends(get_current_user),
) -> SessionRead:
    """Retrieve a session by ID.

    Args:
        session_id: Session UUID.
        db: Injected database session.
        _user: Injected auth user.

    Returns:
        :class:`~app.domain.session.SessionRead` for the found session.
    """
    service = SessionService(db)
    session = await service.get_or_404(session_id)
    return SessionRead.model_validate(session.model_dump())


@router.post(
    "/{session_id}/process",
    response_model=dict,
    status_code=202,
    summary="Start pipeline processing",
    description="Enqueue a pipeline job for the specified session.",
)
async def start_processing(
    session_id: uuid.UUID,
    preset: ProfilePreset = Query(
        default=ProfilePreset.STANDARD,
        description="Processing profile preset to use.",
    ),
    profile_id: Optional[uuid.UUID] = Query(
        default=None,
        description="UUID of a saved advanced profile (required for ADVANCED preset).",
    ),
    db: AsyncSession = Depends(get_async_session),
    _user: Optional[dict] = Depends(get_current_user),
) -> dict:
    """Enqueue a pipeline job for a session.

    Args:
        session_id: Session UUID to process.
        preset: Processing preset (quick/standard/quality/advanced).
        profile_id: Saved profile UUID for ADVANCED preset.
        db: Injected database session.
        _user: Injected auth user.

    Returns:
        Dict with ``job_id`` of the created pipeline job.
    """
    service = JobService(db)
    job = await service.start_pipeline(
        session_id=session_id,
        preset=preset,
        profile_id=profile_id,
    )
    return {"job_id": str(job.id), "status": job.status}


@router.post(
    "/{session_id}/cancel",
    response_model=dict,
    summary="Cancel active pipeline job",
)
async def cancel_processing(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
    _user: Optional[dict] = Depends(get_current_user),
) -> dict:
    """Cancel the active pipeline job for a session.

    Args:
        session_id: Session UUID.
        db: Injected database session.
        _user: Injected auth user.

    Returns:
        Dict with ``message`` confirming cancellation request.
    """
    job_service = JobService(db)
    sess_service = SessionService(db)
    session = await sess_service.get_or_404(session_id)

    # Find the active job for this session
    from app.infrastructure.repositories.job_repo import JobRepository  # noqa: PLC0415

    job_repo = JobRepository(db)
    active_job = await job_repo.get_active_job_for_session(session_id)

    if active_job is None:
        return {"message": "No active job to cancel for this session."}

    await job_service.cancel_job(active_job.id)
    return {"message": f"Cancellation requested for job {active_job.id}."}


@router.delete(
    "/{session_id}/scratch",
    status_code=204,
    summary="Delete session scratch files",
    description=(
        "Remove the working directory (calibrated, registered, process/ subdirs) "
        "for a session. Original frames in /inbox and final renders in /output "
        "are never touched. Returns 409 if a pipeline job is currently running."
    ),
)
async def delete_scratch(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
    _user: Optional[dict] = Depends(get_current_user),
) -> None:
    """Delete intermediate pipeline files for a session.

    Args:
        session_id: Session UUID.
        db: Injected database session.
        _user: Injected auth user.
    """
    sess_service = SessionService(db)
    await sess_service.get_or_404(session_id)

    from app.infrastructure.repositories.job_repo import JobRepository  # noqa: PLC0415

    job_repo = JobRepository(db)
    active_job = await job_repo.get_active_job_for_session(session_id)

    if active_job is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Pipeline job {active_job.id} is currently running. Cancel it before deleting scratch files.",
        )

    file_store = FileStore()
    await file_store.cleanup_work_dir(session_id)


@router.get(
    "/{session_id}/step-preview/{step_name}",
    response_class=FileResponse,
    summary="Get a per-step JPEG preview",
    description=(
        "Returns the JPEG preview generated after a specific pipeline step. "
        "Returns 404 if the step has not yet produced a preview."
    ),
)
async def get_step_preview(
    session_id: uuid.UUID,
    step_name: str,
    _user: Optional[dict] = Depends(get_current_user),
) -> FileResponse:
    """Serve a per-step JPEG preview image.

    Args:
        session_id: Session UUID.
        step_name: Machine-readable step name (e.g. ``preprocessing``).
        _user: Injected auth user.

    Returns:
        JPEG image as a :class:`~fastapi.responses.FileResponse`.

    Raises:
        HTTPException: 404 if the preview does not exist yet.
    """
    file_store = FileStore()
    preview_path = file_store.step_preview_path(session_id, step_name)
    if not preview_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No preview available for step '{step_name}'.",
        )
    return FileResponse(str(preview_path), media_type="image/jpeg")


# ── Light-frame preview ───────────────────────────────────────────────────────

def _generate_light_preview(frame_path: Path, output_path: Path) -> None:
    """Generate a JPEG preview from a single light frame (sync, run in thread).

    Supports FITS files (via astropy) and RAW/DSLR files (via rawpy).
    The image is stretched to enhance visibility before saving.

    Args:
        frame_path: Absolute path to the source light frame.
        output_path: Destination path for the generated JPEG.

    Raises:
        RuntimeError: If the file format is not supported or data cannot be read.
    """
    import io as _io  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = frame_path.suffix.lower()

    if suffix in {".fit", ".fits", ".fts"}:
        from astropy.io import fits as astropy_fits  # noqa: PLC0415

        with astropy_fits.open(str(frame_path)) as hdul:
            # Find the first HDU with 2-D (or 3-D) image data
            data: np.ndarray | None = None
            for hdu in hdul:
                if hdu.data is not None and hdu.data.ndim >= 2:
                    data = hdu.data.astype(np.float32)
                    break
            if data is None:
                raise RuntimeError("No image data found in FITS file.")

        # Collapse colour axis if present (take first channel)
        if data.ndim == 3:
            data = data[0]

        # Percentile stretch for visibility
        lo, hi = np.percentile(data, [1.0, 99.5])
        data = np.clip((data - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        img = Image.fromarray((data * 255).astype(np.uint8), mode="L").convert("RGB")

    else:
        # RAW DSLR files
        import rawpy  # noqa: PLC0415

        with rawpy.imread(str(frame_path)) as raw:
            rgb = raw.postprocess(use_camera_wb=True, output_bps=8)
        img = Image.fromarray(rgb)

    # Downscale to a sensible thumbnail size (max 800 px wide)
    img.thumbnail((800, 800), Image.LANCZOS)
    buf = _io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    output_path.write_bytes(buf.getvalue())


@router.get(
    "/{session_id}/light-preview",
    response_class=FileResponse,
    summary="Get a light-frame thumbnail preview",
    description=(
        "Returns a JPEG thumbnail generated from the first discovered light frame. "
        "The image is generated on demand and cached permanently. "
        "Returns 404 if no light frames are available."
    ),
)
async def get_light_preview(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
    _user: Optional[dict] = Depends(get_current_user),
) -> FileResponse:
    """Serve a light-frame thumbnail for session cards.

    Args:
        session_id: Session UUID.
        db: Async database session.
        _user: Injected auth user.

    Returns:
        JPEG image as a :class:`~fastapi.responses.FileResponse`.

    Raises:
        HTTPException: 404 if no light frames exist or the session is not found.
    """
    import asyncio  # noqa: PLC0415

    file_store = FileStore()
    cached = file_store.light_preview_path(session_id)
    if cached.exists():
        return FileResponse(str(cached), media_type="image/jpeg")

    session_svc = SessionService(db)
    session = await session_svc.get_or_404(session_id)
    frames = file_store.discover_frames(Path(session.inbox_path))
    lights = frames["lights"]
    if not lights:
        raise HTTPException(status_code=404, detail="No light frames found for this session.")

    await asyncio.to_thread(_generate_light_preview, lights[0], cached)
    return FileResponse(str(cached), media_type="image/jpeg")
