"""REST API endpoints for pipeline jobs."""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.middleware.auth import get_current_user
from app.core.database import get_async_session
from app.core.errors import ErrorCode, NotFoundException
from app.domain.job import JobRead
from app.services.job_service import JobService

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get(
    "/{job_id}",
    response_model=JobRead,
    summary="Get job status and step results",
)
async def get_job(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
    _user: Optional[dict] = Depends(get_current_user),
) -> JobRead:
    """Retrieve detailed job status including all step results.

    Args:
        job_id: Pipeline job UUID.
        db: Injected database session.
        _user: Injected auth user.

    Returns:
        :class:`~app.domain.job.JobRead` with step details.
    """
    service = JobService(db)
    return await service.get_job_with_steps(job_id)


@router.get(
    "/{job_id}/output/preview",
    summary="Download JPEG preview image",
)
async def download_preview(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
    _user: Optional[dict] = Depends(get_current_user),
) -> FileResponse:
    """Stream the JPEG preview image for a completed job.

    Args:
        job_id: Pipeline job UUID.
        db: Injected database session.
        _user: Injected auth user.

    Returns:
        :class:`~fastapi.responses.FileResponse` with the JPEG file.
    """
    from pathlib import Path  # noqa: PLC0415

    service = JobService(db)
    job_read = await service.get_job_with_steps(job_id)

    if not job_read.output_preview_path:
        raise NotFoundException(
            ErrorCode.PIPE_EXPORT_FAILED,
            f"Preview image not available for job '{job_id}'.",
        )

    return FileResponse(
        path=job_read.output_preview_path,
        media_type="image/jpeg",
        filename=f"preview_{job_id}.jpg",
    )


@router.get(
    "/{job_id}/output/fits",
    summary="Download final FITS file",
)
async def download_fits(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
    _user: Optional[dict] = Depends(get_current_user),
) -> FileResponse:
    """Stream the final FITS output file for a completed job.

    Args:
        job_id: Pipeline job UUID.
        db: Injected database session.
        _user: Injected auth user.

    Returns:
        :class:`~fastapi.responses.FileResponse` with the FITS file.
    """
    service = JobService(db)
    job_read = await service.get_job_with_steps(job_id)

    if not job_read.output_fits_path:
        raise NotFoundException(
            ErrorCode.PIPE_EXPORT_FAILED,
            f"FITS output not available for job '{job_id}'.",
        )

    return FileResponse(
        path=job_read.output_fits_path,
        media_type="application/octet-stream",
        filename=f"final_{job_id}.fits",
    )


@router.get(
    "/{job_id}/output/tiff",
    summary="Download high-quality 16-bit TIFF",
)
async def download_tiff(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
    _user: Optional[dict] = Depends(get_current_user),
) -> FileResponse:
    """Stream the final 16-bit TIFF rendition for a completed job.

    The TIFF is produced from the same processed image as the JPEG preview,
    with the export-time HDR polish applied (midtone S-curve, highlight
    rolloff, and saturation boost). 16-bit depth + Deflate compression keeps
    the file editable in Photoshop, GIMP, Krita, Affinity Photo, and Apple
    Preview / Windows Photos without further conversion.

    Args:
        job_id: Pipeline job UUID.
        db: Injected database session.
        _user: Injected auth user.

    Returns:
        :class:`~fastapi.responses.FileResponse` with the TIFF file.
    """
    service = JobService(db)
    job_read = await service.get_job_with_steps(job_id)

    if not job_read.output_tiff_path:
        raise NotFoundException(
            ErrorCode.PIPE_EXPORT_FAILED,
            f"TIFF output not available for job '{job_id}'.",
        )

    return FileResponse(
        path=job_read.output_tiff_path,
        media_type="image/tiff",
        filename=f"final_{job_id}.tiff",
    )
