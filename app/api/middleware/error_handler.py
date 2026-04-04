"""Global exception handler for structured JSON error responses.

Registers a FastAPI exception handler that converts all
:class:`~app.core.errors.AstroStackException` subclasses into a consistent
JSON payload with an ``error_code`` field for i18n support.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.errors import AstroStackException
from app.core.logging import get_logger

logger = get_logger(__name__)


def register_error_handlers(app: FastAPI) -> None:
    """Attach global exception handlers to the FastAPI application.

    Args:
        app: The FastAPI application instance.
    """

    @app.exception_handler(AstroStackException)
    async def astro_stack_exception_handler(
        request: Request,
        exc: AstroStackException,
    ) -> JSONResponse:
        """Handle all :class:`~app.core.errors.AstroStackException` subclasses.

        Args:
            request: The incoming HTTP request.
            exc: The raised exception.

        Returns:
            A :class:`~fastapi.responses.JSONResponse` with status code and
            structured error body.
        """
        logger.warning(
            "api_error",
            error_code=exc.error_code.value,
            message=exc.message,
            path=str(request.url),
            status_code=exc.status_code,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error_code": exc.error_code.value,
                "message": exc.message,
                "details": exc.details,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            },
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        """Catch-all handler for unexpected errors (HTTP 500).

        Args:
            request: The incoming HTTP request.
            exc: The unhandled exception.

        Returns:
            A generic 500 JSON error response.
        """
        logger.exception(
            "unhandled_exception",
            path=str(request.url),
            exc_info=exc,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error_code": "SYS_INTERNAL_ERROR",
                "message": "An unexpected internal error occurred.",
                "details": {},
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            },
        )
