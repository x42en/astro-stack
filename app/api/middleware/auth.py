"""JWT authentication middleware and helpers.

Authentication is optional and can be toggled via the ``AUTH_ENABLED``
environment variable. When disabled, token validation is a no-op — allowing
development and initial deployment without authentication overhead.

When enabled, tokens are HS256 JWT signed with ``JWT_SECRET``. The
``sub`` claim is used as the user identifier.

WebSocket tokens are passed as a ``?token=<jwt>`` query parameter since
browsers cannot set ``Authorization`` headers on WebSocket connections.
"""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import get_settings
from app.core.errors import AuthException, ErrorCode
from app.core.logging import get_logger

logger = get_logger(__name__)

_bearer_scheme = HTTPBearer(auto_error=False)


def _decode_token(token: str) -> dict:
    """Decode and validate a JWT token.

    Args:
        token: Raw JWT string.

    Returns:
        Decoded payload dict.

    Raises:
        AuthException: If the token is invalid or expired.
    """
    try:
        from jose import JWTError, jwt  # noqa: PLC0415

        settings = get_settings()
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
        )
        return payload
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).lower()
        if "expired" in msg:
            raise AuthException(
                ErrorCode.AUTH_TOKEN_EXPIRED,
                "JWT token has expired.",
                status_code=401,
            ) from exc
        raise AuthException(
            ErrorCode.AUTH_TOKEN_INVALID,
            f"JWT token is invalid: {exc}",
            status_code=401,
        ) from exc


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> Optional[dict]:
    """FastAPI dependency: validate the Bearer token and return the payload.

    When ``AUTH_ENABLED=false``, always returns ``None`` without validation.

    Args:
        credentials: Bearer credentials extracted by the HTTPBearer scheme.

    Returns:
        Decoded JWT payload dict when auth is enabled and token is valid,
        or ``None`` when auth is disabled.

    Raises:
        HTTPException: 401 if auth is enabled but no/invalid token provided.
    """
    settings = get_settings()
    if not settings.auth_enabled:
        return None

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error_code": ErrorCode.AUTH_REQUIRED.value,
                "message": "Authentication is required. Provide a Bearer token.",
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        return _decode_token(credentials.credentials)
    except AuthException as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.to_dict(),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


async def validate_optional_token(token: Optional[str]) -> Optional[dict]:
    """Validate a WebSocket query-param token (optional when auth is disabled).

    Args:
        token: Raw JWT string from ``?token=<jwt>`` query parameter, or ``None``.

    Returns:
        Decoded JWT payload when auth is enabled and token is valid,
        or ``None`` when auth is disabled or no token provided.

    Raises:
        AuthException: If auth is enabled and the provided token is invalid.
    """
    settings = get_settings()
    if not settings.auth_enabled:
        return None

    if token is None:
        raise AuthException(
            ErrorCode.AUTH_REQUIRED,
            "WebSocket authentication required. Provide ?token=<jwt>.",
            status_code=401,
        )

    return _decode_token(token)


async def extract_ws_token(token: Optional[str]) -> Optional[str]:
    """Extract user ID from a WebSocket token for logging/audit purposes.

    Args:
        token: Raw JWT string or ``None``.

    Returns:
        The ``sub`` claim value as a string, or ``None`` if unavailable.
    """
    settings = get_settings()
    if not settings.auth_enabled or token is None:
        return None
    try:
        payload = _decode_token(token)
        return str(payload.get("sub", "unknown"))
    except AuthException:
        return None


def _mock_user_to_uuid(value: str) -> uuid.UUID:
    """Map a free-text mock username to a deterministic UUID.

    Uses :func:`uuid.uuid5` with the configured namespace so the same username
    always resolves to the same persistence key. When real auth lands, a
    one-shot data migration can rewrite owner_user_id columns by reapplying
    this mapping to the legacy mock-user table.
    """
    settings = get_settings()
    namespace = uuid.UUID(settings.mock_user_namespace)
    return uuid.uuid5(namespace, value)


async def get_user_id_or_mock(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> uuid.UUID:
    """Return the caller's user UUID, supporting the mock-auth bridge.

    Resolution order:
        1. ``AUTH_ENABLED=true`` → validate the Bearer JWT, return ``UUID(sub)``.
        2. ``AUTH_ENABLED=false`` and ``X-Mock-User`` header present →
           ``uuid5(namespace, header_value)``.
        3. Otherwise → 401.

    Raises:
        HTTPException: 401 when no usable identity is available.
    """
    settings = get_settings()

    if settings.auth_enabled:
        if credentials is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "error_code": ErrorCode.AUTH_REQUIRED.value,
                    "message": "Authentication is required. Provide a Bearer token.",
                },
                headers={"WWW-Authenticate": "Bearer"},
            )
        payload = _decode_token(credentials.credentials)
        sub = payload.get("sub")
        if sub is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "error_code": ErrorCode.AUTH_TOKEN_INVALID.value,
                    "message": "JWT 'sub' claim is missing.",
                },
            )
        try:
            return uuid.UUID(str(sub))
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "error_code": ErrorCode.AUTH_TOKEN_INVALID.value,
                    "message": "JWT 'sub' claim is not a valid UUID.",
                },
            ) from exc

    mock_value = request.headers.get(settings.mock_user_header)
    if mock_value:
        return _mock_user_to_uuid(mock_value.strip())

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={
            "error_code": ErrorCode.AUTH_REQUIRED.value,
            "message": (
                "User identity required. Send a Bearer token (auth-enabled) "
                f"or '{settings.mock_user_header}' header (mock auth)."
            ),
        },
    )


async def get_optional_user_id(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> Optional[uuid.UUID]:
    """Like :func:`get_user_id_or_mock` but returns ``None`` when anonymous.

    Useful for public endpoints that adapt their response if the caller
    happens to be authenticated.
    """
    settings = get_settings()

    if settings.auth_enabled:
        if credentials is None:
            return None
        try:
            payload = _decode_token(credentials.credentials)
        except AuthException:
            return None
        sub = payload.get("sub")
        if sub is None:
            return None
        try:
            return uuid.UUID(str(sub))
        except (TypeError, ValueError):
            return None

    mock_value = request.headers.get(settings.mock_user_header)
    if mock_value:
        return _mock_user_to_uuid(mock_value.strip())
    return None
