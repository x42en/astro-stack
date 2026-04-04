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
