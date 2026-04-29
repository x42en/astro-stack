"""OIDC / JWT authentication middleware and FastAPI dependency helpers.

Supports three operating modes controlled by the ``AUTH_MODE`` environment
variable (or derived from the legacy ``AUTH_ENABLED`` flag):

* ``disabled`` — All auth checks are bypassed.  Useful for local development
                  without an auth server.
* ``mock``     — HS256 JWT tokens signed with ``JWT_SECRET``, plus the
                  ``X-Mock-User`` header bridge that maps a free-text name to a
                  deterministic UUID for per-user data persistence during dev.
* ``oidc``     — OAuth 2.1 / OIDC tokens from an external provider
                  (default: ``auth.astromote.com``).  Tokens are validated
                  against the provider's public JWKS (RS256 / ES256).  The
                  ``sub`` claim is used as the owner UUID.  Roles and
                  permissions come from the ``roles`` / ``permissions`` OIDC
                  claims injected by AuthService.

WebSocket tokens are passed as ``?token=<value>`` query parameters (browsers
cannot set ``Authorization`` headers on WebSocket upgrades).  The value can
be either a full JWT **or** a short-lived single-use ticket (prefix ``wst_``)
issued by ``POST /api/v1/auth/ws-ticket``.

Public FastAPI dependencies (use via ``Depends``):

* :func:`get_current_user` — Optional raw claims dict (None when disabled).
* :func:`get_user_id_or_mock` — Mandatory user UUID; falls back to mock header.
* :func:`get_optional_user_id` — Optional user UUID; None for anonymous.
* :func:`validate_optional_token` — Validate a WS query-param token/ticket.
* :func:`require_role` — Dependency factory asserting a role claim.
* :func:`require_permission` — Dependency factory asserting a permission claim.
* :func:`prefetch_jwks` — Async helper called from the app lifespan.
* :func:`issue_ws_ticket` — Mint a single-use Redis-backed WS ticket.
"""

from __future__ import annotations

import asyncio
import secrets
import time
import uuid
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Optional

import httpx
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import get_settings
from app.core.errors import AuthException, ErrorCode
from app.core.logging import get_logger

logger = get_logger(__name__)

_bearer_scheme = HTTPBearer(auto_error=False)

# ---------------------------------------------------------------------------
# Auth identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuthIdentity:
    """Normalised caller identity extracted from any supported token type."""

    user_id: uuid.UUID
    email: Optional[str] = None
    name: Optional[str] = None
    roles: tuple[str, ...] = dc_field(default_factory=tuple)
    permissions: tuple[str, ...] = dc_field(default_factory=tuple)
    raw_claims: dict = dc_field(default_factory=dict)


# ---------------------------------------------------------------------------
# JWKS cache
# ---------------------------------------------------------------------------


class _JWKSCache:
    """Async JWKS key cache with TTL and automatic kid-miss refetch."""

    def __init__(self) -> None:
        self._keys: dict[Optional[str], dict] = {}
        self._fetched_at: float = 0.0
        self._lock: asyncio.Lock = asyncio.Lock()

    async def get_key(self, kid: Optional[str], jwks_url: str, ttl: int) -> dict:
        """Return the JWK matching *kid*, fetching / refreshing as needed.

        On a kid miss after a fresh fetch the caller receives an
        ``AUTH_TOKEN_INVALID`` exception — the token's signing key is genuinely
        unknown and the request must be rejected.
        """
        async with self._lock:
            now = time.monotonic()
            if now - self._fetched_at > ttl or kid not in self._keys:
                await self._fetch(jwks_url)
            if kid not in self._keys:
                raise AuthException(
                    ErrorCode.AUTH_TOKEN_INVALID,
                    f"Signing key '{kid}' not found in JWKS after refresh.",
                    status_code=401,
                )
        return self._keys[kid]

    async def prefetch(self, jwks_url: str) -> None:
        """Eagerly warm the cache (called at application startup)."""
        async with self._lock:
            await self._fetch(jwks_url)

    async def _fetch(self, jwks_url: str) -> None:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(jwks_url)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("jwks_fetch_failed", url=jwks_url, error=str(exc))
            # Retain stale keys so in-flight requests can still validate
            return
        new_keys: dict[Optional[str], dict] = {}
        for key in data.get("keys", []):
            new_keys[key.get("kid")] = key
        self._keys = new_keys
        self._fetched_at = time.monotonic()
        logger.info("jwks_refreshed", key_count=len(new_keys), url=jwks_url)


_jwks_cache = _JWKSCache()

# ---------------------------------------------------------------------------
# WS ticket store (Redis-backed, lazy-initialised)
# ---------------------------------------------------------------------------

_TICKET_PREFIX = "ws_ticket:"
_TICKET_TTL_SECONDS = 30

_redis_client = None


def _get_ticket_redis():  # type: ignore[return]
    """Return a lazily-initialised async Redis client used for WS tickets."""
    global _redis_client  # noqa: PLW0603
    if _redis_client is None:
        import redis.asyncio as aioredis  # noqa: PLC0415

        settings = get_settings()
        _redis_client = aioredis.from_url(settings.redis_url_str, decode_responses=True)
    return _redis_client


async def issue_ws_ticket(user_id: uuid.UUID) -> str:
    """Mint a single-use 30-second WS authentication ticket for *user_id*.

    The ticket is stored in Redis with a 30-second TTL and is consumed
    atomically on first WebSocket handshake, ensuring single-use semantics.

    Returns:
        Raw ticket string (``wst_<random>``).
    """
    ticket = f"wst_{secrets.token_urlsafe(32)}"
    redis = _get_ticket_redis()
    await redis.set(f"{_TICKET_PREFIX}{ticket}", str(user_id), ex=_TICKET_TTL_SECONDS)
    logger.info("ws_ticket_issued", user_id=str(user_id))
    return ticket


async def _consume_ws_ticket(ticket: str) -> uuid.UUID:
    """Validate and atomically consume a WS ticket.

    Raises:
        AuthException: If the ticket is unknown, expired, or already used.
    """
    redis = _get_ticket_redis()
    key = f"{_TICKET_PREFIX}{ticket}"
    user_id_str = await redis.getdel(key)
    if user_id_str is None:
        raise AuthException(
            ErrorCode.AUTH_TOKEN_INVALID,
            "WebSocket ticket is invalid, expired, or already used.",
            status_code=401,
        )
    return uuid.UUID(user_id_str)


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------


def _subject_to_uuid(sub: str) -> uuid.UUID:
    """Convert an OIDC ``sub`` claim to a UUID.

    BetterAuth issues UUID-shaped subs directly.  For non-UUID subs (future
    providers) we derive a deterministic UUID via uuid5 using the mock
    namespace so that the data model stays consistent across auth modes.
    """
    try:
        return uuid.UUID(sub)
    except (ValueError, AttributeError):
        settings = get_settings()
        namespace = uuid.UUID(settings.mock_user_namespace)
        return uuid.uuid5(namespace, sub)


async def _verify_oidc_token(token: str) -> AuthIdentity:
    """Validate an OIDC JWT against the provider's JWKS (RS256 / ES256).

    Raises:
        AuthException: On any validation failure (expired, bad sig, bad iss…).
    """
    from jose import JWTError, jwt  # noqa: PLC0415
    from jose.exceptions import ExpiredSignatureError  # noqa: PLC0415

    settings = get_settings()
    jwks_url = settings.oidc_jwks_endpoint
    if not jwks_url:
        raise AuthException(
            ErrorCode.AUTH_TOKEN_INVALID,
            "OIDC mode is active but OIDC_ISSUER is not configured on the server.",
            status_code=500,
        )

    try:
        header = jwt.get_unverified_header(token)
    except JWTError as exc:
        raise AuthException(
            ErrorCode.AUTH_TOKEN_INVALID,
            f"Malformed JWT header: {exc}",
            status_code=401,
        ) from exc

    kid: Optional[str] = header.get("kid")
    alg: str = header.get("alg", "RS256")
    key = await _jwks_cache.get_key(kid, jwks_url, settings.oidc_jwks_cache_ttl_seconds)

    try:
        payload = jwt.decode(
            token,
            key,
            algorithms=[alg, "RS256", "ES256"],
            audience=settings.oidc_audience or None,
            issuer=settings.oidc_issuer,
            options={"leeway": 60},
        )
    except ExpiredSignatureError as exc:
        raise AuthException(
            ErrorCode.AUTH_TOKEN_EXPIRED, "JWT token has expired.", status_code=401
        ) from exc
    except JWTError as exc:
        raise AuthException(
            ErrorCode.AUTH_TOKEN_INVALID,
            f"JWT validation failed: {exc}",
            status_code=401,
        ) from exc

    sub = payload.get("sub")
    if not sub:
        raise AuthException(
            ErrorCode.AUTH_TOKEN_INVALID, "JWT 'sub' claim is missing.", status_code=401
        )

    return AuthIdentity(
        user_id=_subject_to_uuid(str(sub)),
        email=payload.get("email"),
        name=payload.get("name"),
        roles=tuple(payload.get("roles", [])),
        permissions=tuple(payload.get("permissions", [])),
        raw_claims=payload,
    )


def _decode_token(token: str) -> dict:
    """Decode and validate an HS256 JWT (mock mode only).

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
                ErrorCode.AUTH_TOKEN_EXPIRED, "JWT token has expired.", status_code=401
            ) from exc
        raise AuthException(
            ErrorCode.AUTH_TOKEN_INVALID,
            f"JWT token is invalid: {exc}",
            status_code=401,
        ) from exc


async def _resolve_identity(token: str) -> AuthIdentity:
    """Resolve a Bearer token to an AuthIdentity, dispatching on auth mode."""
    settings = get_settings()
    mode = settings.effective_auth_mode

    if mode == "oidc":
        return await _verify_oidc_token(token)

    # mock mode — HS256 JWT
    payload = _decode_token(token)
    sub = str(payload.get("sub", ""))
    return AuthIdentity(
        user_id=_subject_to_uuid(sub) if sub else uuid.uuid4(),
        email=payload.get("email"),
        name=payload.get("preferred_username") or payload.get("name"),
        roles=tuple(payload.get("roles", [])),
        permissions=tuple(payload.get("permissions", [])),
        raw_claims=payload,
    )


# ---------------------------------------------------------------------------
# Mock-auth bridge
# ---------------------------------------------------------------------------


def _mock_user_to_uuid(value: str) -> uuid.UUID:
    """Map a free-text mock username to a deterministic UUID.

    Uses :func:`uuid.uuid5` with the configured namespace so the same username
    always resolves to the same persistence key.  When migrating to real OIDC
    a one-shot data migration can rewrite ``owner_user_id`` columns by
    re-applying this mapping against the legacy mock-user table.
    """
    settings = get_settings()
    namespace = uuid.UUID(settings.mock_user_namespace)
    return uuid.uuid5(namespace, value)


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> Optional[dict]:
    """FastAPI dependency: validate Bearer token and return the raw claims dict.

    When ``effective_auth_mode=disabled``, always returns ``None`` without
    validation.

    Returns:
        Decoded claims dict when auth is enabled and token is valid,
        or ``None`` when auth is disabled.

    Raises:
        HTTPException: 401 if auth is enabled but no/invalid token provided.
    """
    settings = get_settings()
    mode = settings.effective_auth_mode

    if mode == "disabled":
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
        identity = await _resolve_identity(credentials.credentials)
        return identity.raw_claims
    except AuthException as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.to_dict(),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


async def get_user_id_or_mock(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> uuid.UUID:
    """Return the caller's user UUID, supporting the mock-auth bridge.

    Resolution order:
        1. ``effective_auth_mode in (oidc, mock)`` — validate Bearer JWT, return
           ``UUID(sub)``.
        2. ``effective_auth_mode=disabled`` and ``X-Mock-User`` header present —
           ``uuid5(namespace, header_value)``.
        3. Otherwise — 401.

    Raises:
        HTTPException: 401 when no usable identity is available.
    """
    settings = get_settings()
    mode = settings.effective_auth_mode

    if mode in ("oidc", "mock"):
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
            identity = await _resolve_identity(credentials.credentials)
            return identity.user_id
        except AuthException as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.to_dict()) from exc

    # disabled mode — fall back to mock header
    mock_value = request.headers.get(settings.mock_user_header)
    if mock_value:
        return _mock_user_to_uuid(mock_value.strip())

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={
            "error_code": ErrorCode.AUTH_REQUIRED.value,
            "message": (
                "User identity required. Send a Bearer token (auth enabled) "
                f"or '{settings.mock_user_header}' header (disabled mode)."
            ),
        },
    )


async def get_optional_user_id(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> Optional[uuid.UUID]:
    """Like :func:`get_user_id_or_mock` but returns ``None`` when anonymous.

    Useful for public endpoints that adapt their response based on whether the
    caller is authenticated.
    """
    settings = get_settings()
    mode = settings.effective_auth_mode

    if mode in ("oidc", "mock"):
        if credentials is None:
            return None
        try:
            identity = await _resolve_identity(credentials.credentials)
            return identity.user_id
        except AuthException:
            return None

    # disabled mode — mock header optional
    mock_value = request.headers.get(settings.mock_user_header)
    if mock_value:
        return _mock_user_to_uuid(mock_value.strip())
    return None


async def validate_optional_token(token: Optional[str]) -> Optional[dict]:
    """Validate a WebSocket query-param token or single-use ticket.

    Accepts:
    * A full JWT (OIDC RS256 or mock HS256 depending on mode).
    * A WS ticket string (prefix ``wst_``) issued by
      ``POST /api/v1/auth/ws-ticket``.  Tickets are consumed atomically.

    Args:
        token: Raw token/ticket string from the ``?token=`` query parameter,
               or ``None``.

    Returns:
        Raw claims dict when auth is enabled and the token is valid,
        or ``None`` when auth is disabled.

    Raises:
        AuthException: If auth is enabled and the token is missing or invalid.
    """
    settings = get_settings()
    mode = settings.effective_auth_mode

    if mode == "disabled":
        return None

    if token is None:
        raise AuthException(
            ErrorCode.AUTH_REQUIRED,
            "WebSocket authentication required. Provide ?token=<jwt_or_ticket>.",
            status_code=401,
        )

    # WS ticket path (single-use, Redis-backed)
    if token.startswith("wst_"):
        user_id = await _consume_ws_ticket(token)
        return {"sub": str(user_id)}

    # JWT path
    identity = await _resolve_identity(token)
    return identity.raw_claims


async def extract_ws_token(token: Optional[str]) -> Optional[str]:
    """Extract user ID string from a WS token for logging/audit purposes.

    Does NOT consume WS tickets — only returns a stable string for logging.

    Args:
        token: Raw token/ticket string or ``None``.

    Returns:
        The user UUID string, ``"ticket-auth"`` for unconsumed tickets, or
        ``None`` if unavailable.
    """
    settings = get_settings()
    if settings.effective_auth_mode == "disabled" or token is None:
        return None
    try:
        if token.startswith("wst_"):
            return "ticket-auth"
        identity = await _resolve_identity(token)
        return str(identity.user_id)
    except AuthException:
        return None


# ---------------------------------------------------------------------------
# Role / permission dependency factories
# ---------------------------------------------------------------------------


def require_role(role: str):  # type: ignore[return]
    """Dependency factory that enforces a role claim.

    * ``disabled`` mode: no-op (always passes).
    * ``mock`` mode: the admin role passes if ``X-Mock-User`` header equals
      ``settings.mock_admin_user``; all other roles pass freely.
    * ``oidc`` mode: validates the Bearer token and checks the ``roles`` claim.

    Usage::

        @router.get("/admin-only")
        async def admin_endpoint(_: None = Depends(require_role("admin"))):
            ...
    """

    async def _check(
        request: Request,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
    ) -> None:
        settings = get_settings()
        mode = settings.effective_auth_mode

        if mode == "disabled":
            return

        if mode == "mock":
            if role == settings.oidc_admin_role:
                mock_value = request.headers.get(settings.mock_user_header, "")
                if mock_value.strip() != settings.mock_admin_user:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail={
                            "error_code": ErrorCode.AUTH_INSUFFICIENT_PERMISSIONS.value,
                            "message": f"Role '{role}' is required.",
                        },
                    )
            return  # other roles pass freely in mock mode

        # OIDC mode
        if credentials is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "error_code": ErrorCode.AUTH_REQUIRED.value,
                    "message": "Authentication is required.",
                },
                headers={"WWW-Authenticate": "Bearer"},
            )
        try:
            identity = await _resolve_identity(credentials.credentials)
        except AuthException as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.to_dict()) from exc

        if role not in identity.roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error_code": ErrorCode.AUTH_INSUFFICIENT_PERMISSIONS.value,
                    "message": f"Role '{role}' is required.",
                },
            )

    return _check


def require_permission(permission: str):  # type: ignore[return]
    """Dependency factory that enforces a permission claim (OIDC mode only).

    In ``disabled`` or ``mock`` mode: no-op.

    Usage::

        @router.post("/sessions")
        async def create_session(_: None = Depends(require_permission("sessions.write"))):
            ...
    """

    async def _check(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
    ) -> None:
        settings = get_settings()
        mode = settings.effective_auth_mode

        if mode in ("disabled", "mock"):
            return

        if credentials is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "error_code": ErrorCode.AUTH_REQUIRED.value,
                    "message": "Authentication is required.",
                },
                headers={"WWW-Authenticate": "Bearer"},
            )
        try:
            identity = await _resolve_identity(credentials.credentials)
        except AuthException as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.to_dict()) from exc

        if permission not in identity.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error_code": ErrorCode.AUTH_INSUFFICIENT_PERMISSIONS.value,
                    "message": f"Permission '{permission}' is required.",
                },
            )

    return _check


# ---------------------------------------------------------------------------
# Lifespan helper
# ---------------------------------------------------------------------------


async def prefetch_jwks() -> None:
    """Eagerly warm the JWKS cache at application startup (best-effort).

    A failure here does NOT prevent startup — the cache will be populated
    lazily on the first authenticated request.
    """
    settings = get_settings()
    if settings.effective_auth_mode != "oidc":
        return
    jwks_url = settings.oidc_jwks_endpoint
    if not jwks_url:
        logger.warning("oidc_jwks_prefetch_skipped", reason="OIDC_ISSUER not configured")
        return
    try:
        await _jwks_cache.prefetch(jwks_url)
        logger.info("oidc_jwks_prefetched", url=jwks_url)
    except Exception as exc:  # noqa: BLE001
        logger.warning("oidc_jwks_prefetch_failed", url=jwks_url, error=str(exc))



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
