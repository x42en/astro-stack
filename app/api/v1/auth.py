"""Auth utilities API — WebSocket ticket issuance.

Provides a single endpoint that exchanges a valid Bearer JWT for a short-lived,
single-use WebSocket authentication ticket.  This avoids long-lived tokens
appearing in proxy / Nginx access logs (where ``?token=...`` query parameters
are typically recorded).

Usage flow:
    1. Client holds a valid OIDC access token.
    2. Client calls ``POST /api/v1/auth/ws-ticket`` with ``Authorization: Bearer <token>``.
    3. Server validates the token, stores a 30-second Redis-backed ticket, and
       returns ``{ "ticket": "wst_...", "expires_in": 30 }``.
    4. Client opens ``wss://.../ws/sessions/{id}?token=wst_...`` within 30 seconds.
    5. Backend consumes the ticket atomically (single-use) and upgrades the WS.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends

from app.api.middleware.auth import get_user_id_or_mock, issue_ws_ticket

router = APIRouter(prefix="/auth", tags=["auth"])

_TICKET_TTL_SECONDS = 30


@router.post(
    "/ws-ticket",
    summary="Issue a short-lived WebSocket authentication ticket",
)
async def create_ws_ticket(
    user_id: uuid.UUID = Depends(get_user_id_or_mock),
) -> dict:
    """Issue a single-use 30-second ticket for WebSocket authentication.

    The returned ``ticket`` value should be passed as the ``?token=`` query
    parameter on any WebSocket endpoint within 30 seconds.  Each ticket is
    consumed on first use and cannot be reused.

    Returns:
        ``ticket``: The raw ticket string (``wst_<random>``).
        ``expires_in``: Ticket lifetime in seconds (30).
    """
    ticket = await issue_ws_ticket(user_id)
    return {"ticket": ticket, "expires_in": _TICKET_TTL_SECONDS}
