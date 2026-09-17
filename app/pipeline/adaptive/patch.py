"""Validates and clamps a critic-proposed config patch against the tool catalog.

The vision critic is a natural-language model and its JSON output cannot be
trusted blindly: it may propose a field the current step does not own, a
value of the wrong type, or a numeric value outside a sane range. This
module is the single place where a raw critic patch is turned into a patch
that is safe to merge into a real :class:`~app.domain.profile.ProcessingProfileConfig`.
"""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger
from app.pipeline.adaptive.tool_catalog import ToolCapability, ValueType, get_capability

logger = get_logger(__name__)


def sanitize_patch(
    raw_patch: dict[str, Any],
    *,
    allowed_fields: tuple[str, ...],
) -> dict[str, Any]:
    """Filter and clamp a critic-proposed patch to safe, allowed values.

    Any field not in ``allowed_fields`` is dropped — the critic is only
    allowed to touch fields explicitly exposed for the current step. Values
    are type-coerced and, for fields with catalog-declared choices/bounds,
    clamped into range rather than rejected outright: a slightly
    out-of-range suggestion usually still reflects a valid intent (e.g.
    "stretch a bit more") that clamping preserves instead of discarding.

    Args:
        raw_patch: Patch dict as proposed by the vision critic.
        allowed_fields: Field names the critic was allowed to change for the
            current pipeline step.

    Returns:
        Sanitised patch containing only valid, in-range, allowed fields.
    """
    sanitized: dict[str, Any] = {}
    allowed = set(allowed_fields)
    for field_name, raw_value in raw_patch.items():
        if field_name not in allowed:
            logger.warning("adaptive_patch_field_not_allowed", field=field_name)
            continue
        cap = get_capability(field_name)
        value = _coerce_and_clamp(field_name, raw_value, cap)
        if value is not None:
            sanitized[field_name] = value
    return sanitized


def _coerce_and_clamp(field_name: str, raw_value: Any, cap: ToolCapability) -> Any:
    """Coerce ``raw_value`` to the capability's declared type and clamp it.

    Args:
        field_name: Profile field name (used only for log context).
        raw_value: Raw value as proposed by the critic.
        cap: Catalog entry describing the expected type/choices/bounds.

    Returns:
        The sanitised value, or ``None`` if it could not be coerced or does
        not satisfy the catalog's declared choices.
    """
    try:
        if cap.value_type == ValueType.BOOL:
            if not isinstance(raw_value, bool):
                raise TypeError(f"expected bool, got {type(raw_value).__name__}")
            value: Any = raw_value
        elif cap.value_type == ValueType.INT:
            if isinstance(raw_value, bool):
                raise TypeError("expected int, got bool")
            value = int(raw_value)
        elif cap.value_type == ValueType.FLOAT:
            if isinstance(raw_value, bool):
                raise TypeError("expected float, got bool")
            value = float(raw_value)
        else:  # STR
            value = str(raw_value)
    except (TypeError, ValueError):
        logger.warning("adaptive_patch_type_coercion_failed", field=field_name, value=raw_value)
        return None

    if cap.choices is not None and value not in cap.choices:
        logger.warning(
            "adaptive_patch_value_not_in_choices",
            field=field_name,
            value=value,
            choices=cap.choices,
        )
        return None

    if cap.min_value is not None and value < cap.min_value:
        value = cap.min_value
    if cap.max_value is not None and value > cap.max_value:
        value = cap.max_value

    return value
