"""Structured logging configuration using structlog.

Provides a ``get_logger`` factory and a ``configure_logging`` initialiser
that must be called once at startup. All log output is JSON in production
and human-readable in development (based on ``log_level``).

Example:
    >>> logger = get_logger(__name__)
    >>> logger.info("session_detected", session_id=str(session_id), path=path)
"""

from __future__ import annotations

import logging
import sys

import structlog


def _safe_add_logger_name(
    logger: object,
    method_name: str,
    event_dict: structlog.types.EventDict,
) -> structlog.types.EventDict:
    """Wrapper around add_logger_name that tolerates a None logger.

    ARQ's internal stdlib logger calls can pass None as the logger instance,
    which causes add_logger_name to raise AttributeError. This wrapper skips
    the processor in that case.
    """
    if logger is None:
        return event_dict
    return structlog.stdlib.add_logger_name(logger, method_name, event_dict)


def configure_logging(log_level: str = "info") -> None:
    """Configure structlog for the application.

    In ``debug`` mode, logs are rendered as coloured key-value pairs.
    In other modes, logs are rendered as JSON for log aggregation pipelines.

    Args:
        log_level: One of ``debug``, ``info``, ``warning``, ``error``.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        _safe_add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if log_level == "debug":
        renderer: structlog.types.Processor = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            *shared_processors,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    # Silence noisy third-party loggers
    for noisy in ("uvicorn.access", "watchdog"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structlog bound logger for the given module name.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A bound structlog logger instance.
    """
    return structlog.get_logger(name)  # type: ignore[return-value]
