"""Centralized JSON logging configuration for slm_sms.

Usage
-----
Import and call ``configure_logging()`` once at application startup (in
``lifespan``, ``cli.py`` entry-points, or the top of any script).  Every
module that needs a logger should do::

    from slm_sms.core.logging import get_logger
    logger = get_logger(__name__)

The logger name is preserved so log records are still filterable by module,
but all output is emitted as newline-delimited JSON understood by log
aggregators (Loki, Cloud Logging, Datadog, etc.).
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any


class _JsonFormatter(logging.Formatter):
    """Minimal JSON log formatter with automatic OpenTelemetry trace correlation."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        import json

        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Inject active OpenTelemetry trace/span IDs for log-to-trace correlation
        try:
            from opentelemetry import trace

            span = trace.get_current_span()
            span_ctx = span.get_span_context() if span is not None else None
            if span_ctx is not None and span_ctx.is_valid:
                payload["trace_id"] = format(span_ctx.trace_id, "032x")
                payload["span_id"] = format(span_ctx.span_id, "016x")
                payload["trace_flags"] = f"{int(span_ctx.trace_flags):02x}"
        except Exception:
            pass

        # Structured extra fields attached with logger.info("msg", extra={...})
        _STANDARD_ATTRS = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "message",
            "taskName",
        }
        for key, val in record.__dict__.items():
            if key not in _STANDARD_ATTRS:
                payload[key] = val

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)

        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(
    level: str | None = None,
    use_json: bool | None = None,
) -> None:
    """Configure root logging for the application.

    Parameters
    ----------
    level:
        Log level string (``"DEBUG"``, ``"INFO"``, …).  Falls back to the
        ``SLM_LOG_LEVEL`` env-var, then ``"INFO"``.
    use_json:
        Emit JSON lines when ``True``.  Falls back to the ``SLM_LOG_JSON``
        env-var (``"true"``/``"false"``), then ``True`` unless a TTY is
        detected (so local dev stays human-readable).
    """
    effective_level = (level or os.getenv("SLM_LOG_LEVEL", "INFO")).upper()

    if use_json is None:
        env_val = os.getenv("SLM_LOG_JSON", "").lower()
        if env_val in ("true", "1", "yes"):
            use_json = True
        elif env_val in ("false", "0", "no"):
            use_json = False
        else:
            # Auto-detect: use plain text on interactive terminals
            use_json = not sys.stdout.isatty()

    handler = logging.StreamHandler(sys.stdout)

    if use_json:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )

    # Apply to the package root logger so all slm_sms.* modules inherit it
    pkg_logger = logging.getLogger("slm_sms")
    pkg_logger.setLevel(effective_level)
    pkg_logger.handlers.clear()
    pkg_logger.addHandler(handler)
    pkg_logger.propagate = False

    # Also set the root logger so third-party libs respect the level
    logging.basicConfig(level=effective_level, handlers=[handler], force=True)


def get_logger(name: str) -> logging.Logger:
    """Return a logger namespaced under ``slm_sms``."""
    if not name.startswith("slm_sms"):
        name = f"slm_sms.{name}"
    return logging.getLogger(name)
