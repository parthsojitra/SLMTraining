"""Gunicorn configuration for production deployment of the SLM SMS API.

Usage
-----
    gunicorn slm_sms.inference.server:app -c gunicorn.conf.py

Environment variables that override defaults
--------------------------------------------
    GUNICORN_WORKERS   — number of worker processes (default: 1 — keep at 1 per GPU)
    GUNICORN_TIMEOUT   — worker timeout in seconds (default: 120)
    GUNICORN_PORT      — port to bind (default: 8000)
    SLM_DRAIN_SECONDS  — seconds to drain in-flight requests on SIGTERM (default: 10)

Notes
-----
* Keep ``workers = 1`` when using GPU inference so the model is loaded only
  once and all requests share the same GPU context.  Scale horizontally
  (multiple pods/containers) instead.
* ``worker_class`` must be ``uvicorn.workers.UvicornWorker`` for ASGI apps.
* ``graceful_timeout`` must be >= ``SLM_DRAIN_SECONDS`` to allow the
  lifespan shutdown hook to drain in-flight requests before the worker exits.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Binding
# ---------------------------------------------------------------------------
bind = f"0.0.0.0:{os.getenv('GUNICORN_PORT', '8000')}"

# ---------------------------------------------------------------------------
# Worker configuration
# ---------------------------------------------------------------------------
# Keep at 1 per GPU — do NOT increase without profiling GPU memory carefully.
workers = int(os.getenv("GUNICORN_WORKERS", "1"))
worker_class = "uvicorn.workers.UvicornWorker"

# ---------------------------------------------------------------------------
# Timeouts
# ---------------------------------------------------------------------------
# LLM inference can be slow on CPU — give it 2 minutes.
timeout = int(os.getenv("GUNICORN_TIMEOUT", "120"))

# Must be >= SLM_DRAIN_SECONDS to let the lifespan hook drain in-flight requests.
graceful_timeout = int(os.getenv("SLM_DRAIN_SECONDS", "10")) + 5

keepalive = 5

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
accesslog = "-"      # stdout
errorlog = "-"       # stdout
loglevel = os.getenv("SLM_LOG_LEVEL", "info").lower()
access_log_format = (
    '{"time":"%(t)s","method":"%(m)s","path":"%(U)s","status":%(s)s,'
    '"response_length":%(b)s,"referrer":"%(f)s","user_agent":"%(a)s","duration_ms":%(D)s}'
)

# ---------------------------------------------------------------------------
# Process naming
# ---------------------------------------------------------------------------
proc_name = "slm-sms-api"

# Note: OpenTelemetry providers (BatchSpanProcessor / PeriodicExportingMetricReader)
# are initialized inside FastAPI's post-fork worker lifecycle (`create_app` / `lifespan`)
# rather than at pre-fork import time, avoiding broken background threads if `workers > 1`.
