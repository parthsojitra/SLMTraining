"""Centralized application settings for SLM-SMS.

All configuration is sourced from environment variables (with sane defaults)
so there is a **single source of truth** — no more hardcoded model names or
magic strings scattered across files.

Usage
-----
    from slm_sms.config import settings

    engine = SMSExtractorEngine(model_id_or_path=settings.model_id, ...)

Environment variables
---------------------
All variables are prefixed with ``SLM_`` and can be placed in a ``.env``
file (auto-loaded via python-dotenv when present).

    SLM_MODEL_ID            — HuggingFace model ID or local path
    SLM_MODEL_REVISION      — commit hash to pin (blank = latest, insecure)
    SLM_ADAPTER_PATH        — LoRA adapter directory (blank = base model only)
    SLM_LAZY_LOAD           — do not load model at startup (true/false)
    SLM_CORS_ORIGINS        — comma-separated allowed CORS origins
    SLM_API_KEY             — API key (blank = no auth enforcement)
    SLM_INFERENCE_THREADS   — thread-pool size for async inference offload
    SLM_DRAIN_SECONDS       — seconds to drain on graceful shutdown
    SLM_LOG_LEVEL           — log level (DEBUG/INFO/WARNING/ERROR)
    SLM_LOG_JSON            — force JSON logs (true/false; auto if unset)
    SLM_MAX_NEW_TOKENS      — max tokens the model may generate per request
    SLM_TEMPERATURE         — sampling temperature (0 = greedy)
    SLM_TOP_P               — nucleus sampling p
    SLM_ENVIRONMENT         — deployment environment (development/staging/production)
    SLM_OTEL_ENABLED        — enable OpenTelemetry instrumentation (true/false)
    SLM_OTEL_SERVICE_NAME   — OpenTelemetry service.name (or OTEL_SERVICE_NAME)
    SLM_OTEL_EXPORTER       — exporter mode: otlp | console | none
    OTEL_EXPORTER_OTLP_ENDPOINT — OTLP collector base URL (e.g. http://localhost:4318)
    OTEL_EXPORTER_OTLP_PROTOCOL — http/protobuf | grpc
    SLM_OTEL_TRACE_SAMPLE_RATE — trace sampling ratio (0.0 to 1.0)
    SLM_OTEL_METRIC_EXPORT_INTERVAL_MS — periodic metric export interval (ms)
    SLM_OTEL_EXCLUDED_URLS  — comma-separated regex URLs excluded from HTTP spans
"""

from __future__ import annotations

import os

# Optional: load .env for local development (python-dotenv)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


class Settings:
    """Application-wide configuration derived from environment variables.

    Instantiated once at import time via ``settings`` singleton; prefer
    ``from slm_sms.config import settings`` rather than constructing a new
    instance.
    """

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    @property
    def model_id(self) -> str:
        return os.getenv("SLM_MODEL_ID", "Qwen/Qwen3.5-2B")

    @property
    def model_revision(self) -> str | None:
        return os.getenv("SLM_MODEL_REVISION") or None

    @property
    def adapter_path(self) -> str | None:
        return os.getenv("SLM_ADAPTER_PATH") or None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @property
    def max_new_tokens(self) -> int:
        return int(os.getenv("SLM_MAX_NEW_TOKENS", "128"))

    @property
    def temperature(self) -> float:
        return float(os.getenv("SLM_TEMPERATURE", "0.01"))

    @property
    def top_p(self) -> float:
        return float(os.getenv("SLM_TOP_P", "0.95"))

    # ------------------------------------------------------------------
    # Server
    # ------------------------------------------------------------------
    @property
    def lazy_load(self) -> bool:
        return os.getenv("SLM_LAZY_LOAD", "false").lower() in ("true", "1", "yes")

    @property
    def cors_origins(self) -> list[str]:
        raw = os.getenv("SLM_CORS_ORIGINS", "http://localhost:3000")
        return [o.strip() for o in raw.split(",") if o.strip()]

    @property
    def api_key(self) -> str | None:
        return os.getenv("SLM_API_KEY") or None

    @property
    def inference_threads(self) -> int:
        return int(os.getenv("SLM_INFERENCE_THREADS", "2"))

    @property
    def drain_seconds(self) -> int:
        return int(os.getenv("SLM_DRAIN_SECONDS", "10"))

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    @property
    def log_level(self) -> str:
        return os.getenv("SLM_LOG_LEVEL", "INFO").upper()

    @property
    def log_json(self) -> bool | None:
        """None means auto-detect (JSON on non-TTY)."""
        raw = os.getenv("SLM_LOG_JSON", "").lower()
        if raw in ("true", "1", "yes"):
            return True
        if raw in ("false", "0", "no"):
            return False
        return None  # auto

    # ------------------------------------------------------------------
    # Observability / OpenTelemetry
    # ------------------------------------------------------------------
    @property
    def environment(self) -> str:
        return os.getenv("SLM_ENVIRONMENT") or os.getenv("DEPLOYMENT_ENVIRONMENT", "development")

    @property
    def otel_enabled(self) -> bool:
        return os.getenv("SLM_OTEL_ENABLED", "true").lower() in ("true", "1", "yes")

    @property
    def otel_service_name(self) -> str:
        return os.getenv("OTEL_SERVICE_NAME") or os.getenv("SLM_OTEL_SERVICE_NAME", "slm-sms")

    @property
    def otel_otlp_endpoint(self) -> str | None:
        return os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or None

    @property
    def otel_otlp_protocol(self) -> str:
        return os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf").strip().lower()

    @property
    def otel_exporter(self) -> str:
        """Exporter mode: 'otlp', 'console', or 'none'."""
        explicit = os.getenv("SLM_OTEL_EXPORTER", "").strip().lower()
        if explicit:
            return explicit
        # Auto-select 'otlp' whenever an OTLP endpoint is configured
        return "otlp" if self.otel_otlp_endpoint else "none"

    @property
    def otel_trace_sample_rate(self) -> float:
        rate = float(os.getenv("SLM_OTEL_TRACE_SAMPLE_RATE", "1.0"))
        return max(0.0, min(1.0, rate))

    @property
    def otel_metric_export_interval_ms(self) -> int:
        return int(os.getenv("SLM_OTEL_METRIC_EXPORT_INTERVAL_MS", "15000"))

    @property
    def otel_excluded_urls(self) -> str:
        return os.getenv("SLM_OTEL_EXCLUDED_URLS", "/health,/health/live,/health/ready")

    # ------------------------------------------------------------------
    # Convenience repr
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"Settings(model_id={self.model_id!r}, revision={self.model_revision!r}, "
            f"lazy_load={self.lazy_load}, log_level={self.log_level!r}, "
            f"otel_exporter={self.otel_exporter!r})"
        )


# ---------------------------------------------------------------------------
# Public singleton — import this everywhere
# ---------------------------------------------------------------------------
settings = Settings()
