"""Cross-cutting core infrastructure: settings, logging, and OpenTelemetry."""

from slm_sms.core.config import Settings, load_yaml_config, settings
from slm_sms.core.logging import configure_logging, get_logger
from slm_sms.core.telemetry import (
    configure_telemetry,
    flush_telemetry,
    get_meter_provider,
    get_metrics,
    get_tracer,
    get_tracer_provider,
    instrument_fastapi_app,
    record_engine_ready,
    run_in_executor_with_context,
    shutdown_telemetry,
)

__all__ = [
    "Settings",
    "configure_logging",
    "configure_telemetry",
    "flush_telemetry",
    "get_logger",
    "get_meter_provider",
    "get_metrics",
    "get_tracer",
    "get_tracer_provider",
    "instrument_fastapi_app",
    "load_yaml_config",
    "record_engine_ready",
    "run_in_executor_with_context",
    "settings",
    "shutdown_telemetry",
]
