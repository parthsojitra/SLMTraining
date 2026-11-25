"""Production OpenTelemetry observability setup for traces, metrics, and log correlation.

Provides:
- ``configure_telemetry()``: builds SDK ``TracerProvider`` and ``MeterProvider``
  with OTLP (HTTP/protobuf or gRPC), Console, or no-op local mode, W3C TraceContext
  + Baggage propagation, and domain-tuned latency histogram buckets.
- ``instrument_fastapi_app()``: instruments a FastAPI app via ``FastAPIInstrumentor``
  while excluding noisy health probe URLs and ASGI send/receive micro-spans.
- ``get_tracer()`` / ``get_metrics()``: centralized handles used across inference
  and training modules.
- ``run_in_executor_with_context()``: propagates active OpenTelemetry + ``contextvars``
  context across ``ThreadPoolExecutor`` worker threads so child spans and logs share
  the parent request's ``trace_id``.
- ``shutdown_telemetry()``: flushes and shuts down providers on graceful exit.
"""

from __future__ import annotations

import asyncio
import atexit
import contextvars
import functools
from collections.abc import Callable
from concurrent.futures import Executor
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI
from opentelemetry import metrics, propagate, trace
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    MetricReader,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from slm_sms import __version__
from slm_sms.config import settings
from slm_sms.logging_config import get_logger

logger = get_logger(__name__)

_INFERENCE_DURATION_BUCKETS_MS = (
    1.0,
    5.0,
    10.0,
    25.0,
    50.0,
    100.0,
    250.0,
    500.0,
    1000.0,
    2500.0,
    5000.0,
    10000.0,
    30000.0,
)


@dataclass
class SLMMetrics:
    """Domain-specific OpenTelemetry metric instruments for SLM inference."""

    requests_total: metrics.Counter
    inference_duration_ms: metrics.Histogram
    batch_size: metrics.Histogram
    input_tokens_total: metrics.Counter
    output_tokens_total: metrics.Counter
    fallback_total: metrics.Counter
    engine_ready: metrics.UpDownCounter


def _build_instruments(meter: metrics.Meter) -> SLMMetrics:
    """Create SLM inference metric instruments on the given Meter."""
    return SLMMetrics(
        requests_total=meter.create_counter(
            name="slm.inference.requests",
            description="Total SMS extraction inference operations processed",
            unit="{request}",
        ),
        inference_duration_ms=meter.create_histogram(
            name="slm.inference.duration",
            description="End-to-end SMS extraction latency",
            unit="ms",
        ),
        batch_size=meter.create_histogram(
            name="slm.inference.batch_size",
            description="Number of SMS texts per batch extraction call",
            unit="{item}",
        ),
        input_tokens_total=meter.create_counter(
            name="slm.inference.tokens.input",
            description="Prompt input tokens consumed by the SLM",
            unit="{token}",
        ),
        output_tokens_total=meter.create_counter(
            name="slm.inference.tokens.output",
            description="Generated output tokens produced by the SLM",
            unit="{token}",
        ),
        fallback_total=meter.create_counter(
            name="slm.inference.fallback.total",
            description="Total invocations handled by deterministic regex fallback",
            unit="{call}",
        ),
        engine_ready=meter.create_up_down_counter(
            name="slm.engine.ready",
            description="Indicates whether neural model weights are loaded (1=ready, 0=unloaded)",
            unit="{state}",
        ),
    )


class _TelemetryState:
    """Module-level state storing active OpenTelemetry providers and instruments."""

    def __init__(self) -> None:
        self.configured: bool = False
        self.tracer_provider: TracerProvider | None = None
        self.meter_provider: MeterProvider | None = None
        self.instruments: SLMMetrics | None = None
        self._engine_ready_state: int = 0


_state = _TelemetryState()


def _build_resource(
    service_name: str | None = None,
    environment: str | None = None,
) -> Resource:
    """Construct an OpenTelemetry Resource following semantic conventions."""
    return Resource.create(
        {
            "service.name": service_name or settings.otel_service_name,
            "service.version": __version__,
            "deployment.environment.name": environment or settings.environment,
            "slm.model.id": settings.model_id,
        }
    )


def _create_otlp_span_exporter(endpoint: str | None, protocol: str) -> SpanExporter:
    """Instantiate OTLP span exporter for HTTP/protobuf or gRPC."""
    if protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter as GrpcSpanExporter,
        )

        return GrpcSpanExporter(endpoint=endpoint) if endpoint else GrpcSpanExporter()

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter as HttpSpanExporter,
    )

    if endpoint:
        traces_url = endpoint.rstrip("/")
        if not traces_url.endswith("/v1/traces"):
            traces_url = f"{traces_url}/v1/traces"
        return HttpSpanExporter(endpoint=traces_url)
    return HttpSpanExporter()


def _create_otlp_metric_reader(
    endpoint: str | None,
    protocol: str,
    interval_ms: int,
) -> MetricReader:
    """Instantiate OTLP PeriodicExportingMetricReader for HTTP/protobuf or gRPC."""
    if protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
            OTLPMetricExporter as GrpcMetricExporter,
        )

        exporter = GrpcMetricExporter(endpoint=endpoint) if endpoint else GrpcMetricExporter()
    else:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter as HttpMetricExporter,
        )

        if endpoint:
            metrics_url = endpoint.rstrip("/")
            if not metrics_url.endswith("/v1/metrics"):
                metrics_url = f"{metrics_url}/v1/metrics"
            exporter = HttpMetricExporter(endpoint=metrics_url)
        else:
            exporter = HttpMetricExporter()

    return PeriodicExportingMetricReader(exporter, export_interval_millis=interval_ms)


def configure_telemetry(
    *,
    service_name: str | None = None,
    environment: str | None = None,
    exporter_mode: str | None = None,
    otlp_endpoint: str | None = None,
    otlp_protocol: str | None = None,
    sample_rate: float | None = None,
    metric_interval_ms: int | None = None,
    span_processor: SpanProcessor | None = None,
    span_exporter: SpanExporter | None = None,
    metric_reader: MetricReader | None = None,
    force: bool = False,
) -> tuple[TracerProvider | None, MeterProvider | None]:
    """Initialize OpenTelemetry TracerProvider, MeterProvider, and W3C propagators.

    When ``settings.otel_enabled`` is False (and no explicit test exporters are passed),
    telemetry stays disabled and returns ``(None, None)``.

    In ``exporter_mode="none"`` (default when no ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set),
    SDK providers are still installed in-process without background network exporters so
    that trace IDs and log correlation remain active locally.
    """
    if _state.configured and not force:
        return _state.tracer_provider, _state.meter_provider

    has_custom_override = any(x is not None for x in (span_processor, span_exporter, metric_reader))
    if not settings.otel_enabled and not has_custom_override:
        logger.info("OpenTelemetry disabled via SLM_OTEL_ENABLED=false")
        return None, None

    if _state.configured and force:
        shutdown_telemetry()

    resource = _build_resource(service_name=service_name, environment=environment)
    mode = (exporter_mode or settings.otel_exporter).lower()
    endpoint = otlp_endpoint if otlp_endpoint is not None else settings.otel_otlp_endpoint
    protocol = (otlp_protocol or settings.otel_otlp_protocol).lower()
    rate = settings.otel_trace_sample_rate if sample_rate is None else sample_rate
    export_ms = (
        settings.otel_metric_export_interval_ms
        if metric_interval_ms is None
        else metric_interval_ms
    )

    # ------------------------------------------------------------------
    # Propagators: W3C TraceContext + W3C Baggage
    # ------------------------------------------------------------------
    propagate.set_global_textmap(
        CompositePropagator([TraceContextTextMapPropagator(), W3CBaggagePropagator()])
    )

    # ------------------------------------------------------------------
    # TracerProvider
    # ------------------------------------------------------------------
    sampler = ParentBased(TraceIdRatioBased(rate))
    tracer_provider = TracerProvider(resource=resource, sampler=sampler)

    if span_processor is not None:
        tracer_provider.add_span_processor(span_processor)
    elif span_exporter is not None:
        tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    elif mode == "otlp":
        otlp_spans = _create_otlp_span_exporter(endpoint=endpoint, protocol=protocol)
        tracer_provider.add_span_processor(BatchSpanProcessor(otlp_spans))
    elif mode == "console":
        tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    # Register globally the first time (only when currently ProxyTracerProvider)
    if isinstance(trace.get_tracer_provider(), trace.ProxyTracerProvider):
        trace.set_tracer_provider(tracer_provider)

    # ------------------------------------------------------------------
    # MeterProvider + Custom Histogram Buckets
    # ------------------------------------------------------------------
    readers: list[MetricReader] = []
    if metric_reader is not None:
        readers.append(metric_reader)
    elif mode == "otlp":
        readers.append(
            _create_otlp_metric_reader(endpoint=endpoint, protocol=protocol, interval_ms=export_ms)
        )
    elif mode == "console":
        readers.append(
            PeriodicExportingMetricReader(ConsoleMetricExporter(), export_interval_millis=export_ms)
        )

    views = [
        View(
            instrument_name="slm.inference.duration",
            aggregation=ExplicitBucketHistogramAggregation(list(_INFERENCE_DURATION_BUCKETS_MS)),
        ),
        View(
            instrument_name="slm.inference.batch_size",
            aggregation=ExplicitBucketHistogramAggregation([1.0, 2.0, 5.0, 10.0, 25.0, 50.0]),
        ),
    ]
    meter_provider = MeterProvider(resource=resource, metric_readers=readers, views=views)

    if isinstance(metrics.get_meter_provider(), metrics.NoOpMeterProvider):
        metrics.set_meter_provider(meter_provider)

    # Cache in module state and refresh metric instruments
    _state.tracer_provider = tracer_provider
    _state.meter_provider = meter_provider
    _state.instruments = _build_instruments(meter_provider.get_meter("slm_sms", __version__))
    _state._engine_ready_state = 0
    _state.configured = True

    logger.info(
        "OpenTelemetry configured",
        extra={
            "otel_service_name": service_name or settings.otel_service_name,
            "otel_exporter": mode,
            "otel_protocol": protocol if mode == "otlp" else None,
            "otel_sample_rate": rate,
        },
    )
    return tracer_provider, meter_provider


def get_tracer_provider() -> trace.TracerProvider:
    """Return active TracerProvider (falls back to OpenTelemetry global provider)."""
    return _state.tracer_provider or trace.get_tracer_provider()


def get_meter_provider() -> metrics.MeterProvider:
    """Return active MeterProvider (falls back to OpenTelemetry global provider)."""
    return _state.meter_provider or metrics.get_meter_provider()


def get_tracer(name: str = "slm_sms") -> trace.Tracer:
    """Return a Tracer bound to the active TracerProvider."""
    return get_tracer_provider().get_tracer(name, __version__)


def get_metrics() -> SLMMetrics:
    """Return domain metric instruments bound to the active MeterProvider."""
    if _state.instruments is None:
        _state.instruments = _build_instruments(
            get_meter_provider().get_meter("slm_sms", __version__)
        )
    return _state.instruments


def record_engine_ready(is_ready: bool) -> None:
    """Update the ``slm.engine.ready`` gauge-like UpDownCounter cleanly on state transitions."""
    target = 1 if is_ready else 0
    delta = target - _state._engine_ready_state
    if delta != 0:
        get_metrics().engine_ready.add(delta, attributes={"model_id": settings.model_id})
        _state._engine_ready_state = target


def instrument_fastapi_app(app: FastAPI) -> None:
    """Attach OpenTelemetry ASGI instrumentation to a FastAPI application instance."""
    if not settings.otel_enabled and not _state.configured:
        return

    if getattr(app, "_is_instrumented_by_opentelemetry", False):
        return

    FastAPIInstrumentor.instrument_app(
        app,
        tracer_provider=get_tracer_provider(),
        meter_provider=get_meter_provider(),
        excluded_urls=settings.otel_excluded_urls,
        exclude_spans=["receive", "send"],
    )


async def run_in_executor_with_context[T](
    executor: Executor | None,
    func: Callable[..., T],
    *args: Any,
) -> T:
    """Run blocking ``func(*args)`` in ``executor`` preserving OTel & contextvars context."""
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    bound = functools.partial(func, *args)
    return await loop.run_in_executor(executor, ctx.run, bound)


def flush_telemetry(timeout_millis: int = 5000) -> None:
    """Flush pending spans and metrics without shutting down active providers."""
    if _state.tracer_provider is not None:
        try:
            _state.tracer_provider.force_flush(timeout_millis=timeout_millis)
        except Exception as exc:
            logger.warning("Failed flushing TracerProvider", extra={"error": str(exc)})

    if _state.meter_provider is not None:
        try:
            _state.meter_provider.force_flush(timeout_millis=timeout_millis)
        except Exception as exc:
            logger.warning("Failed flushing MeterProvider", extra={"error": str(exc)})


def shutdown_telemetry() -> None:
    """Flush and shut down active OpenTelemetry providers."""
    if _state.tracer_provider is not None:
        try:
            _state.tracer_provider.force_flush(timeout_millis=5000)
            _state.tracer_provider.shutdown()
        except Exception as exc:
            logger.warning("Failed shutting down TracerProvider", extra={"error": str(exc)})

    if _state.meter_provider is not None:
        try:
            _state.meter_provider.force_flush(timeout_millis=5000)
            _state.meter_provider.shutdown()
        except Exception as exc:
            logger.warning("Failed shutting down MeterProvider", extra={"error": str(exc)})

    _state.tracer_provider = None
    _state.meter_provider = None
    _state.instruments = None
    _state._engine_ready_state = 0
    _state.configured = False


atexit.register(shutdown_telemetry)
