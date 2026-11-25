"""Tests for OpenTelemetry tracing, metrics, propagation, PII safety, and log correlation."""

from __future__ import annotations

import json
import logging
import os

from fastapi.testclient import TestClient
from opentelemetry import trace
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from slm_sms.core.logging import _JsonFormatter
from slm_sms.core.telemetry import configure_telemetry, get_tracer, shutdown_telemetry
from slm_sms.inference.engine import SMSExtractorEngine


def _collect_metric_points(reader: InMemoryMetricReader) -> dict[str, list]:
    """Flatten InMemoryMetricReader MetricData into {metric_name: [data_points...]}."""
    metrics_data = reader.get_metrics_data()
    collected: dict[str, list] = {}
    if not metrics_data:
        return collected
    for resource_metrics in metrics_data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                pts = list(getattr(metric.data, "data_points", []))
                collected.setdefault(metric.name, []).extend(pts)
    return collected


def test_single_extract_traces_across_executor_and_records_metrics():
    """Verify HTTP span -> worker thread child spans, attributes, PII safety, and metrics."""
    os.environ["SLM_LAZY_LOAD"] = "true"
    os.environ["SLM_DRAIN_SECONDS"] = "0"

    span_exporter = InMemorySpanExporter()
    metric_reader = InMemoryMetricReader()
    configure_telemetry(
        service_name="slm-sms-test",
        environment="test",
        span_exporter=span_exporter,
        metric_reader=metric_reader,
        force=True,
    )

    try:
        from slm_sms.inference.server import create_app

        app = create_app(init_telemetry=False)
        stub_engine = SMSExtractorEngine.__new__(SMSExtractorEngine)
        stub_engine.model = None
        stub_engine.tokenizer = None

        secret_otp = "918273"
        sms_text = (
            f"{secret_otp} is your secret OTP for Rs.2,499.00 on Axis Bank Card **7712 at ZOMATO."
        )

        with TestClient(app, raise_server_exceptions=True) as client:
            client.app.state.engine = stub_engine
            res = client.post(
                "/api/v1/extract",
                json={"text": sms_text},
                headers={"X-Request-ID": "req-otel-001"},
            )
            assert res.status_code == 200
            assert res.headers.get("X-Request-ID") == "req-otel-001"
            assert len(res.headers.get("X-Trace-ID", "")) == 32

        spans = span_exporter.get_finished_spans()
        by_name = {s.name: s for s in spans}

        assert "POST /api/v1/extract" in by_name
        assert "slm.engine.extract" in by_name
        assert "slm.fallback.regex" in by_name

        http_span = by_name["POST /api/v1/extract"]
        extract_span = by_name["slm.engine.extract"]
        fb_span = by_name["slm.fallback.regex"]

        # Parent-child trace linkage must survive ThreadPoolExecutor boundary
        assert http_span.context.trace_id == extract_span.context.trace_id
        assert extract_span.parent is not None
        assert extract_span.parent.span_id == http_span.context.span_id
        assert fb_span.parent is not None
        assert fb_span.parent.span_id == extract_span.context.span_id

        # Trace header must match emitted span trace_id
        assert res.headers["X-Trace-ID"] == format(http_span.context.trace_id, "032x")

        # Safe attributes attached, zero PII leaked
        assert http_span.attributes.get("http.request_id") == "req-otel-001"
        assert extract_span.attributes.get("slm.entity.type") == "otp"
        assert extract_span.attributes.get("slm.fallback.used") is True
        assert extract_span.attributes.get("slm.text.length") == len(sms_text)
        assert fb_span.attributes.get("slm.fallback.reason") == "model_unloaded"

        for span in spans:
            serialized = json.dumps(dict(span.attributes or {}), default=str)
            assert secret_otp not in serialized
            assert "Axis Bank" not in serialized
            assert "ZOMATO" not in serialized

        # Metrics recorded cleanly
        metric_points = _collect_metric_points(metric_reader)
        assert "slm.inference.requests" in metric_points
        assert "slm.inference.duration" in metric_points
        assert "slm.inference.fallback.total" in metric_points
        assert metric_points["slm.inference.requests"][0].value == 1
        assert metric_points["slm.inference.requests"][0].attributes.get("entity_type") == "otp"
        assert metric_points["slm.inference.fallback.total"][0].value == 1
    finally:
        shutdown_telemetry()


def test_batch_extract_spans_and_metrics():
    """Verify batch endpoint records slm.engine.extract_batch span and batch_size metric."""
    os.environ["SLM_LAZY_LOAD"] = "true"
    os.environ["SLM_DRAIN_SECONDS"] = "0"

    span_exporter = InMemorySpanExporter()
    metric_reader = InMemoryMetricReader()
    configure_telemetry(
        span_exporter=span_exporter,
        metric_reader=metric_reader,
        force=True,
    )

    try:
        from slm_sms.inference.server import create_app

        app = create_app(init_telemetry=False)
        stub_engine = SMSExtractorEngine.__new__(SMSExtractorEngine)
        stub_engine.model = None
        stub_engine.tokenizer = None

        with TestClient(app, raise_server_exceptions=True) as client:
            client.app.state.engine = stub_engine
            res = client.post(
                "/api/v1/extract/batch",
                json={
                    "texts": [
                        "[VM-HDFCBK] Rs.1,450.00 debited from A/c **4821 to swiggy@axisbank",
                        "489201 is your secret OTP for Rs.2,499.00 on Card **7712",
                    ]
                },
            )
            assert res.status_code == 200

        spans = span_exporter.get_finished_spans()
        batch_spans = [s for s in spans if s.name == "slm.engine.extract_batch"]
        assert len(batch_spans) == 1
        assert batch_spans[0].attributes.get("slm.batch.size") == 2

        metric_points = _collect_metric_points(metric_reader)
        assert "slm.inference.batch_size" in metric_points
        assert metric_points["slm.inference.batch_size"][0].sum == 2
        assert metric_points["slm.inference.fallback.total"][0].value == 2
    finally:
        shutdown_telemetry()


def test_health_endpoints_excluded_and_traceparent_propagated():
    """Verify health probes create zero HTTP spans and incoming W3C traceparent is honored."""
    os.environ["SLM_LAZY_LOAD"] = "true"
    os.environ["SLM_DRAIN_SECONDS"] = "0"

    span_exporter = InMemorySpanExporter()
    metric_reader = InMemoryMetricReader()
    configure_telemetry(
        span_exporter=span_exporter,
        metric_reader=metric_reader,
        force=True,
    )

    try:
        from slm_sms.inference.server import create_app

        app = create_app(init_telemetry=False)
        incoming_trace_id = "4bf92f3577b34da6a3ce929d0e0e4736"
        traceparent = f"00-{incoming_trace_id}-00f067aa0ba902b7-01"

        with TestClient(app, raise_server_exceptions=True) as client:
            assert client.get("/health/live").status_code == 200
            assert client.get("/health/ready").status_code == 503
            assert client.get("/health").status_code == 200
            assert span_exporter.get_finished_spans() == ()

            res = client.get("/", headers={"traceparent": traceparent})
            assert res.status_code == 200
            assert res.headers.get("X-Trace-ID") == incoming_trace_id

        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert format(spans[0].context.trace_id, "032x") == incoming_trace_id
    finally:
        shutdown_telemetry()


def test_json_formatter_injects_trace_context():
    """JSON logs emitted while a span is active must contain matching trace_id and span_id."""
    span_exporter = InMemorySpanExporter()
    metric_reader = InMemoryMetricReader()
    configure_telemetry(
        span_exporter=span_exporter,
        metric_reader=metric_reader,
        force=True,
    )

    try:
        tracer = get_tracer("test.logging")
        formatter = _JsonFormatter()

        with tracer.start_as_current_span("test.operation") as span:
            record = logging.LogRecord(
                name="slm_sms.test",
                level=logging.INFO,
                pathname=__file__,
                lineno=1,
                msg="correlated event",
                args=(),
                exc_info=None,
            )
            record.request_id = "req-log-123"
            formatted = json.loads(formatter.format(record))

        ctx = span.get_span_context()
        assert formatted["message"] == "correlated event"
        assert formatted["request_id"] == "req-log-123"
        assert formatted["trace_id"] == format(ctx.trace_id, "032x")
        assert formatted["span_id"] == format(ctx.span_id, "016x")
        assert formatted["trace_flags"] == f"{int(ctx.trace_flags):02x}"
        assert ctx.trace_flags.sampled is True
        assert trace.get_current_span().get_span_context().is_valid is False
    finally:
        shutdown_telemetry()


def test_neural_path_records_token_metrics():
    """Mock neural generation path to verify GenAI token spans and counters."""
    import torch

    span_exporter = InMemorySpanExporter()
    metric_reader = InMemoryMetricReader()
    configure_telemetry(
        span_exporter=span_exporter,
        metric_reader=metric_reader,
        force=True,
    )

    class _FakeBatch(dict):
        def __init__(self, ids):
            super().__init__(input_ids=ids, attention_mask=torch.ones_like(ids))
            self.input_ids = ids
            self.attention_mask = torch.ones_like(ids)

        def to(self, _device):
            return self

    class _FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 1
        padding_side = "right"

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return messages[1]["content"]

        def __call__(self, text, **_kwargs):
            return _FakeBatch(torch.tensor([[10, 11, 12]]))

        def decode(self, _tokens, skip_special_tokens=True):
            return '{"type":"debit","amount":500.0,"currency":"INR","merchant":"Swiggy","account":"1234","spam":false}'

    class _FakeModel:
        def generate(self, **kwargs):
            # 3 input tokens + 4 generated tokens
            return torch.tensor([[10, 11, 12, 21, 22, 23, 24]])

    try:
        engine = SMSExtractorEngine.__new__(SMSExtractorEngine)
        engine.model_id_or_path = "Qwen/Qwen3.5-2B"
        engine.device = torch.device("cpu")
        engine.max_new_tokens = 128
        engine.temperature = 0.0
        engine.top_p = 0.95
        engine.tokenizer = _FakeTokenizer()
        engine.model = _FakeModel()

        res = engine.extract("[VM-HDFCBK] Rs.500 debited at Swiggy A/c **1234")
        assert res["fallback_used"] is False
        assert res["entity"]["type"] == "debit"

        spans = {s.name: s for s in span_exporter.get_finished_spans()}
        assert "slm.llm.generate" in spans
        assert "slm.entity.parse" in spans
        assert spans["slm.llm.generate"].attributes.get("gen_ai.usage.input_tokens") == 3
        assert spans["slm.llm.generate"].attributes.get("gen_ai.usage.output_tokens") == 4
        assert spans["slm.entity.parse"].attributes.get("slm.parse.valid") is True

        pts = _collect_metric_points(metric_reader)
        assert pts["slm.inference.tokens.input"][0].value == 3
        assert pts["slm.inference.tokens.output"][0].value == 4
        assert pts["slm.inference.requests"][0].attributes.get("mode") == "model"
    finally:
        shutdown_telemetry()
