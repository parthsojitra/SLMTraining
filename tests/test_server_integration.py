"""HTTP integration tests for the FastAPI server.

These tests spin up the full ASGI app via Starlette's TestClient so that
all middleware (request-ID injection, CORS), routing, request validation,
and response serialisation are exercised end-to-end without a real model.

The inference engine is kept in regex-fallback mode (SLM_LAZY_LOAD=true +
SLM_DRAIN_SECONDS=0) so tests finish fast and require no GPU or internet.

Run:
    uv run python -m pytest tests/test_server_integration.py -v
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

# Keep server in fast/offline mode for tests
os.environ["SLM_LAZY_LOAD"] = "true"
os.environ["SLM_DRAIN_SECONDS"] = "0"

# Import AFTER setting env vars so Settings picks them up
from slm_sms.inference.server import app  # noqa: E402

# ---------------------------------------------------------------------------
# Shared client fixture — lifespan runs once per test module
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    """TestClient with lifespan (startup + shutdown) executed once."""
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
# Info / Health endpoints
# ---------------------------------------------------------------------------


def test_root_returns_online(client):
    res = client.get("/")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "online"
    assert "model" in body
    assert "engine_loaded" in body


def test_liveness_always_200(client):
    res = client.get("/health/live")
    assert res.status_code == 200
    assert res.json()["status"] == "alive"


def test_legacy_health_always_200(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"


def test_readiness_503_when_engine_not_loaded(client):
    """In SLM_LAZY_LOAD=true mode the engine is never loaded → must return 503."""
    res = client.get("/health/ready")
    assert res.status_code == 503


# ---------------------------------------------------------------------------
# Request-ID middleware
# ---------------------------------------------------------------------------


def test_request_id_echoed_back(client):
    """Server must echo an X-Request-ID provided by the client."""
    rid = "test-rid-abc123"
    res = client.get("/health/live", headers={"X-Request-ID": rid})
    assert res.headers.get("X-Request-ID") == rid


def test_request_id_generated_when_absent(client):
    """Server must generate a UUID request-ID when none is supplied."""
    res = client.get("/health/live")
    rid = res.headers.get("X-Request-ID", "")
    assert len(rid) == 36  # UUID4 string length
    assert rid.count("-") == 4


# ---------------------------------------------------------------------------
# /api/v1/extract — input validation
# ---------------------------------------------------------------------------


def test_extract_rejects_empty_text(client):
    res = client.post("/api/v1/extract", json={"text": ""})
    assert res.status_code == 422  # Pydantic validation error


def test_extract_rejects_oversized_text(client):
    res = client.post("/api/v1/extract", json={"text": "x" * 481})
    assert res.status_code == 422


def test_extract_rejects_missing_text(client):
    res = client.post("/api/v1/extract", json={})
    assert res.status_code == 422


def test_extract_503_when_engine_not_loaded(client):
    """Lazy-load mode: engine is None → 503 on inference endpoints."""
    res = client.post(
        "/api/v1/extract",
        json={"text": "[VM-HDFCBK] Rs.500 debited from A/c **1234"},
    )
    assert res.status_code == 503


# ---------------------------------------------------------------------------
# /api/v1/extract/batch — input validation
# ---------------------------------------------------------------------------


def test_batch_rejects_empty_list(client):
    res = client.post("/api/v1/extract/batch", json={"texts": []})
    assert res.status_code == 422


def test_batch_rejects_oversized_list(client):
    texts = ["Rs.100 debited"] * 51  # 51 > max 50
    res = client.post("/api/v1/extract/batch", json={"texts": texts})
    assert res.status_code == 422


def test_batch_rejects_item_exceeding_max_length(client):
    texts = ["x" * 481]  # single item > 480 chars
    res = client.post("/api/v1/extract/batch", json={"texts": texts})
    assert res.status_code == 422


def test_batch_rejects_empty_item(client):
    res = client.post("/api/v1/extract/batch", json={"texts": [""]})
    assert res.status_code == 422


def test_batch_503_when_engine_not_loaded(client):
    res = client.post(
        "/api/v1/extract/batch",
        json={"texts": ["Rs.500 debited", "Rs.100 credited"]},
    )
    assert res.status_code == 503


# ---------------------------------------------------------------------------
# End-to-end fallback inference (engine injected manually)
# ---------------------------------------------------------------------------


def test_extract_returns_entity_via_fallback():
    """
    Force the engine into fallback mode and inject it into app.state so
    inference endpoints return 200 with valid entity JSON.
    """
    from slm_sms.inference.engine import SMSExtractorEngine

    # Build a stub engine in regex-fallback mode (no model weights needed)
    stub_engine = SMSExtractorEngine.__new__(SMSExtractorEngine)
    stub_engine.model = None
    stub_engine.tokenizer = None

    with TestClient(app, raise_server_exceptions=True) as c:
        c.app.state.engine = stub_engine  # inject

        sms = "[VM-HDFCBK] Rs.1,450.00 debited from A/c **4821 to swiggy@axisbank"
        res = c.post("/api/v1/extract", json={"text": sms})
        assert res.status_code == 200

        body = res.json()
        assert body["success"] is True
        assert body["fallback_used"] is True
        entity = body["entity"]
        assert entity["type"] == "debit"
        assert entity["amount"] == 1450.0
        assert entity["account"] == "4821"


def test_batch_returns_entities_via_fallback():
    from slm_sms.inference.engine import SMSExtractorEngine

    stub_engine = SMSExtractorEngine.__new__(SMSExtractorEngine)
    stub_engine.model = None
    stub_engine.tokenizer = None

    with TestClient(app, raise_server_exceptions=True) as c:
        c.app.state.engine = stub_engine

        texts = [
            "[VM-HDFCBK] Rs.1,450.00 debited from A/c **4821 to swiggy@axisbank",
            "489201 is your secret OTP for transaction of Rs.2,499.00 on Axis Bank Card **7712",
        ]
        res = c.post("/api/v1/extract/batch", json={"texts": texts})
        assert res.status_code == 200

        body = res.json()
        assert body["success"] is True
        assert body["count"] == 2
        assert len(body["results"]) == 2
        assert body["results"][0]["entity"]["type"] == "debit"
        assert body["results"][1]["entity"]["type"] == "otp"


def test_extract_response_shape(client):
    """Verify the response schema fields are present (even on 503)."""
    res = client.post("/api/v1/extract", json={"text": "Rs.500 debited"})
    # In lazy mode engine is None → 503 with detail string
    assert res.status_code == 503
    assert "detail" in res.json()


def test_openapi_schema_accessible(client):
    """OpenAPI JSON schema must be accessible (FastAPI auto-generates it)."""
    res = client.get("/openapi.json")
    assert res.status_code == 200
    schema = res.json()
    assert "paths" in schema
    assert "/api/v1/extract" in schema["paths"]
    assert "/api/v1/extract/batch" in schema["paths"]
