"""FastAPI microservice for production Indian SMS entity extraction."""

from __future__ import annotations

import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel, Field, field_validator

from slm_sms.config import settings
from slm_sms.inference.engine import SMSExtractorEngine
from slm_sms.logging_config import configure_logging, get_logger
from slm_sms.telemetry import (
    configure_telemetry,
    flush_telemetry,
    instrument_fastapi_app,
    record_engine_ready,
    run_in_executor_with_context,
)

# ---------------------------------------------------------------------------
# Logging — configure before anything else so all submodules pick it up
# ---------------------------------------------------------------------------
configure_logging(level=settings.log_level, use_json=settings.log_json)
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# API Request / Response Models
# ---------------------------------------------------------------------------


class ExtractRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=480,
        description="Raw SMS message text (1–480 characters)",
        json_schema_extra={
            "example": (
                "[VM-HDFCBK] Rs.1,450.00 debited from A/c **4821 on 23-11-26 "
                "to swiggy@axisbank (UPI Ref: 432819284129). Avl Bal: Rs.34,210.50."
            )
        },
    )


class ExtractResponse(BaseModel):
    success: bool
    entity: dict[str, Any]
    latency_ms: float
    fallback_used: bool = False


class BatchExtractRequest(BaseModel):
    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of SMS message strings (1–50 items, each 1–480 chars)",
        json_schema_extra={
            "example": [
                "[VM-HDFCBK] Rs.1,450.00 debited from A/c **4821 to swiggy@axisbank",
                "489201 is your secret OTP for transaction of Rs.2,499.00 on your Axis Bank Card **7712 at ZOMATO.",
            ]
        },
    )

    @field_validator("texts")
    @classmethod
    def validate_each_text(cls, v: list[str]) -> list[str]:
        for i, t in enumerate(v):
            if not (1 <= len(t) <= 480):
                raise ValueError(f"texts[{i}] must be 1–480 characters (got {len(t)})")
        return v


class BatchExtractResponse(BaseModel):
    success: bool
    count: int
    results: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Lifespan — single authoritative init / teardown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: load ML engine once at startup, clean up at shutdown."""
    configure_telemetry()

    executor = ThreadPoolExecutor(
        max_workers=settings.inference_threads, thread_name_prefix="slm-infer"
    )
    app.state.executor = executor
    app.state.engine: SMSExtractorEngine | None = None

    if not settings.lazy_load:
        try:
            logger.info(
                "Starting inference engine",
                extra={
                    "model_id": settings.model_id,
                    "revision": settings.model_revision,
                    "adapter_path": settings.adapter_path,
                },
            )
            app.state.engine = SMSExtractorEngine(
                model_id_or_path=settings.model_id,
                adapter_path=settings.adapter_path,
                model_revision=settings.model_revision,
                max_new_tokens=settings.max_new_tokens,
                temperature=settings.temperature,
                top_p=settings.top_p,
            )
            record_engine_ready(True)
            logger.info("Inference engine ready", extra={"model_id": settings.model_id})
        except Exception as exc:
            record_engine_ready(False)
            logger.warning("Engine initialisation deferred", extra={"error": str(exc)})
    else:
        record_engine_ready(False)

    yield

    # Graceful shutdown: drain in-flight requests, flush telemetry, release resources
    logger.info(
        "Shutting down — draining in-flight requests",
        extra={"drain_seconds": settings.drain_seconds},
    )
    await asyncio.sleep(settings.drain_seconds)
    record_engine_ready(False)
    flush_telemetry()
    executor.shutdown(wait=False)
    app.state.engine = None
    logger.info("Inference engine released")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def _get_engine(request: Request) -> SMSExtractorEngine:
    engine: SMSExtractorEngine | None = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Inference engine is not ready. Retry shortly.")
    return engine


def create_app(*, init_telemetry: bool = True) -> FastAPI:
    """Construct and instrument the FastAPI application."""
    if init_telemetry:
        configure_telemetry()

    application = FastAPI(
        title="Indian SMS Entity Extraction API",
        description=(
            "Production-grade Microservice powered by Fine-Tuned Qwen Small Language Models (SLMs) "
            "for Indian SMS Information Extraction."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "X-Request-ID",
            "traceparent",
            "tracestate",
        ],
    )

    @application.middleware("http")
    async def request_id_middleware(request: Request, call_next) -> Response:
        rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = rid

        span = trace.get_current_span()
        span_ctx = span.get_span_context() if span is not None else None
        if span is not None and span.is_recording():
            span.set_attribute("http.request_id", rid)

        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        if span_ctx is not None and span_ctx.is_valid:
            response.headers["X-Trace-ID"] = format(span_ctx.trace_id, "032x")
        return response

    @application.get("/", tags=["Info"])
    async def root(request: Request):
        return {
            "service": "Indian SMS Entity Extraction Microservice",
            "model": settings.model_id,
            "status": "online",
            "engine_loaded": getattr(request.app.state, "engine", None) is not None,
        }

    @application.get("/health/live", tags=["Health"])
    async def liveness():
        """Liveness probe — always 200 if the process is running."""
        return {"status": "alive"}

    @application.get("/health/ready", tags=["Health"])
    async def readiness(request: Request):
        """Readiness probe — 503 until the inference engine is loaded."""
        if getattr(request.app.state, "engine", None) is None:
            raise HTTPException(status_code=503, detail="Engine not ready")
        return {
            "status": "ready",
            "device": str(request.app.state.engine.device),
        }

    @application.get("/health", tags=["Health"], include_in_schema=False)
    async def health_legacy():
        return {"status": "healthy"}

    @application.post("/api/v1/extract", response_model=ExtractResponse, tags=["Inference"])
    async def extract_single_sms(payload: ExtractRequest, request: Request):
        """Extract structured financial and transactional entities from a single SMS."""
        rid = getattr(request.state, "request_id", "-")
        engine = _get_engine(request)

        logger.info("extract_start", extra={"request_id": rid, "text_length": len(payload.text)})

        try:
            result = await run_in_executor_with_context(
                request.app.state.executor, engine.extract, payload.text
            )
        except Exception as exc:
            span = trace.get_current_span()
            if span is not None and span.is_recording():
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
            logger.error(
                "extract_error", extra={"request_id": rid, "error": str(exc)}, exc_info=True
            )
            raise HTTPException(status_code=500, detail=f"Inference error: {exc!s}")

        logger.info(
            "extract_done",
            extra={
                "request_id": rid,
                "latency_ms": result["latency_ms"],
                "fallback": result["fallback_used"],
            },
        )
        return ExtractResponse(
            success=True,
            entity=result["entity"],
            latency_ms=result["latency_ms"],
            fallback_used=result["fallback_used"],
        )

    @application.post(
        "/api/v1/extract/batch", response_model=BatchExtractResponse, tags=["Inference"]
    )
    async def extract_batch_sms(payload: BatchExtractRequest, request: Request):
        """Extract entities from multiple SMS messages using native GPU batching."""
        rid = getattr(request.state, "request_id", "-")
        engine = _get_engine(request)

        logger.info(
            "batch_extract_start", extra={"request_id": rid, "batch_size": len(payload.texts)}
        )

        try:
            results = await run_in_executor_with_context(
                request.app.state.executor, engine.extract_batch, payload.texts
            )
        except Exception as exc:
            span = trace.get_current_span()
            if span is not None and span.is_recording():
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
            logger.error(
                "batch_extract_error", extra={"request_id": rid, "error": str(exc)}, exc_info=True
            )
            raise HTTPException(status_code=500, detail=f"Batch inference error: {exc!s}")

        logger.info("batch_extract_done", extra={"request_id": rid, "count": len(results)})
        return BatchExtractResponse(success=True, count=len(results), results=results)

    instrument_fastapi_app(application)
    return application


app = create_app()
