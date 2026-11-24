"""FastAPI microservice for production Indian SMS entity extraction."""

from __future__ import annotations

import asyncio
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from slm_sms.inference.engine import SMSExtractorEngine
from slm_sms.logging_config import configure_logging, get_logger

# ---------------------------------------------------------------------------
# Logging — configure before anything else so all submodules pick it up
# ---------------------------------------------------------------------------
configure_logging()
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Thread-pool for offloading blocking model inference off the event loop
# ---------------------------------------------------------------------------
_INFERENCE_THREADS = int(os.getenv("SLM_INFERENCE_THREADS", "2"))
_executor = ThreadPoolExecutor(max_workers=_INFERENCE_THREADS, thread_name_prefix="slm-infer")


# ---------------------------------------------------------------------------
# Lifespan — single authoritative init / teardown (P0 #4 fix)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: load ML engine once at startup, clean up at shutdown."""
    model_id = os.getenv("SLM_MODEL_ID", "Qwen/Qwen3.5-2B")
    model_revision = os.getenv("SLM_MODEL_REVISION", None)  # P0 #5 — revision pin
    adapter_path = os.getenv("SLM_ADAPTER_PATH", None)
    lazy_load = os.getenv("SLM_LAZY_LOAD", "false").lower() == "true"

    app.state.engine: SMSExtractorEngine | None = None

    if not lazy_load:
        try:
            logger.info(
                "Starting inference engine",
                extra={
                    "model_id": model_id,
                    "revision": model_revision,
                    "adapter_path": adapter_path,
                },
            )
            app.state.engine = SMSExtractorEngine(
                model_id_or_path=model_id,
                adapter_path=adapter_path,
                model_revision=model_revision,
            )
            logger.info("Inference engine ready", extra={"model_id": model_id})
        except Exception as exc:
            logger.warning(
                "Engine initialisation deferred",
                extra={"error": str(exc)},
            )

    yield

    # Graceful shutdown: allow in-flight requests to drain (P3 #15)
    drain_secs = int(os.getenv("SLM_DRAIN_SECONDS", "10"))
    logger.info("Shutting down — draining in-flight requests", extra={"drain_seconds": drain_secs})
    await asyncio.sleep(drain_secs)

    _executor.shutdown(wait=False)
    app.state.engine = None
    logger.info("Inference engine released")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Indian SMS Entity Extraction API",
    description=(
        "Production-grade Microservice powered by Fine-Tuned Qwen Small Language Models (SLMs) "
        "for Indian SMS Information Extraction."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — use explicit origins via env-var; never wildcard + credentials (P0 #1 documented)
_allowed_origins = [
    o.strip()
    for o in os.getenv("SLM_CORS_ORIGINS", "http://localhost:3000").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)


# ---------------------------------------------------------------------------
# Middleware: request-ID injection (P2 #12)
# ---------------------------------------------------------------------------


@app.middleware("http")
async def request_id_middleware(request: Request, call_next) -> Response:
    rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = rid
    response: Response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response


# ---------------------------------------------------------------------------
# Helper: get engine or raise 503
# ---------------------------------------------------------------------------


def _get_engine(request: Request) -> SMSExtractorEngine:
    engine: SMSExtractorEngine | None = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Inference engine is not ready. Retry shortly.")
    return engine


# ---------------------------------------------------------------------------
# API Request / Response Models
# ---------------------------------------------------------------------------


class ExtractRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=480,  # P1 #8 — max SMS length (concatenated)
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
        max_length=50,  # P1 #8 — cap batch size at 50
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
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", tags=["Info"])
async def root(request: Request):
    return {
        "service": "Indian SMS Entity Extraction Microservice",
        "model": os.getenv("SLM_MODEL_ID", "Qwen/Qwen3.5-2B"),
        "status": "online",
        "engine_loaded": getattr(request.app.state, "engine", None) is not None,
    }


# P1 #9 — split liveness from readiness
@app.get("/health/live", tags=["Health"])
async def liveness():
    """Liveness probe — always 200 if the process is running."""
    return {"status": "alive"}


@app.get("/health/ready", tags=["Health"])
async def readiness(request: Request):
    """Readiness probe — 503 until the inference engine is loaded."""
    if getattr(request.app.state, "engine", None) is None:
        raise HTTPException(status_code=503, detail="Engine not ready")
    return {
        "status": "ready",
        "device": str(request.app.state.engine.device),
    }


# Keep legacy /health for backward compat — maps to liveness
@app.get("/health", tags=["Health"], include_in_schema=False)
async def health_legacy():
    return {"status": "healthy"}


@app.post("/api/v1/extract", response_model=ExtractResponse, tags=["Inference"])
async def extract_single_sms(payload: ExtractRequest, request: Request):
    """Extract structured financial and transactional entities from a single SMS."""
    rid = getattr(request.state, "request_id", "-")
    engine = _get_engine(request)

    logger.info(
        "extract_start",
        extra={"request_id": rid, "text_length": len(payload.text)},
    )

    loop = asyncio.get_event_loop()
    try:
        # P1 #7 — offload blocking model call to thread pool
        result = await loop.run_in_executor(_executor, engine.extract, payload.text)
    except Exception as exc:
        logger.error(
            "extract_error",
            extra={"request_id": rid, "error": str(exc)},
            exc_info=True,
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


@app.post("/api/v1/extract/batch", response_model=BatchExtractResponse, tags=["Inference"])
async def extract_batch_sms(payload: BatchExtractRequest, request: Request):
    """Extract entities from multiple SMS messages using native GPU batching."""
    rid = getattr(request.state, "request_id", "-")
    engine = _get_engine(request)

    logger.info(
        "batch_extract_start",
        extra={"request_id": rid, "batch_size": len(payload.texts)},
    )

    loop = asyncio.get_event_loop()
    try:
        # P1 #6 & #7 — native batch inference, offloaded to thread pool
        results = await loop.run_in_executor(_executor, engine.extract_batch, payload.texts)
    except Exception as exc:
        logger.error(
            "batch_extract_error",
            extra={"request_id": rid, "error": str(exc)},
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=f"Batch inference error: {exc!s}")

    logger.info(
        "batch_extract_done",
        extra={"request_id": rid, "count": len(results)},
    )
    return BatchExtractResponse(
        success=True,
        count=len(results),
        results=results,
    )
