"""FastAPI microservice for production Indian SMS entity extraction."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from slm_sms.inference.engine import SMSExtractorEngine

engine: SMSExtractorEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for loading ML models upon startup."""
    global engine
    model_id = os.getenv("SLM_MODEL_ID", "Qwen/Qwen3.5-2B")
    adapter_path = os.getenv("SLM_ADAPTER_PATH", None)

    # Initialize model if not in lazy mode
    lazy_load = os.getenv("SLM_LAZY_LOAD", "false").lower() == "true"
    if not lazy_load:
        try:
            engine = SMSExtractorEngine(model_id_or_path=model_id, adapter_path=adapter_path)
        except Exception as e:
            # Keep server running even if model download is deferred
            print(f"Warning: Engine initialization deferred: {e}")

    yield
    engine = None


app = FastAPI(
    title="Indian SMS Entity Extraction API",
    description="Production-grade Microservice powered by Fine-Tuned Qwen Small Language Models (SLMs) for Indian SMS Information Extraction.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- API Request / Response Models ---------------- #


class ExtractRequest(BaseModel):
    text: str = Field(
        ...,
        description="Raw SMS message text",
        json_schema_extra={
            "example": "[VM-HDFCBK] Rs.1,450.00 debited from A/c **4821 on 23-11-26 to swiggy@axisbank (UPI Ref: 432819284129). Avl Bal: Rs.34,210.50."
        },
    )


class ExtractResponse(BaseModel):
    success: bool
    entity: dict[str, Any]
    latency_ms: float
    fallback_used: bool = False
    raw_output: str | None = None


class BatchExtractRequest(BaseModel):
    texts: list[str] = Field(
        ...,
        description="List of SMS message strings",
        json_schema_extra={
            "example": [
                "[VM-HDFCBK] Rs.1,450.00 debited from A/c **4821 to swiggy@axisbank",
                "489201 is your secret OTP for transaction of Rs.2,499.00 on your Axis Bank Card **7712 at ZOMATO.",
            ]
        },
    )


class BatchExtractResponse(BaseModel):
    success: bool
    count: int
    results: list[dict[str, Any]]


# ---------------- Endpoints ---------------- #


@app.get("/", tags=["Info"])
async def root():
    return {
        "service": "Indian SMS Entity Extraction Microservice",
        "model": os.getenv("SLM_MODEL_ID", "Qwen/Qwen3.5-2B"),
        "status": "online",
        "engine_loaded": engine is not None,
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": engine is not None,
        "device": str(engine.device) if engine else "none",
    }


@app.post("/api/v1/extract", response_model=ExtractResponse, tags=["Inference"])
async def extract_single_sms(payload: ExtractRequest):
    """Extract structured financial and transactional entities from a single SMS."""
    global engine
    if engine is None:
        model_id = os.getenv("SLM_MODEL_ID", "Qwen/Qwen3.5-2B")
        adapter_path = os.getenv("SLM_ADAPTER_PATH", None)
        try:
            engine = SMSExtractorEngine(model_id_or_path=model_id, adapter_path=adapter_path)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model engine failed to initialize: {e}")

    try:
        result = engine.extract(payload.text)
        return ExtractResponse(
            success=True,
            entity=result["entity"],
            latency_ms=result["latency_ms"],
            fallback_used=result["fallback_used"],
            raw_output=result["raw_output"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e!s}")


@app.post("/api/v1/extract/batch", response_model=BatchExtractResponse, tags=["Inference"])
async def extract_batch_sms(payload: BatchExtractRequest):
    """Extract entities from multiple SMS messages concurrently."""
    global engine
    if engine is None:
        model_id = os.getenv("SLM_MODEL_ID", "Qwen/Qwen3.5-2B")
        adapter_path = os.getenv("SLM_ADAPTER_PATH", None)
        try:
            engine = SMSExtractorEngine(model_id_or_path=model_id, adapter_path=adapter_path)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model engine failed to initialize: {e}")

    try:
        results = engine.extract_batch(payload.texts)
        return BatchExtractResponse(
            success=True,
            count=len(results),
            results=results,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch inference error: {e!s}")
