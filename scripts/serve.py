#!/usr/bin/env python3
"""Run production FastAPI microservice."""

import argparse
import os

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Start SMS extraction FastAPI server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    parser.add_argument("--workers", type=int, default=1, help="Workers")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3.5-2B", help="Model ID")
    parser.add_argument("--adapter_path", type=str, default=None, help="LoRA adapter directory")
    args = parser.parse_args()

    os.environ["SLM_MODEL_ID"] = args.model_id
    if args.adapter_path:
        os.environ["SLM_ADAPTER_PATH"] = args.adapter_path

    print(f"Starting server on http://{args.host}:{args.port}")
    uvicorn.run(
        "slm_sms.inference.server:app", host=args.host, port=args.port, workers=args.workers
    )


if __name__ == "__main__":
    main()
