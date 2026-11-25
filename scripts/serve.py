#!/usr/bin/env python3
"""Run production FastAPI microservice.

Development (single worker, auto-reload):
    uv run python scripts/serve.py --reload

Production (gunicorn, recommended):
    gunicorn slm_sms.inference.server:app -c gunicorn.conf.py
"""

import argparse
import os

# Load .env file for local development (python-dotenv)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import uvicorn

from slm_sms.core.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Start SMS extraction FastAPI server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of workers (keep at 1 per GPU)"
    )
    parser.add_argument("--model_id", type=str, default=None, help="Model ID or local path")
    parser.add_argument(
        "--model_revision", type=str, default=None, help="Model revision/commit hash to pin"
    )
    parser.add_argument("--adapter_path", type=str, default=None, help="LoRA adapter directory")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev only)")
    args = parser.parse_args()

    if args.model_id:
        os.environ["SLM_MODEL_ID"] = args.model_id
    if args.model_revision:
        os.environ["SLM_MODEL_REVISION"] = args.model_revision
    if args.adapter_path:
        os.environ["SLM_ADAPTER_PATH"] = args.adapter_path

    logger.info(
        "Starting uvicorn server",
        extra={"host": args.host, "port": args.port, "workers": args.workers},
    )
    uvicorn.run(
        "slm_sms.inference.server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_config=None,  # Let our centralized logger handle it
    )


if __name__ == "__main__":
    main()
