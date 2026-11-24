#!/usr/bin/env python3
"""Run SMS entity extraction inference on a sample message or input text."""

import argparse
import json

from slm_sms.inference.engine import SMSExtractorEngine


def main():
    parser = argparse.ArgumentParser(description="Extract structured JSON from Indian SMS")
    parser.add_argument(
        "--text",
        "-t",
        type=str,
        default="[VM-HDFCBK] Rs.1,450.00 debited from A/c **4821 on 23-11-26 to swiggy@axisbank (UPI Ref: 432819284129). Avl Bal: Rs.34,210.50.",
        help="SMS text to parse",
    )
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3.5-2B", help="Model ID")
    parser.add_argument("--adapter_path", type=str, default=None, help="LoRA adapter directory")
    args = parser.parse_args()

    print(f"\nAnalyzing SMS:\n  '{args.text}'\n")
    engine = SMSExtractorEngine(model_id_or_path=args.model_id, adapter_path=args.adapter_path)
    result = engine.extract(args.text)

    print("Extracted Entity JSON:")
    print(json.dumps(result["entity"], indent=2, ensure_ascii=False))
    print(f"\nLatency: {result['latency_ms']} ms | Fallback used: {result['fallback_used']}")


if __name__ == "__main__":
    main()
