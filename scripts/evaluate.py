#!/usr/bin/env python3
"""Script to evaluate and benchmark model extraction accuracy."""

import argparse
import json

from slm_sms.data.dataset import load_jsonl
from slm_sms.inference.engine import SMSExtractorEngine
from slm_sms.training.metrics import compute_extraction_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on SMS test set")
    parser.add_argument(
        "--test_file", type=str, default="./data/test.jsonl", help="Test JSONL file"
    )
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3.5-2B", help="Model ID")
    parser.add_argument("--adapter_path", type=str, default=None, help="LoRA adapter directory")
    parser.add_argument("--max_samples", type=int, default=100, help="Max test samples to evaluate")
    args = parser.parse_args()

    print(f"Loading test records from {args.test_file} (max {args.max_samples})...")
    records = load_jsonl(args.test_file)[: args.max_samples]

    print(f"Loading engine with model='{args.model_id}', adapter='{args.adapter_path}'...")
    engine = SMSExtractorEngine(model_id_or_path=args.model_id, adapter_path=args.adapter_path)

    predictions = []
    ground_truths = []

    print("Running evaluation...")
    for idx, rec in enumerate(records, start=1):
        sms_text = rec["messages"][1]["content"]
        gt_json = rec["messages"][2]["content"]
        res = engine.extract(sms_text)
        predictions.append(json.dumps(res["entity"]))
        ground_truths.append(json.loads(gt_json))
        if idx % 20 == 0 or idx == len(records):
            print(f"  Processed {idx}/{len(records)} samples...")

    report = compute_extraction_metrics(predictions, ground_truths)
    print("\n" + "=" * 50)
    print("EXTRACTION BENCHMARK RESULTS")
    print("=" * 50)
    for k, v in report.to_dict().items():
        print(f"  {k:22s}: {v}")
    print("=" * 50)


if __name__ == "__main__":
    main()
