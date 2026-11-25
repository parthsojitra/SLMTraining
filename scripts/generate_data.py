#!/usr/bin/env python3
"""Script to generate synthetic Indian SMS dataset."""

import argparse
from pathlib import Path

from slm_sms.core.config import load_yaml_config
from slm_sms.data.dataset import save_jsonl
from slm_sms.data.generator import IndianSMSGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Indian SMS data")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config (config/generation_config.yaml)",
    )
    parser.add_argument("--samples", "-n", type=int, default=2000, help="Total samples to generate")
    parser.add_argument("--output_dir", "-o", type=str, default="./data", help="Output directory")
    parser.add_argument("--val_split", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--test_split", type=float, default=0.05, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.config:
        ds_cfg = load_yaml_config(args.config).get("dataset", {})
        args.samples = int(ds_cfg.get("samples", args.samples))
        args.output_dir = str(ds_cfg.get("output_dir", args.output_dir))
        args.val_split = float(ds_cfg.get("val_split", args.val_split))
        args.test_split = float(ds_cfg.get("test_split", args.test_split))
        args.seed = int(ds_cfg.get("seed", args.seed))

    out_path = Path(args.output_dir)
    print(f"Generating {args.samples} Indian SMS samples into {out_path}...")
    generator = IndianSMSGenerator(seed=args.seed)
    splits = generator.generate_dataset(
        n_samples=args.samples,
        val_split=args.val_split,
        test_split=args.test_split,
    )

    train_file = save_jsonl(splits["train"], out_path / "train.jsonl")
    val_file = save_jsonl(splits["validation"], out_path / "val.jsonl")
    test_file = save_jsonl(splits["test"], out_path / "test.jsonl")

    print("Successfully generated:")
    print(f"  - Train:      {len(splits['train'])} samples -> {train_file}")
    print(f"  - Validation: {len(splits['validation'])} samples -> {val_file}")
    print(f"  - Test:       {len(splits['test'])} samples -> {test_file}")


if __name__ == "__main__":
    main()
