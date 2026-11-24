#!/usr/bin/env python3
"""Script to fine-tune Qwen on Indian SMS data."""

import argparse

from slm_sms.data.dataset import load_hf_dataset_from_dir
from slm_sms.training.trainer import SMSTrainingPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen model with LoRA on Indian SMS data"
    )
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3.5-2B", help="Model ID")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument(
        "--output_dir", type=str, default="./output/qwen-sms-lora", help="Output directory"
    )
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit QLoRA (CUDA)")
    args = parser.parse_args()

    print(f"Loading dataset from {args.data_dir}...")
    dataset = load_hf_dataset_from_dir(args.data_dir)

    print(f"Initializing pipeline for {args.model_id}...")
    pipeline = SMSTrainingPipeline(
        model_id=args.model_id,
        output_dir=args.output_dir,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        use_4bit=args.use_4bit,
    )

    print("Beginning training...")
    metrics = pipeline.train(
        dataset=dataset,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
    )
    print(f"Training finished successfully! Metrics: {metrics}")


if __name__ == "__main__":
    main()
