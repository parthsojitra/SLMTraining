#!/usr/bin/env python3
"""Script to fine-tune Qwen on Indian SMS data."""

import argparse

from slm_sms.core.config import load_yaml_config, settings
from slm_sms.data.dataset import load_hf_dataset_from_dir
from slm_sms.training.trainer import SMSTrainingPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen model with LoRA on Indian SMS data"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Optional YAML config (config/train_config.yaml)"
    )
    parser.add_argument("--model_id", type=str, default=settings.model_id, help="Model ID")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument(
        "--output_dir", type=str, default="./output/qwen-sms-lora", help="Output directory"
    )
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--grad_accum", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=256, help="Max sequence length")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit QLoRA (CUDA)")
    parser.add_argument("--run_name", type=str, default=None, help="Optional versioned run name")
    parser.add_argument(
        "--description",
        "-D",
        type=str,
        default=None,
        help="Note describing what changed in this versioned run",
    )
    parser.add_argument(
        "--init_adapter",
        type=str,
        default=None,
        help="Existing LoRA adapter path to continue training from (fresh optimizer, new version)",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Trainer checkpoint directory (e.g. checkpoint-125) to resume exact state",
    )
    parser.add_argument(
        "--no_versioned",
        action="store_true",
        help="Write directly to output_dir without runs/<timestamp> versioning",
    )
    args = parser.parse_args()

    eval_batch_size = None
    warmup_ratio = 0.05
    logging_steps = 10
    eval_steps = 50
    save_total_limit = 2
    gradient_checkpointing = None
    versioned = not args.no_versioned
    description = args.description
    init_adapter_path = args.init_adapter
    resume_from_checkpoint = args.resume_from_checkpoint

    if args.config:
        raw_cfg = load_yaml_config(args.config)
        m_cfg = raw_cfg.get("model", {})
        l_cfg = raw_cfg.get("lora", {})
        t_cfg = raw_cfg.get("training", {})
        args.model_id = str(m_cfg.get("model_id", args.model_id))
        args.use_4bit = bool(m_cfg.get("use_4bit", args.use_4bit))
        args.lora_r = int(l_cfg.get("r", args.lora_r))
        args.lora_alpha = int(l_cfg.get("lora_alpha", args.lora_alpha))
        args.output_dir = str(t_cfg.get("output_dir", args.output_dir))
        if not args.no_versioned:
            versioned = bool(t_cfg.get("versioned", versioned))
        if description is None:
            description = t_cfg.get("description")
        if init_adapter_path is None:
            init_adapter_path = t_cfg.get("init_adapter_path")
        if resume_from_checkpoint is None:
            resume_from_checkpoint = t_cfg.get("resume_from_checkpoint")
        args.epochs = int(t_cfg.get("num_train_epochs", args.epochs))
        args.batch_size = int(t_cfg.get("per_device_train_batch_size", args.batch_size))
        eval_batch_size = t_cfg.get("per_device_eval_batch_size")
        args.grad_accum = int(t_cfg.get("gradient_accumulation_steps", args.grad_accum))
        args.lr = float(t_cfg.get("learning_rate", args.lr))
        args.max_length = int(t_cfg.get("max_length", args.max_length))
        warmup_ratio = float(t_cfg.get("warmup_ratio", warmup_ratio))
        logging_steps = int(t_cfg.get("logging_steps", logging_steps))
        eval_steps = t_cfg.get("eval_steps", eval_steps)
        save_total_limit = int(t_cfg.get("save_total_limit", save_total_limit))
        gradient_checkpointing = t_cfg.get("gradient_checkpointing")

    print(f"Loading dataset from {args.data_dir}...")
    dataset = load_hf_dataset_from_dir(args.data_dir)

    print(f"Initializing pipeline for {args.model_id}...")
    pipeline = SMSTrainingPipeline(
        model_id=args.model_id,
        output_dir=args.output_dir,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        use_4bit=args.use_4bit,
        versioned=versioned,
        run_name=args.run_name,
        description=description,
        init_adapter_path=init_adapter_path,
    )

    print("Beginning training...")
    metrics = pipeline.train(
        dataset=dataset,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_length=args.max_length,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        save_total_limit=save_total_limit,
        gradient_checkpointing=gradient_checkpointing,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    print(f"Training finished successfully! Metrics: {metrics}")


if __name__ == "__main__":
    main()
