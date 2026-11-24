#!/usr/bin/env python3
"""Export merged standalone Hugging Face model from base model + LoRA adapter."""

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3.5-2B", help="Base model ID")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to LoRA adapter")
    parser.add_argument("--output_dir", type=str, required=True, help="Path for merged model")
    parser.add_argument(
        "--safe_serialization", action="store_true", default=True, help="Save safetensors"
    )
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer from {args.adapter_path} or {args.base_model}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    print(f"Loading base model '{args.base_model}' in bfloat16/float16...")
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Applying LoRA adapter from {args.adapter_path}...")
    peft_model = PeftModel.from_pretrained(base_model, args.adapter_path)

    print("Merging LoRA weights into base model...")
    merged_model = peft_model.merge_and_unload()

    print(f"Saving merged standalone model to {out_path}...")
    merged_model.save_pretrained(out_path, safe_serialization=args.safe_serialization)
    tokenizer.save_pretrained(out_path)

    print(f"Successfully exported merged model to: {out_path}")


if __name__ == "__main__":
    main()
