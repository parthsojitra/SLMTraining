"""Training pipeline for fine-tuning Qwen models using TRL SFTTrainer and PEFT LoRA."""

from __future__ import annotations

import dataclasses
import gc
import inspect
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict
from peft import PeftModel
from transformers import PreTrainedTokenizer
from trl import SFTConfig, SFTTrainer

from slm_sms.core.logging import get_logger
from slm_sms.core.telemetry import get_tracer
from slm_sms.models.model_loader import (
    get_device_and_dtype,
    get_lora_config,
    load_model,
    load_tokenizer,
    setup_peft_model,
)

logger = get_logger(__name__)


class SMSTrainingPipeline:
    """Production training pipeline for Qwen SLM fine-tuning."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3.5-2B",
        output_dir: str | Path = "./output/qwen-sms-lora",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        use_4bit: bool = False,
        use_8bit: bool = False,
    ):
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.lora_config = get_lora_config(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.use_4bit = use_4bit
        self.use_8bit = use_8bit

        self.tokenizer: PreTrainedTokenizer | None = None
        self.model: PeftModel | None = None

    def initialize_model_and_tokenizer(self) -> None:
        """Load tokenizer, base model, and apply LoRA."""
        tracer = get_tracer(__name__)
        with tracer.start_as_current_span("slm.training.init_model") as span:
            span.set_attribute("slm.model.id", self.model_id)
            span.set_attribute("slm.quantized.4bit", self.use_4bit)
            span.set_attribute("slm.quantized.8bit", self.use_8bit)

            logger.info("Initializing training model", extra={"model_id": self.model_id})
            self.tokenizer = load_tokenizer(self.model_id)
            base_model = load_model(
                model_id=self.model_id,
                use_4bit=self.use_4bit,
                use_8bit=self.use_8bit,
            )
            self.model = setup_peft_model(
                base_model,
                lora_config=self.lora_config,
                is_quantized=self.use_4bit or self.use_8bit,
            )

    def train(
        self,
        dataset: DatasetDict | Dataset | dict[str, Any],
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 1,
        per_device_eval_batch_size: int | None = None,
        gradient_accumulation_steps: int = 16,
        learning_rate: float = 2e-4,
        max_length: int = 256,
        warmup_ratio: float = 0.05,
        logging_steps: int = 10,
        eval_steps: int | None = 50,
        save_total_limit: int = 2,
        gradient_checkpointing: bool | None = None,
        load_best_model_at_end: bool = False,
    ) -> dict[str, Any]:
        """Execute SFT training loop on the dataset.

        Args:
            dataset: DatasetDict or Dataset containing 'train' (and optional 'validation').
            num_train_epochs: Number of training epochs.
            per_device_train_batch_size: Training batch size per worker device.
            per_device_eval_batch_size: Eval batch size (defaults to train batch size to avoid OOM).
            gradient_accumulation_steps: Gradient accumulation step count.
            learning_rate: Optimizer initial learning rate.
            max_length: Maximum sequence token length.
            warmup_ratio: Linear warmup ratio.
            logging_steps: Logging cadence in steps.
            eval_steps: Evaluation cadence in steps (if validation dataset present).
            save_total_limit: Maximum checkpoints retained.
            gradient_checkpointing: Enable activation checkpointing (None = auto-enable on GPU/MPS).
            load_best_model_at_end: Load best checkpoint upon completion.

        Returns:
            Dict containing final train and evaluation metrics.
        """
        tracer = get_tracer(__name__)
        with tracer.start_as_current_span("slm.training.run") as run_span:
            run_span.set_attribute("slm.model.id", self.model_id)
            run_span.set_attribute("slm.training.epochs", num_train_epochs)
            run_span.set_attribute("slm.training.batch_size", per_device_train_batch_size)
            run_span.set_attribute("slm.training.learning_rate", learning_rate)

            # Reclaim any lingering tensors on unified/GPU memory before loading or training
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()

            if self.model is None or self.tokenizer is None:
                self.initialize_model_and_tokenizer()

            # Resolve train and evaluation datasets
            if isinstance(dataset, (DatasetDict, dict)):
                if "train" not in dataset:
                    raise ValueError("Dataset dictionary must contain a 'train' split.")
                train_dataset = dataset["train"]
                eval_dataset = (
                    dataset.get("validation", None)
                    or dataset.get("val", None)
                    or dataset.get("eval", None)
                )
            else:
                train_dataset = dataset
                eval_dataset = None

            _, _, dev_name = get_device_and_dtype()
            run_span.set_attribute("slm.device", dev_name)

            # Device & precision resolution
            if dev_name == "cuda":
                use_cpu = False
                bf16 = torch.cuda.is_bf16_supported()
                fp16 = not bf16
            else:
                # For CPU or Apple Silicon MPS, avoid unsupported bf16/gpu flags in TrainingArguments
                use_cpu = dev_name == "cpu"
                bf16 = False
                fp16 = False

            # Auto-enable activation checkpointing on GPU/MPS to keep 2B model inside 20 GB memory
            effective_grad_ckpt = (
                dev_name in ("mps", "cuda")
                if gradient_checkpointing is None
                else gradient_checkpointing
            )
            eval_batch_size = per_device_eval_batch_size or per_device_train_batch_size

            if effective_grad_ckpt and self.model is not None:
                if hasattr(self.model, "config") and hasattr(self.model.config, "use_cache"):
                    self.model.config.use_cache = False
                if hasattr(self.model, "enable_input_require_grads"):
                    self.model.enable_input_require_grads()

            # Configure evaluation and checkpointing strategies
            has_eval = eval_dataset is not None and eval_steps is not None and eval_steps > 0
            eval_strategy = "steps" if has_eval else "no"

            # Build SFTConfig kwargs dynamically to support various transformers/trl versions
            sft_fields = {f.name for f in dataclasses.fields(SFTConfig)}

            config_kwargs: dict[str, Any] = {
                "output_dir": str(self.output_dir),
                "num_train_epochs": num_train_epochs,
                "per_device_train_batch_size": per_device_train_batch_size,
                "per_device_eval_batch_size": eval_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "learning_rate": learning_rate,
                "logging_steps": logging_steps,
                "save_total_limit": save_total_limit,
                "use_cpu": use_cpu,
                "fp16": fp16,
                "bf16": bf16,
                "report_to": "none",
                "optim": "adamw_torch",
            }

            if "prediction_loss_only" in sft_fields:
                config_kwargs["prediction_loss_only"] = True

            if dev_name == "mps" and "torch_empty_cache_steps" in sft_fields:
                config_kwargs["torch_empty_cache_steps"] = 10

            # Gradient/activation checkpointing compatibility (renamed in transformers v5)
            if "activation_checkpointing" in sft_fields:
                config_kwargs["activation_checkpointing"] = effective_grad_ckpt
                if effective_grad_ckpt and "activation_checkpointing_kwargs" in sft_fields:
                    config_kwargs["activation_checkpointing_kwargs"] = {"use_reentrant": False}
            elif "gradient_checkpointing" in sft_fields:
                config_kwargs["gradient_checkpointing"] = effective_grad_ckpt
                if effective_grad_ckpt and "gradient_checkpointing_kwargs" in sft_fields:
                    config_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

            # Warmup compatibility (warmup_ratio removed in transformers v5 in favor of float warmup_steps)
            if "warmup_ratio" in sft_fields:
                config_kwargs["warmup_ratio"] = warmup_ratio
            elif "warmup_steps" in sft_fields:
                config_kwargs["warmup_steps"] = warmup_ratio

            # Sequence length kwarg (max_length in modern TRL, max_seq_length in legacy)
            if "max_length" in sft_fields:
                config_kwargs["max_length"] = max_length
            elif "max_seq_length" in sft_fields:
                config_kwargs["max_seq_length"] = max_length

            # Evaluation & saving strategy kwarg
            if "eval_strategy" in sft_fields:
                config_kwargs["eval_strategy"] = eval_strategy
            elif "evaluation_strategy" in sft_fields:
                config_kwargs["evaluation_strategy"] = eval_strategy

            if has_eval:
                config_kwargs["eval_steps"] = eval_steps
                config_kwargs["save_strategy"] = "steps"
                config_kwargs["save_steps"] = eval_steps
                config_kwargs["load_best_model_at_end"] = load_best_model_at_end
                if load_best_model_at_end:
                    config_kwargs["metric_for_best_model"] = "loss"
            else:
                config_kwargs["save_strategy"] = "epoch"
                config_kwargs["load_best_model_at_end"] = False

            sft_config = SFTConfig(**config_kwargs)

            logger.info(
                "Configured SFT Trainer",
                extra={
                    "device": dev_name,
                    "use_cpu": use_cpu,
                    "fp16": fp16,
                    "bf16": bf16,
                    "eval_strategy": eval_strategy,
                },
            )

            # SFTTrainer argument compatibility (processing_class in modern TRL vs tokenizer in older)
            trainer_kwargs: dict[str, Any] = {
                "model": self.model,
                "train_dataset": train_dataset,
                "eval_dataset": eval_dataset,
                "args": sft_config,
            }

            sft_init_params = inspect.signature(SFTTrainer.__init__).parameters
            if "processing_class" in sft_init_params:
                trainer_kwargs["processing_class"] = self.tokenizer
            elif "tokenizer" in sft_init_params:
                trainer_kwargs["tokenizer"] = self.tokenizer

            trainer = SFTTrainer(**trainer_kwargs)

            logger.info("Starting fine-tuning")
            with tracer.start_as_current_span("slm.training.sft") as sft_span:
                train_result = trainer.train()
                for k, v in train_result.metrics.items():
                    if isinstance(v, (int, float)):
                        sft_span.set_attribute(f"slm.train.{k}", v)

            # Save the fine-tuned LoRA adapter and tokenizer
            logger.info("Saving fine-tuned adapter", extra={"output_dir": str(self.output_dir)})
            trainer.save_model(str(self.output_dir))
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(str(self.output_dir))

            metrics = dict(train_result.metrics)
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

            if eval_dataset is not None:
                logger.info("Running final evaluation on validation set")
                with tracer.start_as_current_span("slm.training.eval") as eval_span:
                    eval_metrics = trainer.evaluate()
                    for k, v in eval_metrics.items():
                        if isinstance(v, (int, float)):
                            eval_span.set_attribute(f"slm.eval.{k}", v)
                trainer.log_metrics("eval", eval_metrics)
                trainer.save_metrics("eval", eval_metrics)
                metrics.update(eval_metrics)

            return metrics
