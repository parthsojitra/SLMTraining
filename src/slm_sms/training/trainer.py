"""Training pipeline for fine-tuning Qwen models using TRL SFTTrainer and PEFT LoRA."""

from __future__ import annotations

import dataclasses
import gc
import inspect
import json
import os
import subprocess
from datetime import UTC, datetime
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
    resolve_adapter_path,
    setup_peft_model,
)

logger = get_logger(__name__)


def _collect_git_context() -> dict[str, Any] | None:
    """Capture git commit, branch, and dirty status cleanly without failing outside git."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if commit.returncode != 0:
            return None
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        return {
            "commit": commit.stdout.strip(),
            "branch": branch.stdout.strip() if branch.returncode == 0 else None,
            "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        }
    except Exception:
        return None


def list_training_runs(output_dir: str | Path = "./output/qwen-sms-lora") -> list[dict[str, Any]]:
    """List versioned training runs sorted newest-first with `is_latest` status."""
    base = Path(output_dir)
    runs_dir = base / "runs"
    if not runs_dir.exists():
        return []

    latest_link = base / "latest"
    latest_resolved = latest_link.resolve() if latest_link.exists() else None

    entries: list[dict[str, Any]] = []
    for run_path in sorted(runs_dir.iterdir(), key=lambda p: p.name, reverse=True):
        if not run_path.is_dir():
            continue
        meta_file = run_path / "run_metadata.json"
        if meta_file.exists():
            try:
                data = json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception:
                data = {"run_id": run_path.name}
        else:
            data = {"run_id": run_path.name}
        data["output_dir"] = str(run_path)
        data["is_latest"] = bool(latest_resolved and run_path.resolve() == latest_resolved)
        entries.append(data)
    return entries


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
        versioned: bool = True,
        run_name: str | None = None,
        description: str | None = None,
        init_adapter_path: str | Path | None = None,
    ):
        self.model_id = model_id
        self.base_output_dir = Path(output_dir)
        self.versioned = versioned
        self.description = description
        self.latest_path = self.base_output_dir / "latest"

        self.init_adapter_path: str | None = resolve_adapter_path(init_adapter_path)
        if (
            self.init_adapter_path is not None
            and not (Path(self.init_adapter_path) / "adapter_config.json").exists()
        ):
            raise FileNotFoundError(
                f"Initial adapter checkpoint not found (missing adapter_config.json): "
                f"{self.init_adapter_path}"
            )

        self.run_id, self.output_dir = self._allocate_run_dir(run_name)
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

    def _allocate_run_dir(self, run_name: str | None = None) -> tuple[str, Path]:
        """Determine the output directory for this training run.

        When ``versioned=True``, creates an immutable directory under
        ``<base_output_dir>/runs/<timestamp_or_run_name>/`` (with collision-safe
        suffixing) so previous runs are never overwritten.
        """
        if not self.versioned:
            run_id = run_name or "default"
            return run_id, self.base_output_dir

        base_id = run_name or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        runs_root = self.base_output_dir / "runs"
        candidate = runs_root / base_id
        run_id = base_id
        suffix = 1
        while candidate.exists():
            run_id = f"{base_id}_{suffix}"
            candidate = runs_root / run_id
            suffix += 1
        return run_id, candidate

    def _update_latest_symlink(self) -> None:
        """Atomically update ``<base_output_dir>/latest`` -> ``runs/<run_id>``."""
        if not self.versioned:
            return

        relative_target = Path("runs") / self.run_id
        tmp_link = self.base_output_dir / f".latest.tmp.{os.getpid()}"
        if tmp_link.exists() or tmp_link.is_symlink():
            tmp_link.unlink()

        try:
            tmp_link.symlink_to(relative_target)
            os.replace(tmp_link, self.latest_path)
        except OSError:
            # Fallback if atomic symlink replacement isn't supported
            if tmp_link.exists() or tmp_link.is_symlink():
                tmp_link.unlink()
            if self.latest_path.exists() or self.latest_path.is_symlink():
                self.latest_path.unlink()
            self.latest_path.symlink_to(relative_target)

        logger.info(
            "Updated latest adapter symlink",
            extra={"latest_path": str(self.latest_path), "target": str(relative_target)},
        )

    def _resolve_resume_checkpoint(
        self, resume_from_checkpoint: str | Path | bool | None
    ) -> str | None:
        """Resolve and validate a checkpoint path for exact Trainer state resumption."""
        if resume_from_checkpoint is None or resume_from_checkpoint is False:
            return None

        def _latest_checkpoint_in(directory: Path) -> Path | None:
            if not directory.exists():
                return None
            candidates = [
                p
                for p in directory.iterdir()
                if p.is_dir()
                and p.name.startswith("checkpoint-")
                and p.name.split("-")[-1].isdigit()
            ]
            if not candidates:
                return None
            return max(candidates, key=lambda p: int(p.name.split("-")[-1]))

        if resume_from_checkpoint is True:
            search_dirs = [self.output_dir]
            if self.latest_path.exists():
                search_dirs.append(self.latest_path.resolve())
            search_dirs.append(self.base_output_dir)
            for candidate_dir in search_dirs:
                found = _latest_checkpoint_in(candidate_dir)
                if found is not None:
                    return str(found.resolve())
            raise FileNotFoundError(
                f"No checkpoint-* directory found under {self.output_dir} or {self.base_output_dir}"
            )

        raw = Path(resume_from_checkpoint)
        resolved = Path(resolve_adapter_path(raw) or raw)
        nested = _latest_checkpoint_in(resolved)
        if nested is not None:
            return str(nested.resolve())
        if (resolved / "trainer_state.json").exists():
            return str(resolved.resolve())
        if (raw / "trainer_state.json").exists():
            return str(raw.resolve())

        raise FileNotFoundError(
            f"Checkpoint not found (missing trainer_state.json or checkpoint-*): {resume_from_checkpoint}"
        )

    def _write_run_metadata(
        self,
        metrics: dict[str, Any],
        *,
        description: str | None = None,
        dataset_info: dict[str, int] | None = None,
        training_params: dict[str, Any] | None = None,
        lineage: dict[str, Any] | None = None,
    ) -> None:
        """Persist rich reproducibility & change metadata alongside the saved adapter."""
        target_modules = getattr(self.lora_config, "target_modules", None)
        if isinstance(target_modules, set):
            target_modules = sorted(target_modules)

        metadata = {
            "run_id": self.run_id,
            "description": description if description is not None else self.description,
            "model_id": self.model_id,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "output_dir": str(self.output_dir),
            "versioned": self.versioned,
            "lineage": lineage or {},
            "git": _collect_git_context(),
            "dataset": dataset_info or {},
            "lora": {
                "r": self.lora_config.r,
                "lora_alpha": self.lora_config.lora_alpha,
                "lora_dropout": self.lora_config.lora_dropout,
                "target_modules": list(target_modules) if target_modules else [],
                "use_4bit": self.use_4bit,
                "use_8bit": self.use_8bit,
            },
            "training_params": training_params or {},
            "metrics": metrics,
        }
        meta_file = self.output_dir / "run_metadata.json"
        meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def initialize_model_and_tokenizer(self, adapter_path: str | Path | None = None) -> None:
        """Load tokenizer, base model, and apply fresh or checkpoint LoRA weights."""
        effective_adapter = resolve_adapter_path(adapter_path) or self.init_adapter_path
        tracer = get_tracer(__name__)
        with tracer.start_as_current_span("slm.training.init_model") as span:
            span.set_attribute("slm.model.id", self.model_id)
            span.set_attribute("slm.quantized.4bit", self.use_4bit)
            span.set_attribute("slm.quantized.8bit", self.use_8bit)
            if effective_adapter:
                span.set_attribute("slm.training.init_adapter", effective_adapter)

            logger.info(
                "Initializing training model",
                extra={"model_id": self.model_id, "init_adapter": effective_adapter},
            )
            tok_source = self.model_id
            if effective_adapter and (Path(effective_adapter) / "tokenizer_config.json").exists():
                tok_source = effective_adapter
            self.tokenizer = load_tokenizer(tok_source)
            base_model = load_model(
                model_id=self.model_id,
                use_4bit=self.use_4bit,
                use_8bit=self.use_8bit,
            )
            self.model = setup_peft_model(
                base_model,
                lora_config=self.lora_config,
                is_quantized=self.use_4bit or self.use_8bit,
                adapter_path=effective_adapter,
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
        description: str | None = None,
        resume_from_checkpoint: str | Path | bool | None = None,
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
        run_description = description if description is not None else self.description
        resolved_resume = self._resolve_resume_checkpoint(resume_from_checkpoint)
        tracer = get_tracer(__name__)
        with tracer.start_as_current_span("slm.training.run") as run_span:
            run_span.set_attribute("slm.model.id", self.model_id)
            run_span.set_attribute("slm.training.run_id", self.run_id)
            run_span.set_attribute("slm.training.output_dir", str(self.output_dir))
            if run_description:
                run_span.set_attribute("slm.training.description", run_description)
            if self.init_adapter_path:
                run_span.set_attribute("slm.training.init_adapter", self.init_adapter_path)
            if resolved_resume:
                run_span.set_attribute("slm.training.resumed_from", resolved_resume)
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
                fallback_adapter = (
                    resolved_resume
                    if (
                        self.init_adapter_path is None
                        and resolved_resume
                        and (Path(resolved_resume) / "adapter_config.json").exists()
                    )
                    else None
                )
                self.initialize_model_and_tokenizer(adapter_path=fallback_adapter)

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

            logger.info(
                "Starting fine-tuning",
                extra={
                    "resume_from_checkpoint": resolved_resume,
                    "init_adapter": self.init_adapter_path,
                },
            )
            with tracer.start_as_current_span("slm.training.sft") as sft_span:
                train_result = trainer.train(resume_from_checkpoint=resolved_resume)
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

            # Persist run metadata and atomically promote `latest` pointer only after full success
            dataset_info: dict[str, int] = {"train_samples": len(train_dataset)}
            if eval_dataset is not None:
                dataset_info["eval_samples"] = len(eval_dataset)

            training_params = {
                "num_train_epochs": num_train_epochs,
                "per_device_train_batch_size": per_device_train_batch_size,
                "per_device_eval_batch_size": eval_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "learning_rate": learning_rate,
                "max_length": max_length,
                "warmup_ratio": warmup_ratio,
                "gradient_checkpointing": effective_grad_ckpt,
                "eval_steps": eval_steps if has_eval else None,
                "device": dev_name,
            }
            lineage = {
                "init_adapter_path": self.init_adapter_path,
                "resume_from_checkpoint": resolved_resume,
            }
            self._write_run_metadata(
                metrics,
                description=run_description,
                dataset_info=dataset_info,
                training_params=training_params,
                lineage=lineage,
            )
            self._update_latest_symlink()

            metrics["run_id"] = self.run_id
            if run_description:
                metrics["description"] = run_description
            if self.init_adapter_path:
                metrics["init_adapter_path"] = self.init_adapter_path
            if resolved_resume:
                metrics["resume_from_checkpoint"] = resolved_resume
            metrics["output_dir"] = str(self.output_dir)
            if self.versioned:
                metrics["latest_path"] = str(self.latest_path)

            return metrics
