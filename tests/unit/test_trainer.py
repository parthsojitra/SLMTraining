"""Unit tests for the training pipeline and SFTTrainer integration."""

import pytest
from datasets import Dataset, DatasetDict
from torch import nn
from transformers import (
    GenerationConfig,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)
from trl import SFTConfig, SFTTrainer

from slm_sms.training.trainer import SMSTrainingPipeline


class DummyConfig(PretrainedConfig):
    model_type = "dummy"
    vocab_size = 200
    hidden_size = 16


class DummyBackbone(nn.Module):
    def __init__(self, vocab_size: int = 200, hidden_size: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids=None, **kwargs):
        from transformers.modeling_outputs import BaseModelOutputWithPast

        return BaseModelOutputWithPast(last_hidden_state=self.proj(self.embed(input_ids)))


class DummyModel(PreTrainedModel):
    config_class = DummyConfig
    base_model_prefix = "model"

    def __init__(self, config):
        super().__init__(config)
        self.model = DummyBackbone(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.generation_config = GenerationConfig()

    def get_output_embeddings(self):
        return self.lm_head

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

    def forward(self, input_ids=None, labels=None, **kwargs):
        from transformers.modeling_outputs import CausalLMOutputWithPast

        outputs = self.model(input_ids=input_ids, **kwargs)
        logits = self.lm_head(outputs.last_hidden_state)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits[:, :-1, :].reshape(-1, self.config.vocab_size),
                labels[:, 1:].reshape(-1),
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)


@pytest.fixture
def dummy_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    tok = Tokenizer(
        WordLevel(
            {
                "<unk>": 0,
                "<pad>": 1,
                "<eos>": 2,
                "<|im_start|>": 3,
                "<|im_end|>": 4,
                "user": 5,
                "assistant": 6,
                "system": 7,
                "hi": 8,
            },
            unk_token="<unk>",
        )
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok)
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "<eos>"
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
        "{% endfor %}"
    )
    return tokenizer


def test_training_pipeline_setup(tmp_path, dummy_tokenizer):
    pipeline = SMSTrainingPipeline(
        model_id="Qwen/Qwen3.5-2B",
        output_dir=tmp_path / "lora_out",
    )

    # Inject mock model and tokenizer to test configuration without downloading weights
    pipeline.model = DummyModel(DummyConfig())
    pipeline.tokenizer = dummy_tokenizer

    ds = DatasetDict(
        {
            "train": Dataset.from_list(
                [
                    {
                        "messages": [
                            {"role": "system", "content": "Extract JSON"},
                            {"role": "user", "content": "Rs 100 debited from A/c 1234"},
                            {"role": "assistant", "content": '{"amount":100.0}'},
                        ]
                    }
                ]
            ),
            "validation": Dataset.from_list(
                [
                    {
                        "messages": [
                            {"role": "system", "content": "Extract JSON"},
                            {"role": "user", "content": "Rs 200 debited from A/c 5678"},
                            {"role": "assistant", "content": '{"amount":200.0}'},
                        ]
                    }
                ]
            ),
        }
    )

    # Verify SFTTrainer creation via train method parameters logic
    from slm_sms.models.model_loader import get_device_and_dtype

    _, _, dev_name = get_device_and_dtype()
    use_cpu = dev_name == "cpu"

    sft_config = SFTConfig(
        output_dir=str(tmp_path / "lora_out"),
        max_length=64,
        use_cpu=use_cpu,
        fp16=False,
        bf16=False,
        num_train_epochs=1,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=pipeline.model,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=pipeline.tokenizer,
        args=sft_config,
    )

    assert trainer is not None
    assert trainer.train_dataset is not None
    assert trainer.eval_dataset is not None


def test_training_pipeline_train_smoke(tmp_path, dummy_tokenizer):
    """Execute pipeline.train() end-to-end with a tiny dummy model to catch SFTConfig/TRL kwargs drift."""
    pipeline = SMSTrainingPipeline(
        model_id="Qwen/Qwen3.5-2B",
        output_dir=tmp_path / "lora_smoke",
    )
    pipeline.model = DummyModel(DummyConfig())
    pipeline.tokenizer = dummy_tokenizer

    ds = DatasetDict(
        {
            "train": Dataset.from_list(
                [
                    {
                        "messages": [
                            {"role": "system", "content": "hi"},
                            {"role": "user", "content": "hi user"},
                            {"role": "assistant", "content": "hi assistant hi"},
                        ]
                    }
                ]
            ),
            "validation": Dataset.from_list(
                [
                    {
                        "messages": [
                            {"role": "system", "content": "hi"},
                            {"role": "user", "content": "hi user"},
                            {"role": "assistant", "content": "hi assistant hi"},
                        ]
                    }
                ]
            ),
        }
    )

    metrics = pipeline.train(
        dataset=ds,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        max_length=64,
        warmup_ratio=0.05,
        logging_steps=1,
        eval_steps=1,
    )
    assert "train_loss" in metrics
    assert "eval_loss" in metrics
    assert metrics["run_id"]
    run_dir = tmp_path / "lora_smoke" / "runs" / metrics["run_id"]
    latest_link = tmp_path / "lora_smoke" / "latest"
    assert run_dir.exists()
    assert (run_dir / "run_metadata.json").exists()
    assert latest_link.is_symlink()
    assert latest_link.resolve() == run_dir.resolve()


def test_versioned_runs_preserve_history_and_update_latest(tmp_path, dummy_tokenizer):
    """Running multiple trainings preserves older runs, avoids collisions, and promotes `latest`."""
    import json

    from slm_sms.models.model_loader import resolve_adapter_path

    base_out = tmp_path / "versioned_out"
    ds = DatasetDict(
        {
            "train": Dataset.from_list(
                [
                    {
                        "messages": [
                            {"role": "system", "content": "hi"},
                            {"role": "user", "content": "hi user"},
                            {"role": "assistant", "content": "hi assistant hi"},
                        ]
                    }
                ]
            )
        }
    )

    # First run with custom run_name="exp" and change description
    p1 = SMSTrainingPipeline(
        model_id="Qwen/Qwen3.5-2B",
        output_dir=base_out,
        run_name="exp",
        description="Initial baseline with 800 samples",
    )
    p1.model = DummyModel(DummyConfig())
    p1.tokenizer = dummy_tokenizer
    m1 = p1.train(
        dataset=ds,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        eval_steps=None,
        logging_steps=1,
    )
    (p1.output_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    # Second run with the same run_name="exp" must auto-suffix to "exp_1" without overwriting "exp"
    p2 = SMSTrainingPipeline(
        model_id="Qwen/Qwen3.5-2B",
        output_dir=base_out,
        run_name="exp",
        description="Increased dataset to 2500 samples + lakh formatting",
    )
    p2.model = DummyModel(DummyConfig())
    p2.tokenizer = dummy_tokenizer
    m2 = p2.train(
        dataset=ds,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        eval_steps=None,
        logging_steps=1,
    )
    (p2.output_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    assert m1["run_id"] == "exp"
    assert m1["description"] == "Initial baseline with 800 samples"
    assert m2["run_id"] == "exp_1"
    assert m2["description"] == "Increased dataset to 2500 samples + lakh formatting"
    assert (base_out / "runs" / "exp" / "run_metadata.json").exists()
    assert (base_out / "runs" / "exp_1" / "run_metadata.json").exists()

    latest_link = base_out / "latest"
    assert latest_link.is_symlink()
    assert latest_link.resolve() == p2.output_dir.resolve()
    assert resolve_adapter_path(base_out) == str(p2.output_dir.resolve())

    meta2 = json.loads((p2.output_dir / "run_metadata.json").read_text())
    assert meta2["run_id"] == "exp_1"
    assert meta2["versioned"] is True
    assert meta2["description"] == "Increased dataset to 2500 samples + lakh formatting"
    assert meta2["dataset"]["train_samples"] == 1
    assert meta2["lora"]["r"] == 16
    assert meta2["training_params"]["num_train_epochs"] == 1

    from slm_sms.training import list_training_runs

    listed = list_training_runs(base_out)
    assert [r["run_id"] for r in listed] == ["exp_1", "exp"]
    assert listed[0]["is_latest"] is True
    assert listed[1]["is_latest"] is False


def test_unversioned_run_writes_directly_to_output_dir(tmp_path, dummy_tokenizer):
    """versioned=False disables runs/<timestamp>/ and latest symlink creation."""
    out = tmp_path / "flat_out"
    ds = DatasetDict(
        {
            "train": Dataset.from_list(
                [
                    {
                        "messages": [
                            {"role": "system", "content": "hi"},
                            {"role": "user", "content": "hi user"},
                            {"role": "assistant", "content": "hi assistant hi"},
                        ]
                    }
                ]
            )
        }
    )
    p = SMSTrainingPipeline(model_id="Qwen/Qwen3.5-2B", output_dir=out, versioned=False)
    p.model = DummyModel(DummyConfig())
    p.tokenizer = dummy_tokenizer
    metrics = p.train(
        dataset=ds,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        eval_steps=None,
        logging_steps=1,
    )
    assert metrics["output_dir"] == str(out)
    assert (out / "run_metadata.json").exists()
    assert not (out / "runs").exists()
    assert not (out / "latest").exists()


def test_warm_start_from_existing_peft_adapter(tmp_path, dummy_tokenizer):
    """Verify setup_peft_model + SMSTrainingPipeline warm-start from a saved LoRA adapter."""
    import json

    from slm_sms.models.model_loader import get_lora_config, setup_peft_model

    # Build a real PEFT LoRA adapter on DummyModel (targeting DummyBackbone's `proj` Linear layer)
    base_model_v1 = DummyModel(DummyConfig())
    lora_cfg = get_lora_config(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["proj"])
    peft_v1 = setup_peft_model(base_model_v1, lora_config=lora_cfg)

    ds = DatasetDict(
        {
            "train": Dataset.from_list(
                [
                    {
                        "messages": [
                            {"role": "system", "content": "hi"},
                            {"role": "user", "content": "hi user"},
                            {"role": "assistant", "content": "hi assistant hi"},
                        ]
                    }
                ]
            )
        }
    )

    out_root = tmp_path / "lora_versions"
    p1 = SMSTrainingPipeline(
        model_id="Qwen/Qwen3.5-2B",
        output_dir=out_root,
        run_name="v1",
        description="Initial v1 LoRA",
    )
    p1.model = peft_v1
    p1.tokenizer = dummy_tokenizer
    p1.train(
        dataset=ds,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        eval_steps=None,
        logging_steps=1,
    )
    assert (p1.output_dir / "adapter_config.json").exists()

    # Warm-start v2 using base_out ("latest") via setup_peft_model(..., adapter_path=...)
    p2 = SMSTrainingPipeline(
        model_id="Qwen/Qwen3.5-2B",
        output_dir=out_root,
        run_name="v2",
        description="Warm-started from v1",
        init_adapter_path=out_root,  # resolves through `latest` -> runs/v1
    )
    assert p2.init_adapter_path == str(p1.output_dir.resolve())

    base_model_v2 = DummyModel(DummyConfig())
    p2.model = setup_peft_model(base_model_v2, adapter_path=p2.init_adapter_path)
    p2.tokenizer = dummy_tokenizer

    # Ensure loaded LoRA weights are actually trainable (is_trainable=True)
    trainable_params, _ = p2.model.get_nb_trainable_parameters()
    assert trainable_params > 0

    m2 = p2.train(
        dataset=ds,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        eval_steps=None,
        logging_steps=1,
    )
    assert m2["init_adapter_path"] == str(p1.output_dir.resolve())

    meta2 = json.loads((p2.output_dir / "run_metadata.json").read_text())
    assert meta2["lineage"]["init_adapter_path"] == str(p1.output_dir.resolve())
    assert (out_root / "latest").resolve() == p2.output_dir.resolve()


def test_resume_from_checkpoint_restores_step_and_records_lineage(tmp_path, dummy_tokenizer):
    """Verify resume_from_checkpoint continues from a saved checkpoint-* and records lineage."""
    import json

    ds = DatasetDict(
        {
            "train": Dataset.from_list(
                [
                    {
                        "messages": [
                            {"role": "system", "content": "hi"},
                            {"role": "user", "content": "hi user"},
                            {"role": "assistant", "content": "hi assistant hi"},
                        ]
                    },
                    {
                        "messages": [
                            {"role": "system", "content": "hi"},
                            {"role": "user", "content": "hi user"},
                            {"role": "assistant", "content": "hi assistant hi"},
                        ]
                    },
                ]
            )
        }
    )

    out_root = tmp_path / "resume_versions"
    p1 = SMSTrainingPipeline(model_id="Qwen/Qwen3.5-2B", output_dir=out_root, run_name="r1")
    p1.model = DummyModel(DummyConfig())
    p1.tokenizer = dummy_tokenizer
    p1.train(
        dataset=ds,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        eval_steps=None,
        logging_steps=1,
    )
    ckpts = list(p1.output_dir.glob("checkpoint-*"))
    assert len(ckpts) == 1

    # Resume from r1's checkpoint for epoch 2 into a new versioned run r2
    p2 = SMSTrainingPipeline(
        model_id="Qwen/Qwen3.5-2B",
        output_dir=out_root,
        run_name="r2",
        description="Resumed from r1 for epoch 2",
    )
    p2.model = DummyModel(DummyConfig())
    p2.tokenizer = dummy_tokenizer
    m2 = p2.train(
        dataset=ds,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        eval_steps=None,
        logging_steps=1,
        resume_from_checkpoint=p1.output_dir,  # passes run dir -> auto-picks checkpoint-*
    )
    assert m2["resume_from_checkpoint"] == str(ckpts[0].resolve())
    meta2 = json.loads((p2.output_dir / "run_metadata.json").read_text())
    assert meta2["lineage"]["resume_from_checkpoint"] == str(ckpts[0].resolve())


def test_missing_init_or_resume_checkpoint_raises(tmp_path):
    """Fail fast with FileNotFoundError if adapter or checkpoint does not exist."""
    missing = tmp_path / "nonexistent"
    with pytest.raises(FileNotFoundError, match="Initial adapter checkpoint not found"):
        SMSTrainingPipeline(output_dir=tmp_path / "out", init_adapter_path=missing)

    p = SMSTrainingPipeline(output_dir=tmp_path / "out2")
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        p._resolve_resume_checkpoint(missing)
