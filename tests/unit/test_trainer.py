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

    def forward(self, input_ids=None, **kwargs):
        from transformers.modeling_outputs import BaseModelOutputWithPast

        return BaseModelOutputWithPast(last_hidden_state=self.embed(input_ids))


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
    assert (tmp_path / "lora_smoke").exists()
