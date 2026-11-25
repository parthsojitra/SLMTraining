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


class DummyModel(PreTrainedModel):
    config_class = DummyConfig

    def __init__(self, config):
        super().__init__(config)
        self.embed = nn.Embedding(200, 16)
        self.linear = nn.Linear(16, 200)
        self.generation_config = GenerationConfig()

    def forward(self, input_ids=None, labels=None, **kwargs):
        x = self.embed(input_ids)
        logits = self.linear(x)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, 200), labels.view(-1))
        from transformers.modeling_outputs import CausalLMOutputWithPast

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
