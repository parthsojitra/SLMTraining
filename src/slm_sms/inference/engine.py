"""Inference engine for high-reliability SMS entity extraction."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import torch
from opentelemetry.trace import Status, StatusCode
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from slm_sms.core.config import settings
from slm_sms.core.logging import get_logger
from slm_sms.core.telemetry import get_metrics, get_tracer
from slm_sms.data.schema import SMSEntity
from slm_sms.models.model_loader import get_device_and_dtype

logger = get_logger(__name__)

# Default model revision — override via SLM_MODEL_REVISION env-var or constructor arg.
# Pin to a known-good commit hash in production to prevent supply-chain risk.
_DEFAULT_REVISION: str | None = None


class SMSExtractorEngine:
    """Production inference engine for extracting structured entities from SMS messages."""

    SYSTEM_PROMPT = "Extract structured JSON from SMS. Output valid JSON only."

    def __init__(
        self,
        model_id_or_path: str = "Qwen/Qwen3.5-2B",
        adapter_path: str | None = None,
        device: str | None = None,
        torch_dtype: torch.dtype | None = None,
        max_new_tokens: int = 128,
        temperature: float = 0.01,
        top_p: float = 0.95,
        model_revision: str | None = _DEFAULT_REVISION,
    ):
        self.model_id_or_path = model_id_or_path
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.model_revision = model_revision

        detected_dev, detected_dtype, _ = get_device_and_dtype()
        self.device = torch.device(device) if device else detected_dev
        self.dtype = torch_dtype or detected_dtype

        self.tokenizer: PreTrainedTokenizer | None = None
        self.model: PreTrainedModel | PeftModel | None = None
        self._load_engine()

    def _load_engine(self) -> None:
        """Load tokenizer, base model, and optional LoRA adapter."""
        tracer = get_tracer(__name__)
        with tracer.start_as_current_span("slm.engine.load") as span:
            span.set_attribute("slm.model.id", self.model_id_or_path)
            span.set_attribute("slm.device", str(self.device))
            span.set_attribute("slm.has_adapter", bool(self.adapter_path))

            logger.info(
                "Loading inference engine",
                extra={
                    "base_model": self.model_id_or_path,
                    "revision": self.model_revision,
                    "adapter": self.adapter_path,
                },
            )

            try:
                tokenizer_source = self.adapter_path or self.model_id_or_path
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_source,
                    trust_remote_code=True,
                    use_fast=True,
                    revision=self.model_revision,
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                base_model = AutoModelForCausalLM.from_pretrained(
                    self.model_id_or_path,
                    torch_dtype=self.dtype,
                    trust_remote_code=True,
                    revision=self.model_revision,
                )

                if self.adapter_path and Path(self.adapter_path).exists():
                    logger.info("Loading LoRA adapter", extra={"adapter_path": self.adapter_path})
                    self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
                else:
                    self.model = base_model

                self.model.to(self.device)
                self.model.eval()
                span.set_attribute("slm.model.loaded", True)
                logger.info(
                    "Inference engine ready",
                    extra={"device": str(self.device), "dtype": str(self.dtype)},
                )
            except Exception as exc:
                span.set_attribute("slm.model.loaded", False)
                span.record_exception(exc)
                logger.warning(
                    "Model weights could not be loaded — operating in regex-fallback mode",
                    extra={"error": str(exc)},
                )
                self.model = None
                self.tokenizer = None

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def build_prompt_messages(self, sms_text: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": sms_text.strip()},
        ]

    # ------------------------------------------------------------------
    # Regex fallback
    # ------------------------------------------------------------------

    def _fallback_regex_extraction(self, text: str) -> SMSEntity:
        """Resilient rule-based fallback if model output is malformed or empty."""
        lower = text.lower()

        sms_type = "info"
        is_spam = False
        if any(
            w in lower
            for w in ["win real cash", "pre-approved", "click bit.ly", "guaranteed return"]
        ):
            sms_type = "spam"
            is_spam = True
        elif "otp" in lower or "verification code" in lower:
            sms_type = "otp"
        elif "debited" in lower or "spent" in lower or "paid" in lower or "wdl" in lower:
            sms_type = "debit"
        elif "credited" in lower or "salary" in lower or "cashback" in lower:
            sms_type = "credit"
        elif "refund" in lower:
            sms_type = "refund"
        elif "bill" in lower or "due date" in lower:
            sms_type = "bill"

        amt_match = re.search(r"(?:rs\.?|inr|₹)\s*([\d,]+(?:\.\d{1,2})?)", text, re.IGNORECASE)
        amount = None
        if amt_match:
            try:
                amount = float(amt_match.group(1).replace(",", ""))
            except ValueError:
                pass

        acc_match = re.search(r"(?:\*{2,}|\.{2,}|xx|ending\s+)(\d{3,6})", text, re.IGNORECASE)
        account = acc_match.group(1) if acc_match else None

        otp_match = re.search(
            r"\b(\d{4,8})\b\s+is\s+(?:(?:your|the|a)\s+)?(?:secret\s+|one[-\s]time\s+)?(?:otp|password|code|verification)",
            text,
            re.IGNORECASE,
        )
        if not otp_match:
            otp_match = re.search(
                r"(?:otp|code|password)\s+(?:is|:)?\s*(\d{4,8})\b",
                text,
                re.IGNORECASE,
            )
        otp = otp_match.group(1) if otp_match else None

        return SMSEntity(
            type=sms_type,
            amount=amount,
            currency="INR" if amount else None,
            account=account,
            otp=otp,
            spam=is_spam,
        )

    # ------------------------------------------------------------------
    # Single inference
    # ------------------------------------------------------------------

    def extract(self, sms_text: str) -> dict[str, Any]:
        """Extract structured entity from a single SMS text.

        This is a **synchronous, blocking** call — callers in async contexts
        must offload it via ``run_in_executor_with_context``.

        Returns dict with keys: entity, latency_ms, fallback_used.
        """
        start_time = time.perf_counter()
        tracer = get_tracer(__name__)
        otel_metrics = get_metrics()
        model_id = getattr(self, "model_id_or_path", settings.model_id)

        with tracer.start_as_current_span("slm.engine.extract") as span:
            # Never store raw SMS content or OTPs on telemetry attributes (PII-safe)
            span.set_attribute("gen_ai.request.model", model_id)
            span.set_attribute("slm.text.length", len(sms_text))

            if self.model is None or self.tokenizer is None:
                with tracer.start_as_current_span("slm.fallback.regex") as fb_span:
                    fb_span.set_attribute("slm.fallback.reason", "model_unloaded")
                    entity = self._fallback_regex_extraction(sms_text)

                latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                span.set_attribute("slm.fallback.used", True)
                span.set_attribute("slm.entity.type", entity.type)

                otel_metrics.fallback_total.add(1, attributes={"reason": "model_unloaded"})
                otel_metrics.requests_total.add(
                    1,
                    attributes={
                        "operation": "single",
                        "mode": "fallback",
                        "status": "ok",
                        "entity_type": entity.type,
                    },
                )
                otel_metrics.inference_duration_ms.record(
                    latency_ms, attributes={"operation": "single", "mode": "fallback"}
                )
                return {
                    "entity": entity.to_dict(),
                    "latency_ms": latency_ms,
                    "fallback_used": True,
                }

            messages = self.build_prompt_messages(sms_text)
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
            input_len = int(inputs.input_ids.shape[1])

            with tracer.start_as_current_span("slm.llm.generate") as gen_span:
                gen_span.set_attribute("gen_ai.request.model", model_id)
                gen_span.set_attribute("gen_ai.request.max_tokens", self.max_new_tokens)
                gen_span.set_attribute("gen_ai.usage.input_tokens", input_len)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature if self.temperature > 0 else None,
                        do_sample=self.temperature > 0,
                        top_p=self.top_p if self.temperature > 0 else None,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                gen_tokens = outputs[0][input_len:]
                output_len = int(gen_tokens.shape[0])
                gen_span.set_attribute("gen_ai.usage.output_tokens", output_len)

            otel_metrics.input_tokens_total.add(input_len, attributes={"operation": "single"})
            otel_metrics.output_tokens_total.add(output_len, attributes={"operation": "single"})

            raw_response = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

            fallback_used = False
            with tracer.start_as_current_span("slm.entity.parse") as parse_span:
                try:
                    entity = SMSEntity.from_raw_text(raw_response)
                    parse_span.set_attribute("slm.parse.valid", True)
                except Exception as exc:
                    parse_span.set_attribute("slm.parse.valid", False)
                    parse_span.record_exception(exc)
                    parse_span.set_status(Status(StatusCode.ERROR, "JSON parse error"))
                    logger.warning(
                        "Failed to parse LLM JSON — using regex fallback",
                        extra={"error": str(exc)},
                    )
                    with tracer.start_as_current_span("slm.fallback.regex") as fb_span:
                        fb_span.set_attribute("slm.fallback.reason", "parse_error")
                        entity = self._fallback_regex_extraction(sms_text)
                    fallback_used = True
                    otel_metrics.fallback_total.add(1, attributes={"reason": "parse_error"})

            mode = "fallback" if fallback_used else "model"
            span.set_attribute("slm.fallback.used", fallback_used)
            span.set_attribute("slm.entity.type", entity.type)

            otel_metrics.requests_total.add(
                1,
                attributes={
                    "operation": "single",
                    "mode": mode,
                    "status": "ok",
                    "entity_type": entity.type,
                },
            )
            otel_metrics.inference_duration_ms.record(
                latency_ms, attributes={"operation": "single", "mode": mode}
            )

            return {
                "entity": entity.to_dict(),
                "latency_ms": latency_ms,
                "fallback_used": fallback_used,
            }

    # ------------------------------------------------------------------
    # Batch inference — native GPU batching
    # ------------------------------------------------------------------

    def extract_batch(self, sms_texts: list[str]) -> list[dict[str, Any]]:
        """Extract entities from multiple SMS texts using native GPU batching.

        When the model is unavailable, falls back to per-item regex extraction.
        This is a **synchronous, blocking** call — offload via ``run_in_executor_with_context``.
        """
        if not sms_texts:
            return []

        start_time = time.perf_counter()
        tracer = get_tracer(__name__)
        otel_metrics = get_metrics()
        model_id = getattr(self, "model_id_or_path", settings.model_id)
        batch_len = len(sms_texts)

        with tracer.start_as_current_span("slm.engine.extract_batch") as span:
            span.set_attribute("gen_ai.request.model", model_id)
            span.set_attribute("slm.batch.size", batch_len)

            if self.model is None or self.tokenizer is None:
                results: list[dict[str, Any]] = []
                with tracer.start_as_current_span("slm.fallback.regex") as fb_span:
                    fb_span.set_attribute("slm.fallback.reason", "model_unloaded")
                    for text in sms_texts:
                        item_start = time.perf_counter()
                        entity = self._fallback_regex_extraction(text)
                        item_ms = round((time.perf_counter() - item_start) * 1000, 2)
                        otel_metrics.requests_total.add(
                            1,
                            attributes={
                                "operation": "batch",
                                "mode": "fallback",
                                "status": "ok",
                                "entity_type": entity.type,
                            },
                        )
                        results.append(
                            {
                                "entity": entity.to_dict(),
                                "latency_ms": item_ms,
                                "fallback_used": True,
                            }
                        )

                total_ms = round((time.perf_counter() - start_time) * 1000, 2)
                span.set_attribute("slm.fallback.used", True)
                otel_metrics.fallback_total.add(batch_len, attributes={"reason": "model_unloaded"})
                otel_metrics.batch_size.record(batch_len, attributes={"mode": "fallback"})
                otel_metrics.inference_duration_ms.record(
                    total_ms, attributes={"operation": "batch", "mode": "fallback"}
                )
                return results

            prompts = [
                self.tokenizer.apply_chat_template(
                    self.build_prompt_messages(t),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for t in sms_texts
            ]

            original_padding_side = self.tokenizer.padding_side
            self.tokenizer.padding_side = "left"
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)
            self.tokenizer.padding_side = original_padding_side

            input_lens = inputs.attention_mask.sum(dim=1).tolist()
            total_input_tokens = int(sum(input_lens))

            with tracer.start_as_current_span("slm.llm.generate_batch") as gen_span:
                gen_span.set_attribute("gen_ai.request.model", model_id)
                gen_span.set_attribute("gen_ai.request.max_tokens", self.max_new_tokens)
                gen_span.set_attribute("gen_ai.usage.input_tokens", total_input_tokens)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature if self.temperature > 0 else None,
                        do_sample=self.temperature > 0,
                        top_p=self.top_p if self.temperature > 0 else None,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                total_output_tokens = int(
                    sum(max(0, len(seq) - inp_len) for seq, inp_len in zip(outputs, input_lens))
                )
                gen_span.set_attribute("gen_ai.usage.output_tokens", total_output_tokens)

            otel_metrics.input_tokens_total.add(
                total_input_tokens, attributes={"operation": "batch"}
            )
            otel_metrics.output_tokens_total.add(
                total_output_tokens, attributes={"operation": "batch"}
            )

            total_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            per_item_latency = round(total_latency_ms / batch_len, 2)

            results = []
            fallback_count = 0
            for i, (output_seq, inp_len, orig_text) in enumerate(
                zip(outputs, input_lens, sms_texts)
            ):
                gen_tokens = output_seq[inp_len:]
                raw_response = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

                fallback_used = False
                try:
                    entity = SMSEntity.from_raw_text(raw_response)
                except Exception as exc:
                    logger.warning(
                        "Batch item parse failed — using regex fallback",
                        extra={"index": i, "error": str(exc)},
                    )
                    entity = self._fallback_regex_extraction(orig_text)
                    fallback_used = True
                    fallback_count += 1

                item_mode = "fallback" if fallback_used else "model"
                otel_metrics.requests_total.add(
                    1,
                    attributes={
                        "operation": "batch",
                        "mode": item_mode,
                        "status": "ok",
                        "entity_type": entity.type,
                    },
                )
                results.append(
                    {
                        "entity": entity.to_dict(),
                        "latency_ms": per_item_latency,
                        "fallback_used": fallback_used,
                    }
                )

            if fallback_count > 0:
                otel_metrics.fallback_total.add(
                    fallback_count, attributes={"reason": "parse_error"}
                )

            span.set_attribute("slm.fallback.count", fallback_count)
            otel_metrics.batch_size.record(batch_len, attributes={"mode": "model"})
            otel_metrics.inference_duration_ms.record(
                total_latency_ms, attributes={"operation": "batch", "mode": "model"}
            )
            return results
