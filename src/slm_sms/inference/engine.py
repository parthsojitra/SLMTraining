"""Inference engine for high-reliability SMS entity extraction."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from slm_sms.data.schema import SMSEntity
from slm_sms.models.model_loader import get_device_and_dtype

logger = logging.getLogger("slm_sms.inference")


class SMSExtractorEngine:
    """Production inference engine for extracting structured entities from SMS messages."""

    SYSTEM_PROMPT = "Extract structured JSON from SMS. Output valid JSON only."

    def __init__(
        self,
        model_id_or_path: str = "Qwen/Qwen3.5-2B",
        adapter_path: str | None = None,
        device: str | None = None,
        torch_dtype: torch.dtype | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.01,
        top_p: float = 0.95,
    ):
        self.model_id_or_path = model_id_or_path
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        # Determine compute target
        detected_dev, detected_dtype, _ = get_device_and_dtype()
        self.device = torch.device(device) if device else detected_dev
        self.dtype = torch_dtype or detected_dtype

        self.tokenizer: PreTrainedTokenizer | None = None
        self.model: PreTrainedModel | PeftModel | None = None
        self._load_engine()

    def _load_engine(self) -> None:
        """Load tokenizer, base model, and optional LoRA adapter."""
        logger.info(
            f"Loading inference engine: base='{self.model_id_or_path}', adapter='{self.adapter_path}'"
        )

        try:
            tokenizer_source = self.adapter_path or self.model_id_or_path
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_source,
                trust_remote_code=True,
                use_fast=True,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_id_or_path,
                torch_dtype=self.dtype,
                trust_remote_code=True,
            )

            if self.adapter_path and Path(self.adapter_path).exists():
                logger.info(f"Loading LoRA adapter from {self.adapter_path}")
                self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
            else:
                self.model = base_model

            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Inference engine ready on {self.device} ({self.dtype})")
        except Exception as e:
            logger.warning(
                f"Model weights could not be loaded ({e}). Operating in offline resilient fallback mode."
            )
            self.model = None
            self.tokenizer = None

    def build_prompt_messages(self, sms_text: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": sms_text.strip()},
        ]

    def _fallback_regex_extraction(self, text: str) -> SMSEntity:
        """Resilient rule-based fallback if model output is malformed or empty."""
        lower = text.lower()

        # Classify basic type
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

        # Extract amount
        amt_match = re.search(r"(?:rs\.?|inr|₹)\s*([\d,]+(?:\.\d{1,2})?)", text, re.IGNORECASE)
        amount = None
        if amt_match:
            try:
                amount = float(amt_match.group(1).replace(",", ""))
            except ValueError:
                pass

        # Extract account (e.g. **4821, XX1042, ...3391, ending 4821)
        acc_match = re.search(r"(?:\*{2,}|\.{2,}|xx|ending\s+)(\d{3,6})", text, re.IGNORECASE)
        account = acc_match.group(1) if acc_match else None

        # Extract OTP (handles "489201 is your secret OTP", "OTP is 489201", etc.)
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

    def extract(self, sms_text: str) -> dict[str, Any]:
        """Extract structured entity from SMS text.

        Returns:
            Dict containing parsed entity, raw LLM output, and execution latency.
        """
        start_time = time.perf_counter()

        # If model is not loaded (offline mode / fallback), return fallback immediately
        if self.model is None or self.tokenizer is None:
            entity = self._fallback_regex_extraction(sms_text)
            latency_ms = (time.perf_counter() - start_time) * 1000
            return {
                "entity": entity.to_dict(),
                "raw_output": entity.to_compact_json(),
                "latency_ms": round(latency_ms, 2),
                "fallback_used": True,
            }
        messages = self.build_prompt_messages(sms_text)

        # Apply chat template
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        input_len = inputs.input_ids.shape[1]

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
        raw_response = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Parse entity
        fallback_used = False
        try:
            entity = SMSEntity.from_raw_text(raw_response)
        except Exception as e:
            logger.warning(
                f"Failed to parse LLM JSON: '{raw_response}'. Using fallback. Error: {e}"
            )
            entity = self._fallback_regex_extraction(sms_text)
            fallback_used = True

        return {
            "entity": entity.to_dict(),
            "raw_output": raw_response,
            "latency_ms": round(latency_ms, 2),
            "fallback_used": fallback_used,
        }

    def extract_batch(self, sms_texts: list[str]) -> list[dict[str, Any]]:
        """Extract entities from multiple SMS texts."""
        return [self.extract(text) for text in sms_texts]
