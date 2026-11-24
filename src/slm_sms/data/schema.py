"""Pydantic schemas for Indian SMS structured entity extraction."""

from __future__ import annotations

import json
import re
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class TransactionType(StrEnum):
    DEBIT = "debit"
    CREDIT = "credit"
    OTP = "otp"
    BILL = "bill"
    REFUND = "refund"
    PROMO = "promo"
    INFO = "info"
    SPAM = "spam"


class SMSEntity(BaseModel):
    """Structured entity extracted from an SMS message."""

    type: Literal["debit", "credit", "otp", "bill", "refund", "promo", "info", "spam"] = Field(
        description="Classification of the SMS message"
    )
    amount: float | None = Field(
        default=None,
        description="Transaction or bill amount in decimal format",
    )
    currency: str | None = Field(
        default="INR",
        description="Currency code (e.g., INR, USD)",
    )
    merchant: str | None = Field(
        default=None,
        description="Merchant, sender, vendor, or UPI handle (e.g., swiggy@axisbank, Amazon)",
    )
    account: str | None = Field(
        default=None,
        description="Last 4 digits or masked identifier of the bank account or card (e.g., 4821)",
    )
    balance: float | None = Field(
        default=None,
        description="Available or net account balance after transaction",
    )
    reference_id: str | None = Field(
        default=None,
        description="Transaction reference ID, UTR number, or order ID",
    )
    date_time: str | None = Field(
        default=None,
        description="Extracted transaction timestamp or date string",
    )
    otp: str | None = Field(
        default=None,
        description="One-time password code (if message contains an OTP)",
    )
    category: str | None = Field(
        default=None,
        description="Broad category (food_dining, shopping, utilities, travel, banking, spam)",
    )
    spam: bool = Field(
        default=False,
        description="Whether the SMS is unsolicited promotional/fraud/spam content",
    )

    @field_validator("account", mode="before")
    @classmethod
    def clean_account(cls, v: Any) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        # Extract digits if surrounded by asterisks/dots like **4821, ...4821, XX4821
        match = re.search(r"(\d{3,6})$", s)
        if match:
            return match.group(1)
        return s if s else None

    @field_validator("amount", "balance", mode="before")
    @classmethod
    def clean_amount(cls, v: Any) -> float | None:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).replace(",", "").replace("₹", "").replace("Rs.", "").replace("Rs", "").strip()
        try:
            return float(s)
        except ValueError:
            return None

    def to_compact_json(self) -> str:
        """Dump JSON omitting None fields for clean training labels."""
        data = self.model_dump(exclude_none=True)
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_raw_text(cls, text: str) -> SMSEntity:
        """Parse raw model output containing JSON markdown or raw JSON."""
        # Find JSON block enclosed in ```json ... ``` or first { ... }
        clean_text = text.strip()
        markdown_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean_text, re.DOTALL)
        if markdown_match:
            clean_text = markdown_match.group(1)
        else:
            json_match = re.search(r"\{.*?\}", clean_text, re.DOTALL)
            if json_match:
                clean_text = json_match.group(0)

        data = json.loads(clean_text)
        return cls(**data)


class SMSChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class SMSChatSample(BaseModel):
    """Hugging Face Chat format sample."""

    messages: list[SMSChatMessage]
