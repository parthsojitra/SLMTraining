"""Unit tests for SMS data schemas and JSON serialization."""

from slm_sms.data.schema import SMSChatMessage, SMSChatSample, SMSEntity


def test_sms_entity_clean_amount():
    entity = SMSEntity(
        type="debit",
        amount="Rs. 1,450.50",
        merchant="swiggy@axisbank",
        account="**4821",
        spam=False,
    )
    assert entity.amount == 1450.50
    assert entity.account == "4821"
    assert entity.currency == "INR"
    assert entity.spam is False


def test_sms_entity_rupee_symbol():
    entity = SMSEntity(
        type="credit",
        amount="₹25,000.00",
        balance="₹1,12,000.50",
        account="XX9012",
        merchant="INFOSYS",
        spam=False,
    )
    assert entity.amount == 25000.00
    assert entity.balance == 112000.50
    assert entity.account == "9012"


def test_sms_entity_from_raw_text_json():
    raw_json = '{"type":"debit","amount":450.0,"currency":"INR","merchant":"uber.india@icici","account":"1122","spam":false}'
    entity = SMSEntity.from_raw_text(raw_json)
    assert entity.type == "debit"
    assert entity.amount == 450.0
    assert entity.merchant == "uber.india@icici"
    assert entity.account == "1122"


def test_sms_entity_from_raw_text_markdown_block():
    markdown_output = """Here is the extracted information:
```json
{
  "type": "otp",
  "amount": 2499.0,
  "currency": "INR",
  "merchant": "ZOMATO",
  "account": "7712",
  "otp": "489201",
  "spam": false
}
```
Hope that helps!"""
    entity = SMSEntity.from_raw_text(markdown_output)
    assert entity.type == "otp"
    assert entity.amount == 2499.0
    assert entity.otp == "489201"
    assert entity.merchant == "ZOMATO"


def test_chat_sample_structure():
    sample = SMSChatSample(
        messages=[
            SMSChatMessage(role="system", content="Extract structured JSON from SMS."),
            SMSChatMessage(role="user", content="Rs 100 debited from A/c 1234"),
            SMSChatMessage(role="assistant", content='{"type":"debit","amount":100.0}'),
        ]
    )
    dumped = sample.model_dump()
    assert len(dumped["messages"]) == 3
    assert dumped["messages"][0]["role"] == "system"
    assert dumped["messages"][1]["role"] == "user"
    assert dumped["messages"][2]["role"] == "assistant"
