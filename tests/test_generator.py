"""Tests for the Indian SMS synthetic dataset generator."""

from slm_sms.data.generator import IndianSMSGenerator
from slm_sms.data.schema import SMSEntity


def test_generator_single_samples():
    gen = IndianSMSGenerator(seed=123)

    # Test debit
    text, entity = gen.generate_debit_sample()
    assert isinstance(text, str)
    assert len(text) > 10
    assert entity.type == "debit"
    assert entity.amount is not None
    assert entity.amount > 0

    # Test credit
    text, entity = gen.generate_credit_sample()
    assert entity.type in ("credit", "refund")
    assert entity.amount is not None

    # Test OTP
    text, entity = gen.generate_otp_sample()
    assert entity.type == "otp"
    assert entity.otp is not None
    assert len(entity.otp) >= 4

    # Test bill
    text, entity = gen.generate_bill_sample()
    assert entity.type == "bill"
    assert entity.amount is not None

    # Test spam
    text, entity = gen.generate_spam_sample()
    assert entity.spam is True


def test_generator_dataset_splits():
    gen = IndianSMSGenerator(seed=999)
    splits = gen.generate_dataset(n_samples=100, val_split=0.15, test_split=0.05)

    assert len(splits["train"]) == 80
    assert len(splits["validation"]) == 15
    assert len(splits["test"]) == 5

    # Check each sample follows HuggingFace chat format
    for item in splits["train"]:
        assert "messages" in item
        assert len(item["messages"]) == 3
        # Verify assistant output is valid SMSEntity JSON
        assistant_json = item["messages"][2]["content"]
        entity = SMSEntity.from_raw_text(assistant_json)
        assert isinstance(entity.type, str)
