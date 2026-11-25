"""Tests for the Indian SMS synthetic dataset generator."""

import pytest

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
    # Disable the class-balance guard for small test datasets
    splits = gen.generate_dataset(
        n_samples=100, val_split=0.15, test_split=0.05, min_class_fraction=0.0
    )

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


def test_generator_class_balance_guard():
    """generate_dataset should raise ValueError when a class is severely underrepresented."""
    gen = IndianSMSGenerator(seed=999)
    # At 100 samples, 'bill' and 'spam' are ~8 % each — well above 15 % floor
    with pytest.raises(ValueError, match="below the"):
        gen.generate_dataset(n_samples=100, min_class_fraction=0.15)


def test_generator_indian_number_format():
    """_format_indian must use lakh/crore grouping, not Western grouping."""
    gen = IndianSMSGenerator(seed=0)
    cases = [
        (500.0, "500.00"),
        (1450.0, "1,450.00"),
        (12450.0, "12,450.00"),
        (145000.0, "1,45,000.00"),
        (1450000.0, "14,50,000.00"),
        (10000000.0, "1,00,00,000.00"),
    ]
    for amt, expected in cases:
        assert gen._format_indian(amt) == expected, f"Failed for {amt}"


def test_generator_rng_isolation():
    """Two generators with the same seed must produce identical output independently."""
    gen_a = IndianSMSGenerator(seed=42)
    gen_b = IndianSMSGenerator(seed=42)
    text_a, entity_a = gen_a.generate_debit_sample()
    text_b, entity_b = gen_b.generate_debit_sample()
    assert text_a == text_b
    assert entity_a.amount == entity_b.amount


def test_generator_different_seeds_differ():
    """Generators with different seeds must produce different output."""
    gen_a = IndianSMSGenerator(seed=1)
    gen_b = IndianSMSGenerator(seed=2)
    texts_a = [gen_a.generate_debit_sample()[0] for _ in range(10)]
    texts_b = [gen_b.generate_debit_sample()[0] for _ in range(10)]
    assert texts_a != texts_b
