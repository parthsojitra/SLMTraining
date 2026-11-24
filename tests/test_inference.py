"""Tests for inference engine fallback parser, metrics calculation, and FastAPI endpoints."""

from fastapi.testclient import TestClient

from slm_sms.inference.engine import SMSExtractorEngine
from slm_sms.inference.server import app
from slm_sms.training.metrics import compute_extraction_metrics


def test_fallback_regex_extractor():
    engine_stub = SMSExtractorEngine.__new__(SMSExtractorEngine)

    # Test debit SMS
    debit_sms = "[VM-HDFCBK] Rs.1,450.00 debited from A/c **4821 on 23-11-26 to swiggy@axisbank. Avl Bal: Rs.34,210.50."
    res = engine_stub._fallback_regex_extraction(debit_sms)
    assert res.type == "debit"
    assert res.amount == 1450.0
    assert res.account == "4821"
    assert res.spam is False

    # Test OTP SMS
    otp_sms = (
        "489201 is your secret OTP for transaction of Rs.2,499.00 on your Axis Bank Card **7712."
    )
    res_otp = engine_stub._fallback_regex_extraction(otp_sms)
    assert res_otp.type == "otp"
    assert res_otp.otp == "489201"
    assert res_otp.amount == 2499.0
    assert res_otp.account == "7712"

    # Test Spam SMS
    spam_sms = "Congratulations! Pre-approved personal loan of Rs.5,00,000 without income proof. Click bit.ly/loan123"
    res_spam = engine_stub._fallback_regex_extraction(spam_sms)
    assert res_spam.type == "spam"
    assert res_spam.amount == 500000.0
    assert res_spam.spam is True


def test_metrics_computation():
    predictions = [
        '{"type":"debit","amount":1450.0,"currency":"INR","merchant":"swiggy@axisbank","account":"4821","spam":false}',
        '{"type":"otp","amount":2499.0,"currency":"INR","merchant":"ZOMATO","account":"7712","otp":"489201","spam":false}',
    ]
    ground_truths = [
        {
            "type": "debit",
            "amount": 1450.0,
            "currency": "INR",
            "merchant": "swiggy@axisbank",
            "account": "4821",
            "spam": False,
        },
        {
            "type": "otp",
            "amount": 2499.0,
            "currency": "INR",
            "merchant": "ZOMATO",
            "account": "7712",
            "otp": "489201",
            "spam": False,
        },
    ]
    report = compute_extraction_metrics(predictions, ground_truths)
    assert report.total_samples == 2
    assert report.valid_json_count == 2
    assert report.json_validity_rate == 1.0
    assert report.type_accuracy == 1.0
    assert report.amount_accuracy == 1.0
    assert report.merchant_accuracy == 1.0
    assert report.account_accuracy == 1.0


def test_fastapi_server_endpoints():
    client = TestClient(app)

    # Test root endpoint
    res_root = client.get("/")
    assert res_root.status_code == 200
    data = res_root.json()
    assert data["status"] == "online"

    # Test health endpoint
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "healthy"
