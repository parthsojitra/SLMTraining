"""Locust load test for the SLM SMS API.

Simulates realistic production traffic patterns against a running server.

Usage
-----
1. Start the API server (with lazy load so it starts fast):
       SLM_LAZY_LOAD=true uv run python scripts/serve.py

   For realistic load testing against a model-loaded server:
       uv run python scripts/serve.py --model_id Qwen/Qwen3.5-2B

2. Run Locust:
       # Interactive web UI (http://localhost:8089)
       uv run locust -f tests/load/locustfile.py --host http://localhost:8000

       # Headless CI mode (100 users, 10 spawn/sec, 60 s duration)
       uv run locust -f tests/load/locustfile.py --host http://localhost:8000 \\
           --users 100 --spawn-rate 10 --run-time 60s --headless \\
           --csv results/load_test

Requirements
------------
Install the locust extra (not in default dev deps):
    uv add --dev locust

Scenario breakdown
------------------
- 70%  Single extract  — peak latency should be < 500 ms on GPU, < 5 s on CPU
- 20%  Batch extract   — 2–10 SMS per request
- 10%  Health checks   — simulates k8s probes
"""

from __future__ import annotations

import random

from locust import HttpUser, TaskSet, between, task

# ---------------------------------------------------------------------------
# Realistic Indian SMS samples for load testing
# ---------------------------------------------------------------------------

DEBIT_SMS = [
    "[VM-HDFCBK] Rs.1,450.00 debited from A/c **4821 on 24-11-26 to swiggy@axisbank (UPI Ref: 432819284129). Avl Bal: Rs.34,210.50.",
    "[AD-ICICIB] INR 2,499.00 spent on Zomato on 24-Nov-26. Card ending 7712. Avl limit: Rs.47,500.",
    "[BZ-SBIINB] Rs 850.00 debited from SBI A/c XX1042 to Uber India via UPI on 24/11/26. Bal: Rs.12,340.00.",
    "[AX-AXISBK] Transaction of Rs.3,200.00 done on Axis Bank Card **5531 at Flipkart on 23-Nov-26.",
    "[VK-KOTAKB] Rs.599.00 paid to Netflix India via CRED. Kotak Bank A/c **9821. Bal: Rs.8,750.50.",
]

CREDIT_SMS = [
    "[VM-HDFCBK] Rs.45,000.00 credited to A/c **4821 on 24-Nov-26. Avl Bal: Rs.79,210.50.",
    "[AD-ICICIB] Salary credit of Rs.75,000.00 to your ICICI A/c **7712. 24-11-26.",
    "[BZ-SBIINB] Cashback of Rs.250.00 credited to SBI A/c XX1042 from Google Pay UPI Ref 432900012345.",
]

OTP_SMS = [
    "489201 is your secret OTP for transaction of Rs.2,499.00 on your Axis Bank Card **7712 at ZOMATO. Valid for 10 mins.",
    "Your OTP for HDFC Bank NetBanking login is 834729. Do NOT share with anyone. Valid for 5 minutes.",
    "Use OTP: 291847 to complete your IRCTC booking. Valid for 15 minutes. - IRCTC",
]

SPAM_SMS = [
    "Congratulations! Pre-approved personal loan of Rs.5,00,000 without income proof. Click bit.ly/loan123",
    "Win real cash Rs.1,00,000 playing online rummy! Download now: bit.ly/rummy-win",
]

ALL_SMS = DEBIT_SMS + CREDIT_SMS + OTP_SMS + SPAM_SMS


# ---------------------------------------------------------------------------
# Task sets
# ---------------------------------------------------------------------------


class ExtractTaskSet(TaskSet):
    """Mixed load: single extract, batch extract, and health probes."""

    @task(7)
    def single_extract(self):
        sms = random.choice(ALL_SMS)
        with self.client.post(
            "/api/v1/extract",
            json={"text": sms},
            name="/api/v1/extract",
            catch_response=True,
        ) as res:
            if res.status_code == 200:
                body = res.json()
                if "entity" not in body:
                    res.failure("Response missing 'entity' field")
                else:
                    res.success()
            elif res.status_code == 503:
                res.failure("Engine not ready (503)")
            else:
                res.failure(f"Unexpected status: {res.status_code}")

    @task(2)
    def batch_extract(self):
        batch_size = random.randint(2, 10)
        texts = random.choices(ALL_SMS, k=batch_size)
        with self.client.post(
            "/api/v1/extract/batch",
            json={"texts": texts},
            name="/api/v1/extract/batch",
            catch_response=True,
        ) as res:
            if res.status_code == 200:
                body = res.json()
                if body.get("count") != batch_size:
                    res.failure(f"Expected {batch_size} results, got {body.get('count')}")
                else:
                    res.success()
            elif res.status_code == 503:
                res.failure("Engine not ready (503)")
            else:
                res.failure(f"Unexpected status: {res.status_code}")

    @task(1)
    def health_check(self):
        self.client.get("/health/live", name="/health/live")
        self.client.get("/health/ready", name="/health/ready")


# ---------------------------------------------------------------------------
# User class
# ---------------------------------------------------------------------------


class SMSApiUser(HttpUser):
    """Simulated API consumer with realistic think-time between requests."""

    tasks = [ExtractTaskSet]
    wait_time = between(0.5, 2.0)  # 0.5–2 seconds between tasks

    def on_start(self):
        """Called once per simulated user at spawn."""
        # Warm-up: check that server is reachable
        self.client.get("/health/live")
