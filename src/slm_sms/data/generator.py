"""Indian SMS synthetic data generator for fine-tuning SLM models.

Generates realistic, varied, high-fidelity Indian banking, UPI, OTP, bill, and spam SMS messages
paired with structured JSON extraction labels following the Hugging Face chat template.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any

from slm_sms.data.schema import SMSChatMessage, SMSChatSample, SMSEntity


class IndianSMSGenerator:
    """Configurable synthetic data generator for Indian SMS messages."""

    SYSTEM_PROMPT = "Extract structured JSON from SMS. Output valid JSON only."

    BANKS = [
        {"name": "HDFC Bank", "code": "HDFCBK", "headers": ["VM-HDFCBK", "AD-HDFCBK", "QP-HDFCBK"]},
        {
            "name": "ICICI Bank",
            "code": "ICICIB",
            "headers": ["AD-ICICIB", "VM-ICICIB", "BZ-ICICIB"],
        },
        {
            "name": "State Bank of India",
            "code": "SBIINB",
            "headers": ["BZ-SBIINB", "MD-SBIINB", "AD-SBISMS"],
        },
        {"name": "Axis Bank", "code": "AXISBK", "headers": ["AX-AXISBK", "AD-AXISBK", "VK-AXISBK"]},
        {
            "name": "Kotak Mahindra Bank",
            "code": "KOTAKB",
            "headers": ["VK-KOTAKB", "AD-KOTAKB", "QP-KOTAKB"],
        },
        {"name": "Punjab National Bank", "code": "PNBSMS", "headers": ["AD-PNBSMS", "BZ-PNBSMS"]},
        {"name": "Bank of Baroda", "code": "BOBTXN", "headers": ["AD-BOBTXN", "VM-BOBTXN"]},
        {"name": "IndusInd Bank", "code": "INDBNK", "headers": ["CP-INDBNK", "AD-INDBNK"]},
        {"name": "Yes Bank", "code": "YESBNK", "headers": ["JM-YESBNK", "AD-YESBNK"]},
        {"name": "Federal Bank", "code": "FEDBNK", "headers": ["MD-FEDBNK", "AD-FEDBNK"]},
    ]

    UPI_APPS = ["Google Pay", "PhonePe", "Paytm", "BHIM", "CRED", "Amazon Pay", "Navi UPI"]

    MERCHANTS = [
        # Food & Groceries
        {"name": "Swiggy", "vpa": "swiggy@axisbank", "category": "food_dining"},
        {"name": "Zomato", "vpa": "zomato@hdfcbank", "category": "food_dining"},
        {"name": "Blinkit", "vpa": "blinkit@icici", "category": "shopping"},
        {"name": "Zepto", "vpa": "zepto@axisbank", "category": "shopping"},
        {"name": "Instamart", "vpa": "swiggyinstamart@icici", "category": "shopping"},
        {"name": "BigBasket", "vpa": "bigbasket@hdfcbank", "category": "shopping"},
        {"name": "McDonald's", "vpa": "mcdonalds@kotak", "category": "food_dining"},
        {"name": "Domino's Pizza", "vpa": "dominos@icici", "category": "food_dining"},
        {"name": "Starbucks India", "vpa": "starbucks@okhdfcbank", "category": "food_dining"},
        # E-commerce & Retail
        {"name": "Amazon India", "vpa": "amazonpay@apl", "category": "shopping"},
        {"name": "Flipkart", "vpa": "flipkart@axisbank", "category": "shopping"},
        {"name": "Myntra", "vpa": "myntra@icici", "category": "shopping"},
        {"name": "Nykaa", "vpa": "nykaa@kotak", "category": "shopping"},
        {"name": "Ajio", "vpa": "ajio@hdfcbank", "category": "shopping"},
        {"name": "DMART", "vpa": "dmartretail@icici", "category": "shopping"},
        {"name": "Reliance Digital", "vpa": "reliancedigital@hdfcbank", "category": "shopping"},
        {"name": "Croma", "vpa": "croma@tatapay", "category": "shopping"},
        # Travel & Mobility
        {"name": "Uber India", "vpa": "uber.india@icici", "category": "travel"},
        {"name": "Ola Cabs", "vpa": "ola@axisbank", "category": "travel"},
        {"name": "IRCTC", "vpa": "irctc@sbi", "category": "travel"},
        {"name": "MakeMyTrip", "vpa": "makemytrip@hdfcbank", "category": "travel"},
        {"name": "IndiGo", "vpa": "indigo@icici", "category": "travel"},
        {"name": "Rapido", "vpa": "rapido@paytm", "category": "travel"},
        # Health & Pharmacy
        {"name": "Apollo Pharmacy", "vpa": "apollopharmacy@hdfcbank", "category": "health"},
        {"name": "Tata 1mg", "vpa": "tata1mg@icici", "category": "health"},
        {"name": "PharmEasy", "vpa": "pharmeasy@axisbank", "category": "health"},
        # Entertainment & Utilities
        {"name": "BookMyShow", "vpa": "bookmyshow@kotak", "category": "entertainment"},
        {"name": "Netflix India", "vpa": "netflix@icici", "category": "entertainment"},
        {"name": "Airtel Payments", "vpa": "airtelbill@airtel", "category": "utilities"},
        {"name": "Jio Infocomm", "vpa": "jio@jioaxis", "category": "utilities"},
        {"name": "Tata Power", "vpa": "tatapower@icici", "category": "utilities"},
        {"name": "BESCOM", "vpa": "bescom@sbi", "category": "utilities"},
        {"name": "Adani Electricity", "vpa": "adani.elec@axisbank", "category": "utilities"},
    ]

    CURRENCY_SYMBOLS = ["Rs.", "Rs ", "INR ", "₹", "Rs. "]

    def __init__(self, seed: int | None = 42):
        if seed is not None:
            random.seed(seed)

    def _rand_amount(self, min_amt: float = 20.0, max_amt: float = 85000.0) -> float:
        amt = round(random.uniform(min_amt, max_amt), 2)
        if amt > 1000 and random.random() < 0.4:
            amt = float(round(amt))
        return amt

    def _format_amt(self, amt: float, curr_sym: str) -> str:
        # Indian number formatting with commas (e.g. 1,450.00 or 1,25,000.00)
        s = f"{amt:,.2f}"
        return f"{curr_sym}{s}"

    def _rand_account(self) -> str:
        return f"{random.randint(1000, 9999)}"

    def _rand_ref(self) -> str:
        # UPI ref or UTR is usually 12 digits
        return f"{random.randint(400000000000, 499999999999)}"

    def _rand_date_time(self) -> tuple[str, str]:
        days_ago = random.randint(0, 45)
        dt = datetime.now() - timedelta(days=days_ago, minutes=random.randint(5, 720))
        date_formats = [
            dt.strftime("%d-%m-%y"),
            dt.strftime("%d-%m-%Y"),
            dt.strftime("%d-%b-%y"),
            dt.strftime("%d-%b-%Y"),
            dt.strftime("%d/%m/%y"),
            dt.strftime("%d/%m/%Y"),
        ]
        time_formats = [
            dt.strftime("%H:%M:%S"),
            dt.strftime("%H:%M"),
            dt.strftime("%I:%M %p"),
        ]
        return random.choice(date_formats), random.choice(time_formats)

    def _rand_otp(self) -> str:
        return f"{random.randint(100000, 999999)}"

    # ---------------- Generator Methods ---------------- #

    def generate_debit_sample(self) -> tuple[str, SMSEntity]:
        bank = random.choice(self.BANKS)
        header = random.choice(bank["headers"])
        merchant = random.choice(self.MERCHANTS)
        account = self._rand_account()
        amount = self._rand_amount(45.0, 35000.0)
        curr_sym = random.choice(self.CURRENCY_SYMBOLS)
        formatted_amount = self._format_amt(amount, curr_sym)
        bal_amount = round(random.uniform(500.0, 150000.0), 2)
        formatted_bal = self._format_amt(bal_amount, curr_sym)
        date_str, time_str = self._rand_date_time()
        ref_id = self._rand_ref()
        use_vpa = random.random() < 0.65
        merchant_str = merchant["vpa"] if use_vpa else merchant["name"]

        # Templates
        templates = [
            # Standard bank SMS
            f"[{header}] {formatted_amount} debited from A/c **{account} on {date_str} to {merchant_str} (UPI Ref: {ref_id}). Avl Bal: {formatted_bal}.",
            f"Dear Customer, {formatted_amount} has been debited from your {bank['name']} A/c ending {account} towards {merchant_str} on {date_str} {time_str}. UPI Ref no {ref_id}. Avl Bal: {formatted_bal}.",
            f"Alert: {formatted_amount} spent on your {bank['name']} Card ending XX{account} at {merchant_str} on {date_str} at {time_str}. Avl Limit: {formatted_bal}.",
            f"A/c *{account} debited for {formatted_amount} on {date_str} {time_str} via UPI to {merchant_str}. UTR: {ref_id}. Bal: {formatted_bal} - {bank['name']}",
            f"Paid {formatted_amount} to {merchant_str} via {random.choice(self.UPI_APPS)} UPI Ref {ref_id} from {bank['name']} A/c ...{account}. Avl Bal: {formatted_bal}",
            f"[{header}] A/c {account} debited by {formatted_amount} on {date_str}. Info: UPI/{ref_id}/{merchant_str}. Total Bal: {formatted_bal}.",
            f"Txn Alert: {formatted_amount} debited from your A/c {account} on {date_str} to VPA {merchant_str}. Ref: {ref_id}. Check balance on {bank['name']} App.",
        ]

        text = random.choice(templates)
        entity = SMSEntity(
            type="debit",
            amount=amount,
            currency="INR",
            merchant=merchant_str,
            account=account,
            balance=bal_amount,
            reference_id=ref_id,
            date_time=f"{date_str} {time_str}".strip(),
            category=merchant["category"],
            spam=False,
        )
        return text, entity

    def generate_credit_sample(self) -> tuple[str, SMSEntity]:
        bank = random.choice(self.BANKS)
        header = random.choice(bank["headers"])
        merchant = random.choice(self.MERCHANTS)
        account = self._rand_account()
        amount = self._rand_amount(100.0, 120000.0)
        curr_sym = random.choice(self.CURRENCY_SYMBOLS)
        formatted_amount = self._format_amt(amount, curr_sym)
        bal_amount = round(random.uniform(5000.0, 250000.0), 2)
        formatted_bal = self._format_amt(bal_amount, curr_sym)
        date_str, time_str = self._rand_date_time()
        ref_id = self._rand_ref()

        credit_types = ["salary", "refund", "upi_credit", "cashback"]
        c_type = random.choice(credit_types)

        if c_type == "salary":
            company = random.choice(
                [
                    "INFOSYS LTD",
                    "TCS CORP",
                    "WIPRO TECH",
                    "GOOGLE INDIA",
                    "ACCENTURE",
                    "TECH MAHINDRA",
                ]
            )
            text = f"[{header}] A/c *{account} credited by {formatted_amount} on {date_str} by Salary transfer from {company}. Avl Bal: {formatted_bal}."
            entity = SMSEntity(
                type="credit",
                amount=amount,
                currency="INR",
                merchant=company,
                account=account,
                balance=bal_amount,
                reference_id=ref_id,
                date_time=date_str,
                category="banking",
                spam=False,
            )
        elif c_type == "refund":
            text = f"Refund of {formatted_amount} processed for your order #{ref_id[:8]} from {merchant['name']} to your Card ending {account}. Expected within 2-4 days."
            entity = SMSEntity(
                type="refund",
                amount=amount,
                currency="INR",
                merchant=merchant["name"],
                account=account,
                balance=None,
                reference_id=ref_id[:8],
                date_time=date_str,
                category=merchant["category"],
                spam=False,
            )
        elif c_type == "cashback":
            text = f"Cashback Alert! {formatted_amount} credited to your A/c {account} from {random.choice(self.UPI_APPS)} for UPI txn {ref_id}. Net Bal: {formatted_bal}."
            entity = SMSEntity(
                type="credit",
                amount=amount,
                currency="INR",
                merchant=random.choice(self.UPI_APPS),
                account=account,
                balance=bal_amount,
                reference_id=ref_id,
                date_time=date_str,
                category="banking",
                spam=False,
            )
        else:
            text = f"Dear Customer, {formatted_amount} credited to your {bank['name']} A/c XX{account} on {date_str} through UPI from {merchant['vpa']} (Ref {ref_id}). Avl Bal: {formatted_bal}."
            entity = SMSEntity(
                type="credit",
                amount=amount,
                currency="INR",
                merchant=merchant["vpa"],
                account=account,
                balance=bal_amount,
                reference_id=ref_id,
                date_time=date_str,
                category=merchant["category"],
                spam=False,
            )

        return text, entity

    def generate_otp_sample(self) -> tuple[str, SMSEntity]:
        bank = random.choice(self.BANKS)
        header = random.choice(bank["headers"])
        merchant = random.choice(self.MERCHANTS)
        account = self._rand_account()
        otp = self._rand_otp()
        amount = self._rand_amount(150.0, 45000.0)
        curr_sym = random.choice(self.CURRENCY_SYMBOLS)
        formatted_amount = self._format_amt(amount, curr_sym)
        date_str, time_str = self._rand_date_time()

        otp_kinds = ["txn", "login", "aadhaar"]
        kind = random.choice(otp_kinds)

        if kind == "txn":
            templates = [
                f"{otp} is your secret OTP for transaction of {formatted_amount} on your {bank['name']} Card **{account} at {merchant['name']}. Valid for 5 mins. Do not share OTP with anyone.",
                f"[{header}] {otp} is your OTP for online purchase of {formatted_amount} at {merchant['name']} using Card ending {account}. Valid for 3 mins. Bank NEVER asks for OTP.",
                f"Use OTP {otp} to approve payment of {formatted_amount} to {merchant['name']}. Generated at {time_str}. Do not disclose to anyone. - {bank['name']}",
            ]
            text = random.choice(templates)
            entity = SMSEntity(
                type="otp",
                amount=amount,
                currency="INR",
                merchant=merchant["name"],
                account=account,
                otp=otp,
                date_time=time_str,
                category="banking",
                spam=False,
            )
        elif kind == "login":
            templates = [
                f"[{header}] {otp} is the OTP to login to your {bank['name']} NetBanking account. Valid for 3 mins. Do not share this with anyone including bank staff.",
                f"Dear Customer, {otp} is your OTP for registering mobile banking on A/c ending {account}. Valid for 5 minutes. - {bank['name']}",
            ]
            text = random.choice(templates)
            entity = SMSEntity(
                type="otp",
                account=account,
                otp=otp,
                category="banking",
                spam=False,
            )
        else:
            text = f"{otp} is your One Time Password (OTP) for Aadhaar verification. Valid for 10 minutes. Please do not share this with anyone. - UIDAI"
            entity = SMSEntity(
                type="otp",
                merchant="UIDAI",
                otp=otp,
                category="banking",
                spam=False,
            )

        return text, entity

    def generate_bill_sample(self) -> tuple[str, SMSEntity]:
        bank = random.choice(self.BANKS)
        header = random.choice(bank["headers"])
        account = self._rand_account()
        amount = self._rand_amount(800.0, 75000.0)
        min_amount = round(amount * random.uniform(0.05, 0.15), 2)
        curr_sym = random.choice(self.CURRENCY_SYMBOLS)
        formatted_amount = self._format_amt(amount, curr_sym)
        formatted_min = self._format_amt(min_amount, curr_sym)
        due_date, _ = self._rand_date_time()

        bill_types = ["credit_card", "electricity", "telecom"]
        b_type = random.choice(bill_types)

        if b_type == "credit_card":
            templates = [
                f"[{header}] Dear Customer, your Credit Card statement for Card ending {account} generated. Total Due: {formatted_amount}, Min Due: {formatted_min}, Due Date: {due_date}. Pay via NetBanking/App.",
                f"Statement alert: Total amount due on your {bank['name']} Card XX{account} is {formatted_amount}. Minimum payment {formatted_min} due on {due_date}.",
            ]
            text = random.choice(templates)
            entity = SMSEntity(
                type="bill",
                amount=amount,
                currency="INR",
                merchant=bank["name"],
                account=account,
                date_time=due_date,
                category="banking",
                spam=False,
            )
        elif b_type == "electricity":
            provider = random.choice(
                ["BESCOM", "Tata Power", "Adani Electricity", "MSEDCL", "Torrent Power"]
            )
            text = f"Dear Consumer {account}, electricity bill of {formatted_amount} for your connection is due on {due_date}. Pay via UPI/portal to avoid disconnection. - {provider}"
            entity = SMSEntity(
                type="bill",
                amount=amount,
                currency="INR",
                merchant=provider,
                account=account,
                date_time=due_date,
                category="utilities",
                spam=False,
            )
        else:
            telecom = random.choice(["Jio", "Airtel Postpaid", "Vi"])
            text = f"Bill Alert: Your {telecom} bill for mobile {account} is {formatted_amount}. Due date is {due_date}. Pay now on My{telecom} app to enjoy uninterrupted services."
            entity = SMSEntity(
                type="bill",
                amount=amount,
                currency="INR",
                merchant=telecom,
                account=account,
                date_time=due_date,
                category="utilities",
                spam=False,
            )

        return text, entity

    def generate_spam_sample(self) -> tuple[str, SMSEntity]:
        loan_amt = random.choice([200000, 300000, 500000, 1000000])
        win_amt = random.choice([25000, 50000, 100000])
        short_link = f"http://bit.ly/{random.choice(['instant-loan', 'sbi-update', 'rummy-cash', 'crypto-rich', 'win-gold'])}{random.randint(10, 99)}"

        spam_templates = [
            (
                f"Congratulations! Pre-approved Personal Loan of Rs.{loan_amt:,} approved with lowest interest rate! No documentation required. Click {short_link} to disburse money into your account.",
                SMSEntity(
                    type="spam", amount=float(loan_amt), currency="INR", category="spam", spam=True
                ),
            ),
            (
                f"Urgent Notice: Your SBI YONO account will be blocked today due to pending KYC verification. Click {short_link} to update Aadhaar/PAN immediately to avoid suspension.",
                SMSEntity(type="spam", merchant="SBI YONO", category="spam", spam=True),
            ),
            (
                f"Play Rummy & Win Real Cash! Get 100% Welcome Bonus up to Rs.{win_amt:,} on your first deposit today. Download official app: {short_link} T&C apply.",
                SMSEntity(
                    type="promo", amount=float(win_amt), currency="INR", category="spam", spam=True
                ),
            ),
            (
                f"FLAT 70% OFF + Extra 15% cashback on top fashion brands! Limited period festive sale on Myntra. Use code EXTRA15 at checkout. Shop now: {short_link}",
                SMSEntity(type="promo", merchant="Myntra", category="shopping", spam=True),
            ),
            (
                f"Earn Rs.50,000 monthly working from home part-time using your smartphone! No investment needed. Limited openings. Contact on WhatsApp: +91-98{random.randint(10000000, 99999999)}",
                SMSEntity(type="spam", amount=50000.0, currency="INR", category="spam", spam=True),
            ),
            (
                "Dear Customer, electricity power supply will be disconnected tonight at 9:30 PM due to unpaid bill. Immediately contact Electricity Officer at 9876543210. - Electricity Dept.",
                SMSEntity(type="spam", category="spam", spam=True),
            ),
            (
                f"Invest Rs.5000 and get Rs.25000 daily return guaranteed! Join our official Telegram trading group: {short_link}",
                SMSEntity(type="spam", amount=5000.0, currency="INR", category="spam", spam=True),
            ),
        ]

        text, entity = random.choice(spam_templates)
        return text, entity

    def generate_single_sample(self) -> tuple[str, SMSEntity]:
        """Generate a single sample with balanced distribution of SMS types."""
        # Distribution: 50% Debit (most common Indian SMS), 20% Credit/Refund, 15% OTP, 8% Bill, 7% Spam/Promo
        rand = random.random()
        if rand < 0.50:
            return self.generate_debit_sample()
        elif rand < 0.70:
            return self.generate_credit_sample()
        elif rand < 0.85:
            return self.generate_otp_sample()
        elif rand < 0.93:
            return self.generate_bill_sample()
        else:
            return self.generate_spam_sample()

    def generate_chat_sample(self) -> SMSChatSample:
        """Generate a single sample in Hugging Face chat format."""
        sms_text, entity = self.generate_single_sample()
        return SMSChatSample(
            messages=[
                SMSChatMessage(role="system", content=self.SYSTEM_PROMPT),
                SMSChatMessage(role="user", content=sms_text),
                SMSChatMessage(role="assistant", content=entity.to_compact_json()),
            ]
        )

    def generate_dataset(
        self,
        n_samples: int = 1000,
        val_split: float = 0.15,
        test_split: float = 0.05,
    ) -> dict[str, list[dict[str, Any]]]:
        """Generate train, val, and test splits."""
        all_samples = [self.generate_chat_sample().model_dump() for _ in range(n_samples)]
        random.shuffle(all_samples)

        n_test = int(n_samples * test_split)
        n_val = int(n_samples * val_split)
        n_train = n_samples - n_val - n_test

        train_data = all_samples[:n_train]
        val_data = all_samples[n_train : n_train + n_val]
        test_data = all_samples[n_train + n_val :]

        return {
            "train": train_data,
            "validation": val_data,
            "test": test_data,
        }
