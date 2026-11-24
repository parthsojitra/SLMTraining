"""Evaluation metrics and benchmarking for SMS structured entity extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from slm_sms.data.schema import SMSEntity


@dataclass
class ExtractionMetricsReport:
    total_samples: int = 0
    valid_json_count: int = 0
    json_validity_rate: float = 0.0

    # Field-level accuracies (computed over valid JSON predictions)
    type_matches: int = 0
    type_accuracy: float = 0.0

    amount_matches: int = 0
    amount_total: int = 0
    amount_accuracy: float = 0.0

    merchant_matches: int = 0
    merchant_total: int = 0
    merchant_accuracy: float = 0.0

    account_matches: int = 0
    account_total: int = 0
    account_accuracy: float = 0.0

    # Spam classification metrics
    spam_tp: int = 0
    spam_fp: int = 0
    spam_fn: int = 0
    spam_tn: int = 0
    spam_precision: float = 0.0
    spam_recall: float = 0.0
    spam_f1: float = 0.0

    # Overall entity score
    overall_score: float = 0.0
    field_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "valid_json_count": self.valid_json_count,
            "json_validity_rate": round(self.json_validity_rate, 4),
            "type_accuracy": round(self.type_accuracy, 4),
            "amount_accuracy": round(self.amount_accuracy, 4),
            "merchant_accuracy": round(self.merchant_accuracy, 4),
            "account_accuracy": round(self.account_accuracy, 4),
            "spam_precision": round(self.spam_precision, 4),
            "spam_recall": round(self.spam_recall, 4),
            "spam_f1": round(self.spam_f1, 4),
            "overall_score": round(self.overall_score, 4),
        }


def normalize_str(s: str | None) -> str:
    if not s:
        return ""
    return s.strip().lower().replace("-", "").replace(" ", "").replace("_", "")


def compute_extraction_metrics(
    predictions: list[str],
    ground_truths: list[dict[str, Any]],
) -> ExtractionMetricsReport:
    """Evaluate raw text predictions against ground truth entity dicts."""
    report = ExtractionMetricsReport(total_samples=len(ground_truths))
    if not ground_truths:
        return report

    for raw_pred, gt_dict in zip(predictions, ground_truths):
        gt = SMSEntity(**gt_dict) if isinstance(gt_dict, dict) else gt_dict

        # Try parsing prediction
        pred: SMSEntity | None = None
        try:
            pred = SMSEntity.from_raw_text(raw_pred)
            report.valid_json_count += 1
        except Exception:
            # Invalid JSON prediction
            pass

        # Evaluate spam detection
        gt_is_spam = bool(gt.spam)
        pred_is_spam = bool(pred.spam) if pred else False
        if gt_is_spam and pred_is_spam:
            report.spam_tp += 1
        elif not gt_is_spam and pred_is_spam:
            report.spam_fp += 1
        elif gt_is_spam and not pred_is_spam:
            report.spam_fn += 1
        else:
            report.spam_tn += 1

        if pred is None:
            continue

        # Type evaluation
        if pred.type.lower() == gt.type.lower():
            report.type_matches += 1

        # Amount evaluation
        if gt.amount is not None:
            report.amount_total += 1
            if pred.amount is not None and abs(pred.amount - gt.amount) < 0.5:
                report.amount_matches += 1

        # Merchant evaluation
        if gt.merchant:
            report.merchant_total += 1
            if (
                pred.merchant
                and normalize_str(pred.merchant) in normalize_str(gt.merchant)
                or normalize_str(gt.merchant) in normalize_str(pred.merchant or "")
            ):
                report.merchant_matches += 1

        # Account evaluation
        if gt.account:
            report.account_total += 1
            if pred.account and pred.account[-4:] == gt.account[-4:]:
                report.account_matches += 1

    # Calculate ratios
    n = report.total_samples
    report.json_validity_rate = report.valid_json_count / n if n > 0 else 0.0
    report.type_accuracy = report.type_matches / n if n > 0 else 0.0

    report.amount_accuracy = (
        report.amount_matches / report.amount_total if report.amount_total > 0 else 1.0
    )
    report.merchant_accuracy = (
        report.merchant_matches / report.merchant_total if report.merchant_total > 0 else 1.0
    )
    report.account_accuracy = (
        report.account_matches / report.account_total if report.account_total > 0 else 1.0
    )

    # Spam PR / F1
    precision_denom = report.spam_tp + report.spam_fp
    recall_denom = report.spam_tp + report.spam_fn
    report.spam_precision = report.spam_tp / precision_denom if precision_denom > 0 else 1.0
    report.spam_recall = report.spam_tp / recall_denom if recall_denom > 0 else 1.0
    if report.spam_precision + report.spam_recall > 0:
        report.spam_f1 = (
            2
            * report.spam_precision
            * report.spam_recall
            / (report.spam_precision + report.spam_recall)
        )
    else:
        report.spam_f1 = 0.0

    # Overall balanced score
    report.overall_score = (
        0.25 * report.json_validity_rate
        + 0.25 * report.type_accuracy
        + 0.20 * report.amount_accuracy
        + 0.15 * report.merchant_accuracy
        + 0.15 * report.account_accuracy
    )

    report.field_scores = {
        "json_validity": report.json_validity_rate,
        "type": report.type_accuracy,
        "amount": report.amount_accuracy,
        "merchant": report.merchant_accuracy,
        "account": report.account_accuracy,
        "spam_f1": report.spam_f1,
    }

    return report
