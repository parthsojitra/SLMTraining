"""Dataset utilities for saving, loading, and formatting Indian SMS data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict


def save_jsonl(samples: list[dict[str, Any]], filepath: str | Path) -> Path:
    """Save records to a JSONL file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(sample, ensure_ascii=False) + "\n" for sample in samples)
    return path


def load_jsonl(filepath: str | Path) -> list[dict[str, Any]]:
    """Load records from a JSONL file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at {path}")
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_hf_dataset(
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]] | None = None,
    test_records: list[dict[str, Any]] | None = None,
) -> DatasetDict:
    """Convert raw message dicts into a Hugging Face DatasetDict."""
    dataset_dict = {"train": Dataset.from_list(train_records)}
    if val_records:
        dataset_dict["validation"] = Dataset.from_list(val_records)
    if test_records:
        dataset_dict["test"] = Dataset.from_list(test_records)
    return DatasetDict(dataset_dict)


def load_hf_dataset_from_dir(data_dir: str | Path) -> DatasetDict:
    """Load train.jsonl, val.jsonl, test.jsonl from directory into Hugging Face DatasetDict."""
    d = Path(data_dir)
    train_path = d / "train.jsonl"
    val_path = d / "val.jsonl"
    test_path = d / "test.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing train.jsonl at {train_path}")

    train_data = load_jsonl(train_path)
    val_data = load_jsonl(val_path) if val_path.exists() else None
    test_data = load_jsonl(test_path) if test_path.exists() else None

    return build_hf_dataset(train_data, val_data, test_data)
