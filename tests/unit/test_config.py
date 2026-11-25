"""Unit tests for centralized settings and YAML configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_sms.core.config import load_yaml_config, settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_load_generation_yaml_config():
    cfg = load_yaml_config(PROJECT_ROOT / "config" / "generation_config.yaml")
    assert cfg["dataset"]["samples"] == 2500
    assert cfg["dataset"]["seed"] == 42


def test_load_train_yaml_config():
    cfg = load_yaml_config(PROJECT_ROOT / "config" / "train_config.yaml")
    assert cfg["model"]["model_id"] == "Qwen/Qwen3.5-2B"
    assert cfg["lora"]["r"] == 16
    assert cfg["training"]["num_train_epochs"] == 3


def test_load_yaml_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_yaml_config(tmp_path / "does_not_exist.yaml")


def test_settings_env_overrides(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SLM_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
    monkeypatch.setenv("SLM_MAX_NEW_TOKENS", "96")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "slm-custom")
    assert settings.model_id == "Qwen/Qwen2.5-1.5B-Instruct"
    assert settings.max_new_tokens == 96
    assert settings.otel_service_name == "slm-custom"
