"""Shared pytest configuration and offline-safe server fixtures."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _offline_test_defaults(monkeypatch: pytest.MonkeyPatch):
    """Ensure fast offline server defaults unless a test explicitly overrides them."""
    if "SLM_LAZY_LOAD" not in os.environ:
        monkeypatch.setenv("SLM_LAZY_LOAD", "true")
    monkeypatch.setenv("SLM_DRAIN_SECONDS", "0")
