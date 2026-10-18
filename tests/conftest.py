import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Deterministic env: synthetic detector on, opt-in flag off unless a test sets it."""
    monkeypatch.setenv("SYNTHETIC_DETECTOR", "1")
    for k in ("ALLOW_UNVALIDATED_RULES", "ONNX_MODEL", "OPERATOR_ZONE"):
        monkeypatch.delenv(k, raising=False)
