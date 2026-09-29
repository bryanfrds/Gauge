"""The documented agreement bound between the two backends, enforced.

README.md and docs/ONNX.md promise ONNX scores stay within 0.002 of torch. That is a
claim about the real model, so it can't be checked with a fake one, and running it
needs both backends plus an export on disk — far too slow for the normal suite.

Opt in with:

    YN_SLOW_TESTS=1 python -m pytest tests/test_backend_agreement.py

It downloads the model on first run and needs `yn export-onnx` to have been run.
"""

from __future__ import annotations

import os

import pytest

from yn.model import DEFAULT_MODEL, Decider
from yn.onnx_backend import is_exported

# conftest's env-cleaning fixture strips YN_ONNX_DIR and YN_MODEL from every test, so
# capture them at import time and put them back for this module only.
_ONNX_DIR = os.environ.get("YN_ONNX_DIR")
_MODEL = os.environ.get("YN_MODEL")


@pytest.fixture(autouse=True)
def _restore_export_location(monkeypatch):
    if _ONNX_DIR:
        monkeypatch.setenv("YN_ONNX_DIR", _ONNX_DIR)
    if _MODEL:
        monkeypatch.setenv("YN_MODEL", _MODEL)

# The bound README.md and docs/ONNX.md state. Raise it only alongside those.
DOCUMENTED_BOUND = 0.002

pytestmark = [
    pytest.mark.skipif(not os.environ.get("YN_SLOW_TESTS"),
                       reason="set YN_SLOW_TESTS=1 (loads the real model)"),
    pytest.mark.skipif(not is_exported(os.environ.get("YN_MODEL") or DEFAULT_MODEL),
                       reason="no ONNX export; run: yn export-onnx"),
]

SHORT = [
    "Great app, very fast",
    "Terrible service today, nobody helped",
    "Quarterly results were announced this morning",
    "Saya suka perkhidmatan bank ini",
]
# Long enough to exercise truncation, which is where the backends drifted most.
LONG = ["The customer reported that "
        + "the transfer failed repeatedly and support was unreachable " * n
        for n in (10, 40)]
CLAIM = "the text describes a negative experience"
OPTIONS = ["positive", "negative", "neutral"]


@pytest.fixture(scope="module")
def scores():
    """Every input through both backends, as {backend: {key: confidence}}."""
    out = {}
    for backend in ("torch", "onnx"):
        d = Decider(backend=backend, device="cpu")
        out[backend] = {}
        for i, text in enumerate(SHORT + LONG):
            out[backend][f"check-{i}"] = d.check(text, CLAIM).scores["true"]
            out[backend][f"decide-{i}"] = d.decide(text, OPTIONS).confidence
    return out


def test_scores_stay_within_the_documented_bound(scores):
    worst = max((abs(scores["torch"][k] - scores["onnx"][k]), k) for k in scores["torch"])
    diff, where = worst
    assert diff < DOCUMENTED_BOUND, (
        f"{where} differs by {diff:.5f}, over the {DOCUMENTED_BOUND} documented in "
        f"README.md and docs/ONNX.md. Either the backends have diverged or the docs "
        f"need updating — do not simply raise this number."
    )


def test_both_backends_pick_the_same_answer(scores):
    """Confidence may drift; the verdict must not."""
    d_torch = Decider(backend="torch", device="cpu")
    d_onnx = Decider(backend="onnx")
    for text in SHORT + LONG:
        assert d_torch.check(text, CLAIM).answer == d_onnx.check(text, CLAIM).answer
        assert d_torch.decide(text, OPTIONS).answer == d_onnx.decide(text, OPTIONS).answer
