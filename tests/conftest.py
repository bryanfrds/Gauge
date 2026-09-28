"""Shared fixtures.

Unit tests never load the real model: `fake_model` replaces `Decider._logits` with a
lookup table and makes `Decider.load` fail loudly if anything reaches it.
"""

from __future__ import annotations

import math

import pytest
import torch

from yn.model import Decider

NUM_LABELS = 3  # entailment + two others, so the entailment index actually matters


class FakeModel:
    """Deterministic stand-in for the NLI model.

    `entail[(text, statement)]` is the raw entailment logit for that pair; the other
    label columns are 0. Unlisted pairs get logit 0.
    """

    def __init__(self):
        self.entail: dict[tuple[str, str], float] = {}
        self.calls: list[list[tuple[str, str]]] = []

    def logits(self, decider: Decider, pairs):
        self.calls.append(list(pairs))
        out = torch.zeros(len(pairs), NUM_LABELS)
        for i, pair in enumerate(pairs):
            out[i, decider._entail_idx] = self.entail.get(pair, 0.0)
        return out

    # Helpers to pick logits that give an exact, known probability.
    @staticmethod
    def check_logit(p_true: float) -> float:
        """Entail logit whose 3-way softmax gives P(entailment) == p_true."""
        return math.log((NUM_LABELS - 1) * p_true / (1 - p_true))

    @staticmethod
    def decide_logit(weight: float) -> float:
        """Logit such that softmax over options is proportional to `weight`."""
        return math.log(weight)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("YN_MODEL", "YN_THRESHOLD", "YN_DEVICE", "YN_VERBOSE", "YN_ROUTES"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def fake_model(monkeypatch):
    fake = FakeModel()

    def fake_logits(self, pairs):
        return fake.logits(self, pairs)

    def no_load(self):
        raise AssertionError("unit tests must not load the real model")

    monkeypatch.setattr(Decider, "_logits", fake_logits)
    monkeypatch.setattr(Decider, "load", no_load)
    return fake


@pytest.fixture
def decider(fake_model):
    d = Decider(model_name="fake-model", threshold=0.85)
    d._entail_idx = 1  # non-zero, so reading the wrong column shows up
    return d
