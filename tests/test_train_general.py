"""The multi-task training script's data shaping and calibration layout (no real training)."""

from __future__ import annotations

import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from train_general import (  # noqa: E402
    HELD_OUT_TASKS, TRAIN_TASKS, as_item, label_names, pick_options, score,
)
from train_choices import choose_temperature  # noqa: E402


class FakeDataset:
    """Just enough of a datasets.Dataset for label_names."""

    def __init__(self, columns: dict, features: dict | None = None):
        self._columns = columns
        self.column_names = list(columns)
        self.features = features or {c: SimpleNamespace() for c in columns}

    def __getitem__(self, col):
        return self._columns[col]


def test_names_from_class_label_with_pretty_rewrites():
    ds = FakeDataset({"label": [0]}, {"label": SimpleNamespace(names=["oos", "card_arrival"])})
    spec = {"name": "t", "pretty": {"oos": "something else"}}
    assert label_names(spec, ds) == ["something else", "card arrival"]


def test_names_from_label_text_column_in_id_order():
    ds = FakeDataset({"label": [1, 0, 1], "label_text": ["toxic", "not toxic", "toxic"]})
    assert label_names({"name": "t"}, ds) == ["not toxic", "toxic"]


def test_spec_names_win_over_the_dataset():
    ds = FakeDataset({"label": [0]}, {"label": SimpleNamespace(names=["0", "1"])})
    assert label_names({"name": "t", "names": ["bad", "good"]}, ds) == ["bad", "good"]


def test_missing_names_is_an_error():
    with pytest.raises(ValueError):
        label_names({"name": "t"}, FakeDataset({"label": [0]}))


def test_pick_options_keeps_the_right_answer_and_caps_the_count():
    rng = random.Random(0)
    for gold in range(20):
        ids, where = pick_options(gold, 20, 5, rng)
        assert len(ids) == 5 and len(set(ids)) == 5 and ids[where] == gold


def test_pick_options_offers_every_label_when_few():
    ids, where = pick_options(1, 3, 8, random.Random(0))
    assert sorted(ids) == [0, 1, 2] and ids[where] == 1


def test_as_item_builds_statements_and_points_at_the_right_one():
    spec = {"template": "This text is {label}."}
    tasks = {"t": (spec, ["sad", "happy", "angry"])}
    text, statements, gold = as_item({"text": "yay", "gold": 1, "task": "t"}, tasks, 8,
                                     random.Random(0))
    assert text == "yay"
    assert sorted(statements) == ["This text is angry.", "This text is happy.",
                                  "This text is sad."]
    assert statements[gold] == "This text is happy."


def test_every_template_ends_so_decide_uses_it_verbatim():
    for spec in TRAIN_TASKS + HELD_OUT_TASKS:
        assert spec["template"].rstrip()[-1] in ".!?", spec["name"]
        assert "{label}" in spec["template"], spec["name"]


def test_held_out_tasks_are_not_trained_on():
    trained = {(s["ids"][0], s.get("config")) for s in TRAIN_TASKS}
    held = {(s["ids"][0], s.get("config")) for s in HELD_OUT_TASKS}
    assert not trained & held


def test_padded_logits_calibrate_to_a_finite_temperature():
    # Two tasks with 2 and 4 options, padded with -inf the way check() does.
    torch.manual_seed(0)
    a = torch.randn(60, 2) * 3
    b = torch.randn(60, 4) * 3
    pad = lambda t: torch.nn.functional.pad(t, (0, 4 - t.shape[1]), value=float("-inf"))
    logits = torch.cat([pad(a), pad(b)])
    gold = torch.cat([torch.randint(0, 2, (60,)), torch.randint(0, 4, (60,))])
    temp, _ = choose_temperature(logits, gold)
    assert 0 < temp < float("inf")
    probs = (logits / temp).softmax(-1)
    assert torch.isfinite(probs).all()
    assert (probs.argmax(-1) == logits.argmax(-1)).all()
    s = score(logits, gold, temp)
    assert 0 <= s["accuracy"] <= 1 and s["ece"] == s["ece"]   # not NaN
