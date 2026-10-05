"""The training script's data handling and scoring layout (no real training here)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from train_choices import MAX_CHARS, choice_logits, read_rows, statements  # noqa: E402

LABELS = ["Positive", "Neutral", "Negative"]
TEMPLATE = "This post is {label_lower} about {entity}."


def write(path: Path, rows: list[dict]) -> str:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return str(path)


def test_reads_title_and_body_as_one_text(tmp_path):
    p = write(tmp_path / "d.csv", [{"title": "Outage", "body": "App down all day", "label": "Negative"}])
    assert read_rows(p, ["title", "body"], "label", LABELS)[0]["_text"] == "Outage\nApp down all day"


def test_a_title_the_body_repeats_is_said_once(tmp_path):
    p = write(tmp_path / "d.csv", [{"title": "Outage", "body": "Outage: app down", "label": "Negative"}])
    assert read_rows(p, ["title", "body"], "label", LABELS)[0]["_text"] == "Outage: app down"


def test_rows_with_unknown_labels_or_no_text_are_skipped(tmp_path):
    p = write(tmp_path / "d.csv", [
        {"title": "", "body": "fine", "label": "Mixed"},
        {"title": "", "body": "  ", "label": "Neutral"},
        {"title": "", "body": "ok", "label": "Neutral"},
    ])
    assert [r["_text"] for r in read_rows(p, ["title", "body"], "label", LABELS)] == ["ok"]


def test_long_texts_are_cut_and_huge_fields_are_readable(tmp_path):
    p = write(tmp_path / "d.csv", [{"title": "", "body": "x" * 200_000, "label": "Neutral"}])
    assert len(read_rows(p, ["title", "body"], "label", LABELS)[0]["_text"]) == MAX_CHARS


def test_statements_name_each_rows_own_subject():
    assert statements({"entity": "Acme Bank"}, LABELS, TEMPLATE) == [
        "This post is positive about Acme Bank.",
        "This post is neutral about Acme Bank.",
        "This post is negative about Acme Bank.",
    ]


def test_a_template_column_missing_from_the_data_is_an_error():
    with pytest.raises(KeyError):
        statements({}, LABELS, TEMPLATE)


def test_logits_come_back_one_row_per_text_one_column_per_label():
    """Pairs are sent text-major; the reshape must put each label in its own column.
    The fake model scores a pair by which label its statement names."""
    seen = []

    def tok(texts, hyps, **kw):
        seen.append((texts, hyps))
        ids = torch.tensor([[LABELS.index(next(l for l in LABELS if l.lower() in h))]
                            for h in hyps])
        return SimpleNamespace(to=lambda device: {"input_ids": ids})

    def model(input_ids):
        n = len(input_ids)
        logits = torch.zeros(n, 3)
        logits[:, 1] = input_ids[:, 0].float() * 10   # entailment column = label index
        return SimpleNamespace(logits=logits)

    rows = [{"entity": "A", "_text": "t1"}, {"entity": "B", "_text": "t2"}]
    out = choice_logits(model, tok, rows, LABELS, TEMPLATE, "cpu", entail=1, max_len=64)
    assert out.shape == (2, 3)
    assert out.tolist() == [[0, 10, 20], [0, 10, 20]]
    texts, hyps = seen[0]
    assert texts == ["t1"] * 3 + ["t2"] * 3
    assert hyps[3] == "This post is positive about B."
