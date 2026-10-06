"""The training script's data handling and scoring layout (no real training here)."""

from __future__ import annotations

import csv
import json
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


def test_label_in_the_template_is_the_option_even_with_a_label_column():
    row = {"entity": "Acme Bank", "label": "Negative"}
    assert statements(row, LABELS, "{label}: {entity}.") == [
        "Positive: Acme Bank.", "Neutral: Acme Bank.", "Negative: Acme Bank."]


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


# --- the training loop, end to end on a tiny random model ---------------------------

from train_choices import lr_factor  # noqa: E402


def test_lr_warms_up_then_decays_to_near_zero():
    steps, warmup = 100, 10
    assert lr_factor(0, steps, warmup) == pytest.approx(0.1)
    assert lr_factor(9, steps, warmup) == pytest.approx(1.0)
    assert lr_factor(55, steps, warmup) == pytest.approx(0.5)
    assert 0 < lr_factor(99, steps, warmup) < 0.02
    assert min(lr_factor(s, steps, warmup) for s in range(steps + 5)) >= 0


class FakeTokenizer:
    """Words hashed into a tiny vocabulary: enough to run the real model."""

    def __call__(self, texts, hyps, truncation, max_length, padding, return_tensors):
        seqs = [[1] + [hash(w) % 90 + 5 for w in f"{t} {h}".split()][:max_length - 2] + [2]
                for t, h in zip(texts, hyps)]
        width = max(map(len, seqs))
        ids = torch.tensor([s + [0] * (width - len(s)) for s in seqs])
        mask = (ids != 0).long()
        return SimpleNamespace(to=lambda device: {"input_ids": ids, "attention_mask": mask})

    def save_pretrained(self, path):
        Path(path, "tokenizer_saved").write_text("fake")


@pytest.fixture
def tiny_base(tmp_path, monkeypatch):
    """A random DeBERTa-v2 NLI head, saved in float16 like the real hub model."""
    from transformers import DebertaV2Config, DebertaV2ForSequenceClassification
    import transformers

    torch.manual_seed(0)
    cfg = DebertaV2Config(vocab_size=100, hidden_size=32, num_hidden_layers=2,
                          num_attention_heads=2, intermediate_size=64,
                          max_position_embeddings=64, relative_attention=True,
                          position_buckets=8, max_relative_positions=16,
                          pos_att_type=["p2c", "c2p"], num_labels=3,
                          id2label={0: "entailment", 1: "neutral", 2: "contradiction"},
                          label2id={"entailment": 0, "neutral": 1, "contradiction": 2})
    base = tmp_path / "base"
    DebertaV2ForSequenceClassification(cfg).half().save_pretrained(base)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        staticmethod(lambda *a, **k: FakeTokenizer()))
    import train_choices
    monkeypatch.setattr(train_choices, "AutoTokenizer", transformers.AutoTokenizer)
    return base


def run_main(monkeypatch, argv):
    import train_choices

    monkeypatch.setattr(sys, "argv", ["train_choices.py", *argv])
    train_choices.main()


def test_training_runs_end_to_end_and_saves_a_sound_model(tmp_path, tiny_base, monkeypatch):
    from safetensors.torch import load_file

    rows = [{"title": "", "body": f"post {i} {'great' if i % 3 == 0 else 'awful' if i % 3 == 1 else 'okay'}",
             "label": LABELS[i % 3], "entity": "Acme Bank"} for i in range(24)]
    data = write(tmp_path / "d.csv", rows)
    out = tmp_path / "out"
    run_main(monkeypatch, ["--data", data, "--labels", *LABELS, "--template", TEMPLATE,
                           "--out", str(out), "--base", str(tiny_base), "--device", "cpu",
                           "--epochs", "2", "--batch", "4", "--max-len", "32",
                           "--checkpointing", "--val-frac", "0.25", "--lr", "1e-3"])
    before = load_file(tiny_base / "model.safetensors")
    after = load_file(out / "model.safetensors")
    # float16 weights under Adam turn to NaN on the second step; they must not.
    assert all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in after.values())
    emb = next(k for k in after if "word_embeddings" in k)
    assert torch.equal(after[emb], before[emb].float())          # frozen
    enc = next(k for k in after if "encoder.layer.0" in k and k.endswith("weight"))
    assert not torch.equal(after[enc], before[enc].float())      # trained, despite checkpointing
    summary = json.loads((out / "gauge_training.json").read_text())
    assert summary["epoch"] in (1, 2)
    assert summary["labels"] == LABELS and summary["template"] == TEMPLATE
    assert {"val"} <= summary["before"].keys() and {"val"} <= summary["after"].keys()
    assert "data" not in summary["args"]                         # no private paths saved


def test_a_template_column_the_data_lacks_stops_before_loading(tmp_path, monkeypatch):
    data = write(tmp_path / "d.csv", [{"title": "", "body": "ok", "label": "Neutral"}])
    with pytest.raises(SystemExit, match="entity"):
        run_main(monkeypatch, ["--data", data, "--labels", *LABELS, "--template", TEMPLATE,
                               "--out", str(tmp_path / "o"), "--base", "/nonexistent"])


def test_a_template_without_end_punctuation_is_refused(tmp_path, monkeypatch):
    data = write(tmp_path / "d.csv", [{"title": "", "body": "ok", "label": "Neutral"}])
    with pytest.raises(SystemExit, match="must end"):
        run_main(monkeypatch, ["--data", data, "--labels", *LABELS, "--template", "{label}",
                               "--out", str(tmp_path / "o"), "--base", "/nonexistent"])


def test_zero_epochs_is_a_usage_error(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        run_main(monkeypatch, ["--data", "x", "--labels", "A", "--template", "t.",
                               "--out", "o", "--epochs", "0"])
