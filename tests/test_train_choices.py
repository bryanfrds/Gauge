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
    calib = json.loads((out / "gauge_calibration.json").read_text())
    assert calib["temperature"] == 1.0 and "6 validation rows" in calib["not_fitted"]


def test_the_saved_calibration_is_the_one_measured(tmp_path, tiny_base, monkeypatch):
    """What lands in the file is the chosen temperature and the ECE measured with it."""
    import train_choices

    seen = []
    real_report = train_choices.report

    def spy(logits, rows, labels, temperature=1.0, **kw):
        out = real_report(logits, rows, labels, temperature, **kw)
        seen.append((temperature, len(rows), out["ece"]))
        return out

    monkeypatch.setattr(train_choices, "choose_temperature", lambda lg, gold: (0.07346, None))
    monkeypatch.setattr(train_choices, "report", spy)
    rows = [{"title": "", "body": f"post {i}", "label": LABELS[i % 3], "entity": "Acme Bank"}
            for i in range(40)]
    out = tmp_path / "out"
    run_main(monkeypatch, ["--data", write(tmp_path / "d.csv", rows), "--labels", *LABELS,
                           "--test", write(tmp_path / "t.csv", rows[:7]),
                           "--template", TEMPLATE, "--out", str(out), "--base", str(tiny_base),
                           "--device", "cpu", "--batch", "4", "--max-len", "32",
                           "--val-frac", "0.25"])
    calib = json.loads((out / "gauge_calibration.json").read_text())
    assert calib["temperature"] == 0.0735
    after = next(e for t, n, e in seen if t == 0.0735 and n == 10)
    before = next(e for t, n, e in seen[1:] if t == 1.0 and n == 10)
    assert after != before   # a sharp enough temperature to tell the two apart
    assert calib["ece"]["val"] == {"before": before, "after": after}
    assert calib["ece"]["test"] == {
        "before": next(e for t, n, e in seen[2:] if t == 1.0 and n == 7),
        "after": next(e for t, n, e in seen if t == 0.0735 and n == 7)}


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


def test_ece_is_the_weighted_gap_between_confidence_and_accuracy():
    from train_choices import ece

    assert ece([0.9] * 10, [True] * 9 + [False]) == pytest.approx(0.0)
    # Says 0.95 but is right half the time; says 0.55 and is always right.
    assert ece([0.95, 0.95, 0.55, 0.55], [True, False, True, True]) == pytest.approx(
        0.5 * 0.45 + 0.5 * 0.45)


def test_ece_counts_certain_scores_and_bin_edges():
    from train_choices import ece

    # 1.0 is common after a float32 softmax; it must land in the top bin, not vanish.
    assert ece([1.0, 1.0], [False, False]) == pytest.approx(1.0)
    # 0.5 sits on an edge: it belongs to (0.4, 0.5], with the 0.45 beside it.
    assert ece([0.5, 0.45], [True, True]) == pytest.approx(0.525)


def test_temperature_softens_overconfident_scores_without_changing_answers():
    from train_choices import fit_temperature, report

    torch.manual_seed(0)
    gold = torch.randint(0, 3, (600,))
    # Right 70% of the time, but the margins claim near certainty.
    wrong = (gold + torch.randint(1, 3, gold.shape)) % 3
    picked = torch.where(torch.rand(600) < 0.7, gold, wrong)
    logits = torch.nn.functional.one_hot(picked, 3).float() * 8
    rows = [{"_label": LABELS[g]} for g in gold.tolist()]
    t = fit_temperature(logits, gold)
    raw, fitted = report(logits, rows, LABELS), report(logits, rows, LABELS, t)
    assert t > 1.5
    assert fitted["accuracy"] == raw["accuracy"]
    assert fitted["ece"] < 0.05 < raw["ece"]


def test_fitting_works_on_logits_made_in_inference_mode():
    from train_choices import fit_temperature

    with torch.inference_mode():
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0]])
    assert fit_temperature(logits, torch.tensor([0, 1, 1, 1])) > 0
    with torch.inference_mode():   # and when called from inside inference code
        assert fit_temperature(logits, torch.tensor([0, 1, 1, 1])) > 0


def test_a_perfect_or_tiny_validation_set_keeps_scores_as_they_are():
    from train_choices import MIN_CALIBRATION_ROWS, choose_temperature

    gold = torch.arange(60) % 2
    perfect = torch.nn.functional.one_hot(gold, 2).float()
    assert choose_temperature(perfect, gold) == (1.0, "no validation mistakes to calibrate against")
    t, why = choose_temperature(perfect[:3], torch.tensor([1, 0, 0]))
    assert t == 1.0 and str(MIN_CALIBRATION_ROWS) in why


def test_the_fitted_temperature_stays_in_a_sensible_range():
    from train_choices import TEMPERATURE_RANGE, fit_temperature

    assert TEMPERATURE_RANGE == (0.05, 20.0)
    gold = torch.arange(60) % 2
    # Right 59 times in 60 by a hair: the best fit is a tiny T, i.e. "always certain".
    logits = torch.nn.functional.one_hot(gold, 2).float() * 1e-3
    logits[0] = logits[0].flip(0)
    assert fit_temperature(logits, gold) == pytest.approx(TEMPERATURE_RANGE[0], rel=1e-3)
    # Wrong half the time yet sure: the best fit is "know nothing", the top of the range.
    assert fit_temperature(torch.tensor([[5.0, 0.0]] * 60), gold) == pytest.approx(
        TEMPERATURE_RANGE[1], rel=1e-3)
