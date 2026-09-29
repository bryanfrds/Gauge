"""Unit tests for the ONNX backend (no export run, no weights loaded).

Every test either points YN_ONNX_DIR at tmp_path or stubs the runner, so nothing here
downloads a model, exports one, or starts an ONNX Runtime session.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from yn import onnx_backend
from yn.model import BATCH_SIZE, MAX_STATEMENT_TOKENS, Decider, InputError, _softmax
from yn.onnx_backend import (
    MODEL_FILE,
    TOKENIZER_FILE,
    OnnxRunner,
    _slug,
    export_dir,
    is_exported,
)

MODEL = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"


def write_export(d: Path, entail_idx: int = 1, num_labels: int = 3) -> Path:
    """Minimal set of files that make `is_exported` true (contents are never parsed
    by the tests that only check existence)."""
    d.mkdir(parents=True, exist_ok=True)
    (d / MODEL_FILE).write_bytes(b"not-a-real-onnx-graph")
    (d / TOKENIZER_FILE).write_text("{}")
    (d / onnx_backend.META_FILE).write_text(
        json.dumps({"model": "fake-model", "entail_idx": entail_idx,
                    "num_labels": num_labels, "opset": onnx_backend.OPSET})
    )
    return d


class FakeRunner:
    """Stand-in for OnnxRunner: records the batches it was asked to score."""

    def __init__(self, entail_idx: int = 1, num_labels: int = 3, tokens: int = 5):
        self.entail_idx = entail_idx
        self.num_labels = num_labels
        self.tokens = tokens
        self.batches: list[list[tuple[str, str]]] = []
        self.counted: list[str] = []

    def count_tokens(self, text: str) -> int:
        self.counted.append(text)
        return self.tokens

    def logits(self, pairs):
        self.batches.append(list(pairs))
        return np.zeros((len(pairs), self.num_labels), dtype=np.float32)


# --- _slug -------------------------------------------------------------------------

def test_slug_keeps_safe_characters():
    assert _slug("deberta-v3_base.2") == "deberta-v3_base.2"


def test_slug_replaces_slashes_in_hub_name():
    assert _slug(MODEL) == "MoritzLaurer_deberta-v3-base-zeroshot-v2.0"


@pytest.mark.parametrize(
    "name, expected",
    [
        ("a b c", "a_b_c"),
        ("a//b", "a_b"),           # runs collapse to one underscore
        ("/leading", "leading"),   # separators at the ends are stripped
        ("trailing/", "trailing"),
        ("模型/名", "model"),       # nothing safe left at all
        ("..", "model"),           # would otherwise name the parent directory
        (".", "model"),
    ],
)
def test_slug_normalises_unsafe_names(name, expected):
    assert _slug(name) == expected


# --- export_dir --------------------------------------------------------------------

def test_export_dir_under_xdg_cache_home(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert export_dir(MODEL) == tmp_path / "yn" / "onnx" / _slug(MODEL)


def test_export_dir_falls_back_to_home_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    assert export_dir(MODEL) == tmp_path / ".cache" / "yn" / "onnx" / _slug(MODEL)


def test_export_dir_env_override_wins_over_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("YN_ONNX_DIR", str(tmp_path / "shipped"))
    assert export_dir(MODEL) == tmp_path / "shipped" / _slug(MODEL)


def test_export_dir_expands_tilde_in_override(monkeypatch, tmp_path):
    monkeypatch.setenv("YN_ONNX_DIR", "~/onnx")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert export_dir(MODEL) == tmp_path / "onnx" / _slug(MODEL)


def test_export_dir_ignores_empty_override(monkeypatch, tmp_path):
    monkeypatch.setenv("YN_ONNX_DIR", "")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert export_dir(MODEL) == tmp_path / "yn" / "onnx" / _slug(MODEL)


@pytest.mark.parametrize(
    "name",
    ["../../../etc/passwd", "/etc/passwd", "..", "a/../../b", "~/.ssh/id_rsa"],
)
def test_export_dir_stays_inside_root(monkeypatch, tmp_path, name):
    """A model name is attacker-shaped data; it must not walk out of the cache root."""
    monkeypatch.setenv("YN_ONNX_DIR", str(tmp_path))
    d = export_dir(name)
    assert d.parent == tmp_path
    assert tmp_path in d.resolve().parents


# --- is_exported -------------------------------------------------------------------

def test_is_exported_true_when_model_and_tokenizer_present(monkeypatch, tmp_path):
    monkeypatch.setenv("YN_ONNX_DIR", str(tmp_path))
    write_export(tmp_path / _slug("fake-model"))
    assert is_exported("fake-model") is True


def test_is_exported_false_when_directory_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("YN_ONNX_DIR", str(tmp_path))
    assert is_exported("fake-model") is False


def test_is_exported_false_without_tokenizer(monkeypatch, tmp_path):
    monkeypatch.setenv("YN_ONNX_DIR", str(tmp_path))
    d = write_export(tmp_path / _slug("fake-model"))
    (d / TOKENIZER_FILE).unlink()
    assert is_exported("fake-model") is False


def test_is_exported_false_without_model_file(monkeypatch, tmp_path):
    monkeypatch.setenv("YN_ONNX_DIR", str(tmp_path))
    d = write_export(tmp_path / _slug("fake-model"))
    (d / MODEL_FILE).unlink()
    assert is_exported("fake-model") is False


def test_is_exported_false_when_model_path_is_a_directory(monkeypatch, tmp_path):
    """A stray directory named model.onnx is not an export."""
    monkeypatch.setenv("YN_ONNX_DIR", str(tmp_path))
    d = tmp_path / _slug("fake-model")
    (d / MODEL_FILE).mkdir(parents=True)
    (d / TOKENIZER_FILE).write_text("{}")
    assert is_exported("fake-model") is False


def test_is_exported_is_per_model(monkeypatch, tmp_path):
    monkeypatch.setenv("YN_ONNX_DIR", str(tmp_path))
    write_export(tmp_path / _slug("fake-model"))
    assert is_exported("fake-model") is True
    assert is_exported("other/model") is False


# --- OnnxRunner without an export --------------------------------------------------

def test_runner_missing_export_names_the_export_command(monkeypatch, tmp_path):
    monkeypatch.setenv("YN_ONNX_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="yn export-onnx"):
        OnnxRunner("fake-model")


def test_runner_missing_export_names_the_model_and_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("YN_ONNX_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError) as e:
        OnnxRunner("fake-model")
    assert "fake-model" in str(e.value)
    assert str(tmp_path / _slug("fake-model")) in str(e.value)


def test_runner_missing_export_raises_before_importing_onnxruntime(monkeypatch, tmp_path):
    """The failure must be the clear message even where onnxruntime isn't installed."""
    import builtins

    monkeypatch.setenv("YN_ONNX_DIR", str(tmp_path))
    real_import = builtins.__import__

    def no_onnxruntime(name, *args, **kw):
        if name == "onnxruntime":
            raise ImportError("onnxruntime is not installed")
        return real_import(name, *args, **kw)

    monkeypatch.setattr(builtins, "__import__", no_onnxruntime)
    with pytest.raises(FileNotFoundError, match="no ONNX export"):
        OnnxRunner("fake-model")


# --- _softmax ----------------------------------------------------------------------

def test_softmax_rows_sum_to_one():
    a = np.array([[1.0, 2.0, 3.0], [-5.0, 0.0, 0.5]], dtype=np.float32)
    assert _softmax(a).sum(axis=-1) == pytest.approx([1.0, 1.0])


def test_softmax_matches_hand_computed_values():
    # logits [0, ln 3] => [1/4, 3/4]
    a = np.array([[0.0, np.log(3.0)]])
    assert _softmax(a)[0] == pytest.approx([0.25, 0.75])


def test_softmax_uniform_logits_are_uniform():
    assert _softmax(np.zeros((1, 4)))[0] == pytest.approx([0.25] * 4)


def test_softmax_is_stable_with_large_logits():
    """Un-shifted exp() would overflow to inf and then nan."""
    a = np.array([[1000.0, 1000.0, 1000.0]])
    out = _softmax(a)
    assert np.isfinite(out).all()
    assert out[0] == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_softmax_is_stable_with_large_negative_logits():
    out = _softmax(np.array([[-1000.0, -1000.0]]))
    assert np.isfinite(out).all()
    assert out[0] == pytest.approx([0.5, 0.5])


def test_softmax_saturates_without_nan_on_extreme_gap():
    out = _softmax(np.array([[-800.0, 800.0]]))
    assert np.isfinite(out).all()
    assert out[0] == pytest.approx([0.0, 1.0])


def test_softmax_is_shift_invariant():
    a = np.array([[1.0, 2.0, 3.0]])
    assert _softmax(a) == pytest.approx(_softmax(a + 500.0))


def test_softmax_over_axis_zero():
    a = np.array([[0.0, 0.0], [np.log(3.0), np.log(3.0)]])
    assert _softmax(a, axis=0).ravel() == pytest.approx([0.25, 0.25, 0.75, 0.75])


def test_softmax_single_column_is_all_ones():
    assert _softmax(np.array([[5.0], [-5.0]])).ravel() == pytest.approx([1.0, 1.0])


# --- Decider on the ONNX runner ----------------------------------------------------

@pytest.fixture
def onnx_decider(monkeypatch, tmp_path):
    """Decider whose load() installs a FakeRunner instead of a real ONNX session."""
    monkeypatch.setenv("YN_ONNX_DIR", str(tmp_path))
    write_export(tmp_path / _slug("fake-model"))
    runner = FakeRunner(entail_idx=1)
    monkeypatch.setattr(onnx_backend, "OnnxRunner", lambda name: runner)
    d = Decider(model_name="fake-model", threshold=0.85, backend="onnx")
    return d, runner


def test_load_installs_runner_and_entail_index(onnx_decider):
    d, runner = onnx_decider
    d.load()
    assert d._runner is runner
    assert d._entail_idx == runner.entail_idx


def test_load_forces_cpu_device_for_onnx(onnx_decider):
    d, _ = onnx_decider
    d.device = "mps"
    d.load()
    assert d.device == "cpu"


def test_load_does_not_build_a_torch_model(onnx_decider):
    d, _ = onnx_decider
    d.load()
    assert d._model is None and d._tokenizer is None


def test_load_is_idempotent(monkeypatch, onnx_decider):
    d, runner = onnx_decider
    d.load()
    monkeypatch.setattr(
        onnx_backend, "OnnxRunner",
        lambda name: pytest.fail("load() must build the runner only once"),
    )
    d.load()
    assert d._runner is runner


def test_logits_from_runner_is_numpy_float32(onnx_decider):
    d, _ = onnx_decider
    out = d._logits([("a", "This is spam."), ("b", "This is spam.")])
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    assert out.shape == (2, 3)


def test_logits_batches_pairs_and_concatenates(onnx_decider):
    d, runner = onnx_decider
    pairs = [(f"text {i}", "This is spam.") for i in range(BATCH_SIZE + 3)]
    out = d._logits(pairs)
    assert [len(b) for b in runner.batches] == [BATCH_SIZE, 3]
    assert [p for b in runner.batches for p in b] == pairs
    assert out.shape == (BATCH_SIZE + 3, 3)


def test_check_uses_runner_scores(onnx_decider, monkeypatch):
    d, runner = onnx_decider

    def logits(pairs):
        out = np.zeros((len(pairs), 3), dtype=np.float32)
        out[:, 1] = np.log(2.0)  # P(entailment) = 2/(2+1+1) = 0.5
        return out

    monkeypatch.setattr(runner, "logits", logits)
    r = d.check("you won a free iPhone", "This is spam.")
    assert r.scores == {"true": 0.5, "false": 0.5}
    assert r.sure is False


def test_statement_token_limit_uses_the_runner_tokenizer(onnx_decider, monkeypatch):
    d, runner = onnx_decider
    monkeypatch.setattr(runner, "tokens", MAX_STATEMENT_TOKENS + 1)
    with pytest.raises(InputError, match="tokens; the limit is"):
        d.check("hello", "这是一个很长的说法。")
    assert runner.batches == []


def test_statement_at_token_limit_is_allowed(onnx_decider, monkeypatch):
    d, runner = onnx_decider
    monkeypatch.setattr(runner, "tokens", MAX_STATEMENT_TOKENS)
    d.check("hello", "This is spam.")
    assert len(runner.batches) == 1


def test_check_statements_counts_each_statement_once(onnx_decider):
    d, runner = onnx_decider
    d.check_statements(["a", "b", "a"])
    assert sorted(runner.counted) == ["a", "b"]
    assert runner.batches == []
