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


def write_export(d: Path, entail_idx: int = 1, num_labels: int = 3,
                 model: str = "fake-model", format_version: int | None = None,
                 meta: dict | None = None) -> Path:
    """Every file a real export has, so `is_exported` is true. Contents are only
    parsed for meta.json; the graph and tokenizer are placeholders."""
    d.mkdir(parents=True, exist_ok=True)
    (d / MODEL_FILE).write_bytes(b"not-a-real-onnx-graph")
    (d / onnx_backend.WEIGHTS_FILE).write_bytes(b"not-real-weights")
    (d / TOKENIZER_FILE).write_text("{}")
    payload = {
        "format_version": onnx_backend.FORMAT_VERSION
        if format_version is None else format_version,
        "model": model,
        "entail_idx": entail_idx,
        "num_labels": num_labels,
        "opset": onnx_backend.OPSET,
        "pad_id": 0,
        "pad_token": "[PAD]",
    }
    payload.update(meta or {})
    (d / onnx_backend.META_FILE).write_text(json.dumps(payload))
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
    d = write_export(tmp_path / _slug("fake-model"))
    (d / MODEL_FILE).unlink()
    (d / MODEL_FILE).mkdir()
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


# --- a partial or stale export must not be trusted ---------------------------------
# `auto` promises a silent fallback to torch. Anything it wrongly calls an export
# fails on every later call instead, so each missing or wrong piece is pinned here.

def test_is_exported_false_without_weights_sidecar(monkeypatch, tmp_path):
    """model.onnx is a stub; ONNX Runtime opens model.onnx.data for the weights."""
    monkeypatch.setenv("YN_ONNX_DIR", str(tmp_path))
    d = write_export(tmp_path / _slug("fake-model"))
    (d / onnx_backend.WEIGHTS_FILE).unlink()
    assert is_exported("fake-model") is False


def test_is_exported_false_without_meta(monkeypatch, tmp_path):
    """OnnxRunner reads meta.json, so an export without it is not usable."""
    monkeypatch.setenv("YN_ONNX_DIR", str(tmp_path))
    d = write_export(tmp_path / _slug("fake-model"))
    (d / onnx_backend.META_FILE).unlink()
    assert is_exported("fake-model") is False


def test_is_exported_false_when_meta_is_unreadable(monkeypatch, tmp_path):
    monkeypatch.setenv("YN_ONNX_DIR", str(tmp_path))
    d = write_export(tmp_path / _slug("fake-model"))
    (d / onnx_backend.META_FILE).write_text("{not json")
    assert is_exported("fake-model") is False


def test_is_exported_false_on_older_format(monkeypatch, tmp_path):
    """An export from older code may not match today's graph or tokenizer settings."""
    monkeypatch.setenv("YN_ONNX_DIR", str(tmp_path))
    write_export(tmp_path / _slug("fake-model"),
                 format_version=onnx_backend.FORMAT_VERSION - 1)
    assert is_exported("fake-model") is False


def test_is_exported_false_when_meta_names_another_model(monkeypatch, tmp_path):
    """_slug is lossy: "org/model" and "org_model" share a directory."""
    monkeypatch.setenv("YN_ONNX_DIR", str(tmp_path))
    assert _slug("org/model") == _slug("org_model")  # the collision this guards
    write_export(tmp_path / _slug("org/model"), model="org/model")
    assert is_exported("org/model") is True
    assert is_exported("org_model") is False


def test_runner_rejects_an_export_for_another_model(monkeypatch, tmp_path):
    monkeypatch.setenv("YN_ONNX_DIR", str(tmp_path))
    write_export(tmp_path / _slug("org_model"), model="org/model")
    with pytest.raises(RuntimeError, match="yn export-onnx"):
        OnnxRunner("org_model")


def test_runner_rejects_an_older_format(monkeypatch, tmp_path):
    monkeypatch.setenv("YN_ONNX_DIR", str(tmp_path))
    write_export(tmp_path / _slug("fake-model"),
                 format_version=onnx_backend.FORMAT_VERSION - 1)
    with pytest.raises(RuntimeError, match="yn export-onnx"):
        OnnxRunner("fake-model")


# --- tokenizer settings ------------------------------------------------------------
# These pin the two properties that fail silently: a wrong truncation strategy scores
# a clipped claim as if it were whole, and a truncated counter misreports the size in
# the error a user has to act on.

def _tiny_tokenizer(path: Path, max_length: int = 512) -> None:
    """A real tokenizers.Tokenizer saved the way export() saves one: with the
    truncation settings already baked in."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"[PAD]": 0, "[UNK]": 1, "word": 2, "claim": 3}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.enable_truncation(max_length, strategy="only_first")
    tok.enable_padding()
    path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(path))


@pytest.fixture
def stub_runner(monkeypatch, tmp_path):
    """An OnnxRunner over a real tokenizer with the ONNX session stubbed out."""
    import onnxruntime as ort

    monkeypatch.setenv("YN_ONNX_DIR", str(tmp_path))
    d = write_export(tmp_path / _slug("fake-model"))
    _tiny_tokenizer(d / TOKENIZER_FILE)
    monkeypatch.setattr(ort, "InferenceSession", lambda *a, **kw: object())
    return OnnxRunner("fake-model")


def test_runner_truncates_only_the_input_never_the_claim(stub_runner):
    """longest_first, the library default, would clip the claim into a different
    question and score that instead. This is the bug with no visible symptom."""
    assert stub_runner.tokenizer.truncation["strategy"] == "only_first"
    assert stub_runner.tokenizer.truncation["max_length"] == 512


def test_count_tokens_reports_the_real_size_not_the_cap(stub_runner):
    """tokenizer.json carries truncation, so a naive counter returns exactly 512
    for anything longer and the over-limit error names the wrong number."""
    long_statement = "word " * 600
    assert stub_runner.count_tokens(long_statement) == 600


# --- export() must not delete anything it did not write ----------------------------

def test_replace_export_refuses_a_directory_that_is_not_an_export(tmp_path):
    """--out takes any path, so this guards a user's own directory."""
    from yn.onnx_backend import _replace_export

    victim = tmp_path / "my-models"
    victim.mkdir()
    (victim / "important.bin").write_text("do not delete")
    tmp = tmp_path / "staged"
    tmp.mkdir()

    with pytest.raises(RuntimeError, match="not a yn export"):
        _replace_export(tmp, victim)
    assert (victim / "important.bin").read_text() == "do not delete"


def test_replace_export_replaces_a_previous_export(tmp_path):
    from yn.onnx_backend import _replace_export

    final = write_export(tmp_path / "export", entail_idx=0)
    tmp = tmp_path / "staged"
    write_export(tmp, entail_idx=2)

    _replace_export(tmp, final)
    assert onnx_backend.read_meta(final)["entail_idx"] == 2
    assert not tmp.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["export"]  # nothing left over


def test_replace_export_accepts_an_empty_directory(tmp_path):
    from yn.onnx_backend import _replace_export

    final = tmp_path / "empty"
    final.mkdir()
    tmp = tmp_path / "staged"
    write_export(tmp)
    _replace_export(tmp, final)
    assert (final / onnx_backend.META_FILE).is_file()


def _aged(d: Path, seconds: float) -> Path:
    import os
    import time

    t = time.time() - seconds
    os.utime(d, (t, t))
    return d


def test_replace_export_refuses_a_user_directory_that_has_a_meta_json(tmp_path):
    """meta.json is a common file name; it alone doesn't make a directory ours."""
    from yn.onnx_backend import _replace_export

    victim = tmp_path / "my-project"
    victim.mkdir()
    (victim / "meta.json").write_text('{"name": "my project"}')
    (victim / "precious.txt").write_text("do not delete")
    tmp = write_export(tmp_path / "staged")

    with pytest.raises(RuntimeError, match="not a yn export"):
        _replace_export(tmp, victim)
    assert (victim / "precious.txt").read_text() == "do not delete"


def test_replace_export_refuses_a_folder_holding_only_someone_elses_meta_json(tmp_path):
    """Every file name matches an export's, but the metadata isn't ours."""
    from yn.onnx_backend import _replace_export

    victim = tmp_path / "config"
    victim.mkdir()
    (victim / "meta.json").write_text('{"name": "my settings"}')
    tmp = write_export(tmp_path / "staged")

    with pytest.raises(RuntimeError, match="not a yn export"):
        _replace_export(tmp, victim)
    assert "my settings" in (victim / "meta.json").read_text()


def test_replace_export_refuses_an_export_with_a_user_file_added(tmp_path):
    """Our metadata, but a file we never write: replacing it would delete that file."""
    from yn.onnx_backend import _replace_export

    final = write_export(tmp_path / "export")
    (final / "notes.txt").write_text("mine")
    tmp = write_export(tmp_path / "staged")

    with pytest.raises(RuntimeError, match="not a yn export"):
        _replace_export(tmp, final)
    assert (final / "notes.txt").read_text() == "mine"


def test_replace_export_leaves_a_users_dot_old_sibling_alone(tmp_path):
    """The old export is set aside in a temp dir of our own, not at <out>.old."""
    from yn.onnx_backend import _replace_export

    sibling = tmp_path / "export.old"
    sibling.mkdir()
    (sibling / "user.txt").write_text("mine")
    final = write_export(tmp_path / "export")
    tmp = write_export(tmp_path / "staged")

    _replace_export(tmp, final)
    assert (sibling / "user.txt").read_text() == "mine"


def test_replace_export_puts_the_old_export_back_if_the_swap_fails(tmp_path, monkeypatch):
    import os

    from yn.onnx_backend import _replace_export

    final = write_export(tmp_path / "export", entail_idx=0)
    tmp = write_export(tmp_path / "staged", entail_idx=2)
    real_replace = os.replace

    def flaky(src, dst):
        if Path(src) == tmp:
            raise OSError("disk full")
        real_replace(src, dst)

    monkeypatch.setattr(onnx_backend.os, "replace", flaky)
    with pytest.raises(OSError):
        _replace_export(tmp, final)
    assert onnx_backend.read_meta(final)["entail_idx"] == 0


@pytest.fixture
def cache_root(monkeypatch, tmp_path):
    """YN's own export cache, where sweeping is allowed."""
    monkeypatch.setenv("YN_ONNX_DIR", str(tmp_path / "cache"))
    root = tmp_path / "cache"
    root.mkdir()
    return root


def test_sweep_removes_stale_temp_dirs(cache_root):
    """A killed export leaves most of a gigabyte behind under .export-*."""
    from yn.onnx_backend import _sweep_stale_temp_dirs

    stale = cache_root / ".export-abc123"
    stale.mkdir()
    (stale / "model.onnx").write_text("half written")
    _aged(stale, 2 * 3600)
    (cache_root / ".export-old-xyz").mkdir()
    old_swap = _aged(cache_root / ".export-old-xyz", 2 * 3600)
    keep = cache_root / "real-export"
    keep.mkdir()

    _sweep_stale_temp_dirs(cache_root)
    assert not stale.exists()
    assert not old_swap.exists()
    assert keep.exists()


def test_sweep_leaves_a_temp_dir_another_export_is_still_writing(cache_root):
    from yn.onnx_backend import _sweep_stale_temp_dirs

    running = cache_root / ".export-running"
    running.mkdir()
    _sweep_stale_temp_dirs(cache_root)
    assert running.exists()


def test_sweep_never_runs_outside_yns_own_cache(cache_root, tmp_path):
    """--out can sit in a user's home folder; their .export-* is not ours to delete."""
    from yn.onnx_backend import _sweep_stale_temp_dirs

    home = tmp_path / "home"
    theirs = home / ".export-settings"
    theirs.mkdir(parents=True)
    _aged(theirs, 2 * 3600)
    _sweep_stale_temp_dirs(home)
    assert theirs.exists()


def test_bad_onnx_threads_is_an_error_even_on_auto(monkeypatch, tmp_path):
    """A typo in config should be reported, not quietly turned into a torch fallback."""
    monkeypatch.setenv("YN_ONNX_DIR", str(tmp_path))
    write_export(tmp_path / _slug("fake-model"))
    monkeypatch.setenv("YN_ONNX_THREADS", "four")
    d = Decider(model_name="fake-model", backend="auto")
    with pytest.raises(ValueError, match="YN_ONNX_THREADS"):
        d._load_onnx_runner()


@pytest.mark.parametrize("raw", ["0", "-2"])
def test_onnx_threads_must_be_at_least_one(monkeypatch, raw):
    monkeypatch.setenv("YN_ONNX_THREADS", raw)
    with pytest.raises(ValueError, match="1 or more"):
        onnx_backend._env_threads()


def test_onnx_threads_reads_a_whole_number(monkeypatch):
    monkeypatch.setenv("YN_ONNX_THREADS", "3")
    assert onnx_backend._env_threads() == 3


@pytest.mark.parametrize("device", ["mps", "cuda"])
def test_auto_leaves_an_explicitly_chosen_gpu_on_torch(monkeypatch, tmp_path, device):
    """ONNX runs on the CPU here, so an export must not override YN_DEVICE=mps/cuda."""
    monkeypatch.setenv("YN_ONNX_DIR", str(tmp_path))
    write_export(tmp_path / _slug("fake-model"))
    assert not Decider(model_name="fake-model", backend="auto", device=device)._use_onnx()
    assert Decider(model_name="fake-model", backend="auto", device="cpu")._use_onnx()
    assert Decider(model_name="fake-model", backend="onnx", device=device)._use_onnx()
