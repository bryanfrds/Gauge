"""Unit tests for yn.model with a fake model (no weights loaded)."""

from __future__ import annotations

import pytest

from yn.model import DEFAULT_TEMPLATE, MAX_OPTIONS, Decider, Decision, _as_statement

CLAIM = "This email is spam."


# --- _as_statement ---------------------------------------------------------------

@pytest.mark.parametrize("label", ["billing", "P1", "refund request"])
def test_as_statement_wraps_bare_label_in_template(label):
    assert _as_statement(label, DEFAULT_TEMPLATE) == f"This text is about {label}."


@pytest.mark.parametrize(
    "statement",
    ["The customer wants a refund.", "Ship it now!", "Is this about billing?"],
)
def test_as_statement_keeps_full_statement_unchanged(statement):
    assert _as_statement(statement, DEFAULT_TEMPLATE) == statement


def test_as_statement_strips_whitespace_before_checking_punctuation():
    assert _as_statement("  The order is late.  ", DEFAULT_TEMPLATE) == "The order is late."


def test_as_statement_strips_label_before_templating():
    assert _as_statement("  billing ", DEFAULT_TEMPLATE) == "This text is about billing."


def test_as_statement_uses_custom_template():
    assert _as_statement("cats", "Topic: {}") == "Topic: cats"


# --- input validation --------------------------------------------------------------

@pytest.mark.parametrize("bad", ["", "   ", "\n\t"])
def test_check_rejects_empty_input(decider, bad):
    with pytest.raises(ValueError, match="input must be a non-empty string"):
        decider.check(bad, CLAIM)


@pytest.mark.parametrize("bad", ["", "   "])
def test_check_rejects_empty_claim(decider, bad):
    with pytest.raises(ValueError, match="claim must be a non-empty string"):
        decider.check("hello", bad)


def test_check_many_rejects_non_string_input(decider):
    with pytest.raises(ValueError, match="input must be a non-empty string"):
        decider.check_many(["ok", None], CLAIM)


def test_check_many_rejects_empty_claim_even_with_no_inputs(decider):
    with pytest.raises(ValueError, match="claim must be a non-empty string"):
        decider.check_many([], "")


def test_check_many_rejects_if_any_input_is_empty(decider, fake_model):
    with pytest.raises(ValueError, match="input must be a non-empty string"):
        decider.check_many(["fine", ""], CLAIM)
    assert fake_model.calls == []


@pytest.mark.parametrize("bad", ["", "  "])
def test_decide_rejects_empty_input(decider, bad):
    with pytest.raises(ValueError, match="input must be a non-empty string"):
        decider.decide(bad, ["a", "b"])


@pytest.mark.parametrize("options", [[], ["only"]])
def test_decide_rejects_fewer_than_two_options(decider, options):
    with pytest.raises(ValueError, match=f"options must have 2-50 items, got {len(options)}"):
        decider.decide("text", options)


def test_decide_accepts_exactly_two_options(decider):
    assert decider.decide("text", ["a", "b"]).answer in {"a", "b"}


def test_decide_accepts_max_options(decider):
    options = [f"opt{i}" for i in range(MAX_OPTIONS)]
    assert len(decider.decide("text", options).scores) == MAX_OPTIONS


def test_decide_rejects_more_than_max_options(decider):
    options = [f"opt{i}" for i in range(MAX_OPTIONS + 1)]
    with pytest.raises(ValueError, match="options must have 2-50 items, got 51"):
        decider.decide("text", options)


@pytest.mark.parametrize("bad", ["", "   "])
def test_decide_rejects_empty_option(decider, bad):
    with pytest.raises(ValueError, match="each option must be a non-empty string"):
        decider.decide("text", ["a", bad])


def test_decide_rejects_exact_duplicate_options(decider):
    with pytest.raises(ValueError, match="options must be unique"):
        decider.decide("text", ["billing", "shipping", "billing"])


@pytest.mark.parametrize("dup", [" billing", "billing ", "\tbilling\n"])
def test_decide_rejects_whitespace_duplicate_options(decider, dup):
    with pytest.raises(ValueError, match="options must be unique"):
        decider.decide("text", ["billing", dup])


def test_decide_options_differing_only_in_case_are_distinct(decider):
    assert set(decider.decide("text", ["Billing", "billing"]).scores) == {"Billing", "billing"}


def test_decide_many_validates_options_even_with_no_inputs(decider):
    with pytest.raises(ValueError, match="options must have"):
        decider.decide_many([], ["only"])


# --- empty input list --------------------------------------------------------------

def test_check_many_empty_list_returns_empty_without_model_call(decider, fake_model):
    assert decider.check_many([], CLAIM) == []
    assert fake_model.calls == []


def test_decide_many_empty_list_returns_empty_without_model_call(decider, fake_model):
    assert decider.decide_many([], ["a", "b"]) == []
    assert fake_model.calls == []


# --- check_many scoring ------------------------------------------------------------

def test_check_true_when_entailment_probability_high(decider, fake_model):
    fake_model.entail[("win a prize", CLAIM)] = fake_model.check_logit(0.9)
    d = decider.check("win a prize", CLAIM)
    assert d.answer == "true"
    assert d.confidence == 0.9
    assert d.scores == {"true": 0.9, "false": 0.1}


def test_check_false_when_entailment_probability_low(decider, fake_model):
    fake_model.entail[("meeting at 3", CLAIM)] = fake_model.check_logit(0.2)
    d = decider.check("meeting at 3", CLAIM)
    assert d.answer == "false"
    assert d.confidence == 0.8
    assert d.scores == {"true": 0.2, "false": 0.8}


def test_check_reads_entailment_column_not_column_zero(decider, fake_model):
    # Fake puts the high logit in column 1 (decider._entail_idx); column 0 stays 0.
    fake_model.entail[("x", CLAIM)] = 5.0
    assert decider.check("x", CLAIM).answer == "true"


def test_check_softmax_is_over_all_labels(decider, fake_model):
    # Logit 0 in all 3 columns => P(entailment) = 1/3, not 1/2.
    d = decider.check("neutral", CLAIM)
    assert d.scores == {"true": 0.3333, "false": 0.6667}
    assert d.answer == "false"


def test_check_true_and_false_scores_sum_to_one(decider, fake_model):
    fake_model.entail[("t", CLAIM)] = 1.2345
    d = decider.check("t", CLAIM)
    assert d.scores["true"] + d.scores["false"] == pytest.approx(1.0, abs=1e-4)


def test_check_scores_rounded_to_four_places(decider, fake_model):
    fake_model.entail[("t", CLAIM)] = 0.7
    d = decider.check("t", CLAIM)
    for v in (*d.scores.values(), d.confidence):
        assert v == round(v, 4)
    assert d.confidence == max(d.scores.values())


def test_check_sends_claim_verbatim_as_hypothesis(decider, fake_model):
    decider.check("some text", "It is urgent")  # no trailing period: must NOT be templated
    assert fake_model.calls == [[("some text", "It is urgent")]]


def test_check_many_preserves_input_order(decider, fake_model):
    texts = ["spam1", "ham", "spam2"]
    fake_model.entail[("spam1", CLAIM)] = fake_model.check_logit(0.95)
    fake_model.entail[("ham", CLAIM)] = fake_model.check_logit(0.05)
    fake_model.entail[("spam2", CLAIM)] = fake_model.check_logit(0.6)
    results = decider.check_many(texts, CLAIM)
    assert [r.answer for r in results] == ["true", "false", "true"]
    assert [r.scores["true"] for r in results] == [0.95, 0.05, 0.6]


def test_check_many_single_model_call_for_all_inputs(decider, fake_model):
    decider.check_many(["a", "b", "c"], CLAIM)
    assert fake_model.calls == [[("a", CLAIM), ("b", CLAIM), ("c", CLAIM)]]


def test_check_result_includes_model_name(decider):
    assert decider.check("t", CLAIM).model == "fake-model"


def test_decision_to_dict_has_all_fields(decider, fake_model):
    fake_model.entail[("t", CLAIM)] = fake_model.check_logit(0.9)
    assert decider.check("t", CLAIM).to_dict() == {
        "answer": "true",
        "confidence": 0.9,
        "scores": {"true": 0.9, "false": 0.1},
        "sure": True,
        "model": "fake-model",
    }


# --- sure vs threshold ---------------------------------------------------------------

def test_sure_when_confidence_above_threshold(decider, fake_model):
    decider.threshold = 0.85
    fake_model.entail[("t", CLAIM)] = fake_model.check_logit(0.9)
    assert decider.check("t", CLAIM).sure is True


def test_not_sure_when_confidence_below_threshold(decider, fake_model):
    decider.threshold = 0.85
    fake_model.entail[("t", CLAIM)] = fake_model.check_logit(0.8)
    assert decider.check("t", CLAIM).sure is False


def test_sure_when_confidence_exactly_at_threshold(decider):
    # Equal logits over 2 options => softmax is exactly 0.5.
    decider.threshold = 0.5
    d = decider.decide("t", ["a", "b"])
    assert d.confidence == 0.5
    assert d.sure is True


def test_not_sure_just_above_exact_confidence(decider):
    decider.threshold = 0.5000001
    assert decider.decide("t", ["a", "b"]).sure is False


def test_threshold_from_env(monkeypatch, fake_model):
    monkeypatch.setenv("YN_THRESHOLD", "0.3")
    assert Decider().threshold == 0.3


def test_explicit_threshold_overrides_env(monkeypatch, fake_model):
    monkeypatch.setenv("YN_THRESHOLD", "0.3")
    assert Decider(threshold=0.7).threshold == 0.7


def test_explicit_zero_threshold_is_not_replaced_by_default(fake_model):
    assert Decider(threshold=0.0).threshold == 0.0


def test_default_threshold(fake_model):
    assert Decider().threshold == 0.85


# --- decide_many scoring -------------------------------------------------------------

def _set_weights(fake_model, text, statement_weights):
    for statement, w in statement_weights.items():
        fake_model.entail[(text, statement)] = fake_model.decide_logit(w)


def test_decide_picks_highest_option_with_softmax_across_options(decider, fake_model):
    _set_weights(fake_model, "where is my parcel", {
        "This text is about billing.": 1,
        "This text is about shipping.": 6,
        "This text is about technical.": 3,
    })
    d = decider.decide("where is my parcel", ["billing", "shipping", "technical"])
    assert d.answer == "shipping"
    assert d.confidence == 0.6
    assert d.scores == {"billing": 0.1, "shipping": 0.6, "technical": 0.3}


def test_decide_scores_sum_to_one(decider, fake_model):
    _set_weights(fake_model, "t", {"This text is about a.": 2.5, "This text is about b.": 0.7})
    d = decider.decide("t", ["a", "b", "c", "d"])
    assert sum(d.scores.values()) == pytest.approx(1.0, abs=1e-3)


def test_decide_scores_keyed_by_original_option_strings(decider, fake_model):
    options = ["  billing ", "The parcel is late."]
    _set_weights(fake_model, "t", {"This text is about billing.": 1, "The parcel is late.": 3})
    d = decider.decide("t", options)
    assert list(d.scores) == options
    assert d.answer == "The parcel is late."
    assert d.scores == {"  billing ": 0.25, "The parcel is late.": 0.75}


def test_decide_answer_is_original_unstripped_option(decider, fake_model):
    _set_weights(fake_model, "t", {"This text is about billing.": 9, "This text is about other.": 1})
    assert decider.decide("t", [" billing ", "other"]).answer == " billing "


def test_decide_sends_templated_labels_and_raw_statements_to_model(decider, fake_model):
    decider.decide("txt", ["billing", "The customer wants a refund."])
    assert fake_model.calls == [[
        ("txt", "This text is about billing."),
        ("txt", "The customer wants a refund."),
    ]]


def test_decide_uses_custom_template(fake_model):
    d = Decider(model_name="fake", threshold=0.5, template="Category: {}")
    d.decide("txt", ["a", "b"])
    assert fake_model.calls == [[("txt", "Category: a"), ("txt", "Category: b")]]


def test_decide_many_results_per_input_in_order(decider, fake_model):
    _set_weights(fake_model, "first", {"This text is about a.": 8, "This text is about b.": 2})
    _set_weights(fake_model, "second", {"This text is about a.": 1, "This text is about b.": 3})
    _set_weights(fake_model, "third", {"This text is about a.": 1, "This text is about b.": 1})
    results = decider.decide_many(["first", "second", "third"], ["a", "b"])
    assert [r.answer for r in results] == ["a", "b", "a"]
    assert [r.scores for r in results] == [
        {"a": 0.8, "b": 0.2},
        {"a": 0.25, "b": 0.75},
        {"a": 0.5, "b": 0.5},
    ]


def test_decide_many_softmax_is_per_input_not_across_inputs(decider, fake_model):
    # A huge logit on one text must not drain probability from the other.
    _set_weights(fake_model, "loud", {"This text is about a.": 1e6, "This text is about b.": 1})
    _set_weights(fake_model, "quiet", {"This text is about a.": 1, "This text is about b.": 4})
    loud, quiet = decider.decide_many(["loud", "quiet"], ["a", "b"])
    assert quiet.scores == {"a": 0.2, "b": 0.8}


def test_decide_result_same_when_options_shuffled(decider, fake_model):
    _set_weights(fake_model, "t", {
        "This text is about a.": 5, "This text is about b.": 3, "This text is about c.": 2,
    })
    one = decider.decide("t", ["a", "b", "c"])
    two = decider.decide("t", ["c", "a", "b"])
    assert one.answer == two.answer == "a"
    assert one.scores == two.scores


def test_decide_rounds_scores_to_four_places(decider, fake_model):
    d = decider.decide("t", ["a", "b", "c"])  # all equal => 1/3 each
    assert d.scores == {"a": 0.3333, "b": 0.3333, "c": 0.3333}
    assert d.confidence == 0.3333


def test_decide_tie_picks_first_option(decider):
    assert decider.decide("t", ["x", "y"]).answer == "x"


# --- construction ------------------------------------------------------------------

def test_model_name_from_env(monkeypatch, fake_model):
    monkeypatch.setenv("YN_MODEL", "some/other-model")
    assert Decider().model_name == "some/other-model"


def test_decision_is_dataclass_roundtrip():
    d = Decision("true", 0.9, {"true": 0.9, "false": 0.1}, True, "m")
    assert Decision(**d.to_dict()) == d


def test_sure_uses_rounded_confidence(decider):
    # 0.84996 rounds to 0.85, so it must count as sure at threshold 0.85.
    decider.threshold = 0.85
    r = decider._decision({"true": 0.84996, "false": 0.15004})
    assert r.confidence == 0.85 and r.sure is True
    r = decider._decision({"true": 0.8496, "false": 0.1504})
    assert r.confidence == 0.8496 and r.sure is False


def test_bad_threshold_env_raises_runtime_error(monkeypatch):
    monkeypatch.setenv("YN_THRESHOLD", "abc")
    with pytest.raises(RuntimeError, match="YN_THRESHOLD"):
        Decider()


# --- size limits -----------------------------------------------------------------------

def test_too_many_inputs_rejected(decider, fake_model):
    from yn.model import MAX_INPUTS, InputError
    with pytest.raises(InputError, match="at most"):
        decider.check_many(["x"] * (MAX_INPUTS + 1), "This is spam.")
    assert fake_model.calls == []


def test_too_many_pairs_rejected(decider, fake_model):
    from yn.model import MAX_PAIRS, InputError
    options = [f"option {i}" for i in range(50)]
    with pytest.raises(InputError, match="Split the inputs"):
        decider.decide_many(["x"] * (MAX_PAIRS // 50 + 1), options)
    assert fake_model.calls == []


def test_overlong_input_and_claim_rejected(decider):
    from yn.model import MAX_INPUT_CHARS, MAX_STATEMENT_CHARS, InputError
    with pytest.raises(InputError, match="input is"):
        decider.check("x" * (MAX_INPUT_CHARS + 1), "This is spam.")
    with pytest.raises(InputError, match="claim is"):
        decider.check("hello", "x" * (MAX_STATEMENT_CHARS + 1))
    with pytest.raises(InputError, match="each option is"):
        decider.decide("hello", ["a", "x" * (MAX_STATEMENT_CHARS + 1)])


def test_inference_never_overlaps(monkeypatch):
    """Regression: parallel MCP calls ran the model concurrently and crashed the Apple GPU.

    Exercises the real `_logits` with a fake tokenizer and a slow fake model.
    """
    import threading
    import time
    from types import SimpleNamespace

    import torch

    active = peak = 0
    guard = threading.Lock()

    class Batch(dict):
        def to(self, device):
            return self

    def tokenizer(texts, statements=None, **kw):
        if statements is None:  # token count for one statement
            return {"input_ids": [0] * 5}
        return Batch(n=len(texts))

    def model(n):
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with guard:
            active -= 1
        return SimpleNamespace(logits=torch.zeros(n, 2))

    d = Decider(model_name="fake-model", device="cpu")
    d._tokenizer, d._model = tokenizer, model
    monkeypatch.setattr(Decider, "load", lambda self: None)

    threads = [threading.Thread(target=d.check_many, args=(["a", "b"], "This is spam."))
               for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak == 1


def test_statement_over_token_limit_rejected(monkeypatch):
    """Short in characters but long in tokens (e.g. Chinese) must be a clear InputError."""
    from yn.model import MAX_STATEMENT_TOKENS, InputError

    calls = []

    def tokenizer(text, statements=None, **kw):
        assert statements is None, "model must not run for an over-long claim"
        calls.append(text)
        return {"input_ids": [0] * (MAX_STATEMENT_TOKENS + 1)}

    d = Decider(model_name="fake-model", device="cpu")
    d._tokenizer, d._model = tokenizer, None
    monkeypatch.setattr(Decider, "load", lambda self: None)
    with pytest.raises(InputError, match="tokens; the limit is"):
        d.check("hello", "这是一个很长的说法。")
    assert calls == ["这是一个很长的说法。"]


# --- backend selection -------------------------------------------------------------

def test_default_backend_is_auto(fake_model):
    from yn.model import DEFAULT_BACKEND
    assert Decider().backend == DEFAULT_BACKEND == "auto"


@pytest.mark.parametrize("value", ["auto", "torch", "onnx"])
def test_backend_from_env(monkeypatch, fake_model, value):
    monkeypatch.setenv("YN_BACKEND", value)
    assert Decider().backend == value


def test_explicit_backend_overrides_env(monkeypatch, fake_model):
    monkeypatch.setenv("YN_BACKEND", "onnx")
    assert Decider(backend="torch").backend == "torch"


@pytest.mark.parametrize("bad", ["ONNX", "tensorflow", " torch", "torch "])
def test_bad_backend_raises_runtime_error_at_construction(bad):
    with pytest.raises(RuntimeError, match="YN_BACKEND must be auto, torch or onnx"):
        Decider(backend=bad)


def test_empty_backend_falls_back_to_the_default(fake_model):
    assert Decider(backend="").backend == "auto"


def test_bad_backend_env_raises_runtime_error(monkeypatch):
    monkeypatch.setenv("YN_BACKEND", "tensorflow")
    with pytest.raises(RuntimeError, match="YN_BACKEND"):
        Decider()


def test_backend_torch_never_looks_for_an_export(monkeypatch):
    from yn import onnx_backend
    monkeypatch.setattr(onnx_backend, "is_exported",
                        lambda name: pytest.fail("torch backend must not check for an export"))
    assert Decider(backend="torch")._use_onnx() is False


def test_backend_onnx_is_used_even_with_no_export(monkeypatch):
    from yn import onnx_backend
    monkeypatch.setattr(onnx_backend, "is_exported", lambda name: False)
    assert Decider(backend="onnx")._use_onnx() is True


@pytest.mark.parametrize("exported", [True, False])
def test_backend_auto_follows_is_exported(monkeypatch, exported):
    from yn import onnx_backend
    seen = []

    def is_exported(name):
        seen.append(name)
        return exported

    monkeypatch.setattr(onnx_backend, "is_exported", is_exported)
    assert Decider(model_name="fake-model", backend="auto")._use_onnx() is exported
    assert seen == ["fake-model"]


# --- logits shape ------------------------------------------------------------------

def test_torch_logits_returns_numpy(monkeypatch):
    """Both backends must hand `_logits` callers a numpy array, not a tensor."""
    from types import SimpleNamespace

    import numpy as np
    import torch

    class Batch(dict):
        def to(self, device):
            return self

    def tokenizer(texts, statements=None, **kw):
        if statements is None:
            return {"input_ids": [0] * 5}
        return Batch(n=len(texts))

    def model(n):
        return SimpleNamespace(logits=torch.zeros(n, 3, dtype=torch.float64))

    d = Decider(model_name="fake-model", device="cpu", backend="torch")
    d._tokenizer, d._model = tokenizer, model
    monkeypatch.setattr(Decider, "load", lambda self: None)

    out = d._logits([("a", "This is spam."), ("b", "This is spam.")])
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    assert out.shape == (2, 3)


def test_decide_many_reshapes_to_texts_by_options(decider, fake_model):
    texts, options = ["t1", "t2", "t3"], ["a", "b", "c", "d"]
    results = decider.decide_many(texts, options)
    assert len(results) == len(texts)
    assert [sorted(r.scores) for r in results] == [sorted(options)] * len(texts)
    # One model call, pairs in text-major order, so the reshape lines up.
    assert fake_model.calls == [[(t, _as_statement(o, DEFAULT_TEMPLATE))
                                for t in texts for o in options]]


def test_decide_many_reshape_maps_each_row_to_its_own_text(decider, fake_model):
    """A transposed reshape would still give the right shape, so check the values."""
    for text, winner in (("t1", "b"), ("t2", "c"), ("t3", "a")):
        for option in ("a", "b", "c"):
            statement = _as_statement(option, DEFAULT_TEMPLATE)
            weight = 8.0 if option == winner else 1.0
            fake_model.entail[(text, statement)] = fake_model.decide_logit(weight)
    results = decider.decide_many(["t1", "t2", "t3"], ["a", "b", "c"])
    assert [r.answer for r in results] == ["b", "c", "a"]
    assert [r.confidence for r in results] == [0.8] * 3
