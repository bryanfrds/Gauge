"""Unit tests for the `yn` CLI with a fake model (no weights loaded)."""

from __future__ import annotations

import io
import json

import pytest

from yn.cli import EXIT_FALSE, EXIT_UNSURE, EXIT_USAGE, _parser, main
from yn.model import DEFAULT_MODEL

CLAIM = "This email is spam."


class TTY(io.StringIO):
    def isatty(self):
        return True


@pytest.fixture
def stdin(monkeypatch):
    def set_stdin(text: str):
        monkeypatch.setattr("sys.stdin", io.StringIO(text))
    return set_stdin


@pytest.fixture
def check_p(fake_model):
    """Set P(true) for (text, CLAIM)."""
    def set_p(text: str, p: float, claim: str = CLAIM):
        fake_model.entail[(text, claim)] = fake_model.check_logit(p)
    return set_p


def out_lines(capsys) -> list[str]:
    return capsys.readouterr().out.splitlines()


# --- argument parsing ----------------------------------------------------------------

def test_parse_check_args():
    a = _parser().parse_args(["check", CLAIM, "hello", "--json", "--lines",
                              "--threshold", "0.6", "--exit-code"])
    assert (a.cmd, a.claim, a.text, a.json, a.lines, a.threshold, a.exit_code) == (
        "check", CLAIM, "hello", True, True, 0.6, True)


def test_parse_check_defaults():
    a = _parser().parse_args(["check", CLAIM])
    assert (a.text, a.json, a.lines, a.threshold, a.exit_code) == (None, False, False, None, False)


def test_parse_decide_repeated_options_in_order():
    a = _parser().parse_args(["decide", "-o", "billing", "--option", "shipping", "-o", "tech", "txt"])
    assert a.options == ["billing", "shipping", "tech"]
    assert a.text == "txt"


def test_parse_decide_requires_option(capsys):
    with pytest.raises(SystemExit) as e:
        _parser().parse_args(["decide", "txt"])
    assert e.value.code == 2


def test_parse_requires_subcommand():
    with pytest.raises(SystemExit) as e:
        _parser().parse_args([])
    assert e.value.code == 2


def test_parse_rejects_non_numeric_threshold():
    with pytest.raises(SystemExit) as e:
        _parser().parse_args(["check", CLAIM, "x", "--threshold", "high"])
    assert e.value.code == 2


# --- input sources -------------------------------------------------------------------

def test_text_argument_is_used(fake_model, capsys):
    main(["check", CLAIM, "hello there"])
    assert fake_model.calls == [[("hello there", CLAIM)]]


def test_text_from_stdin_when_no_argument(fake_model, stdin, capsys):
    stdin("piped text\nsecond line\n")
    main(["check", CLAIM])
    assert fake_model.calls == [[("piped text\nsecond line\n", CLAIM)]]


def test_argument_wins_over_stdin(fake_model, stdin, capsys):
    stdin("from stdin")
    main(["check", CLAIM, "from arg"])
    assert fake_model.calls == [[("from arg", CLAIM)]]


def test_lines_splits_and_drops_blank_lines(fake_model, stdin, capsys):
    stdin("one\n\n   \ntwo\n\t\nthree")
    main(["check", CLAIM, "--lines"])
    assert fake_model.calls == [[("one", CLAIM), ("two", CLAIM), ("three", CLAIM)]]
    assert len(out_lines(capsys)) == 3


def test_lines_applies_to_text_argument(fake_model, capsys):
    main(["decide", "-o", "a", "-o", "b", "--lines", "x\ny"])
    texts = [t for t, _ in fake_model.calls[0]]
    assert texts == ["x", "x", "y", "y"]


def test_lines_with_only_blank_lines_prints_nothing_and_exits_zero(fake_model, stdin, capsys):
    stdin("\n  \n")
    assert main(["check", CLAIM, "--lines", "--exit-code"]) == 0
    assert capsys.readouterr().out == ""
    assert fake_model.calls == []


def test_no_input_on_tty_exits_with_message(fake_model, monkeypatch):
    monkeypatch.setattr("sys.stdin", TTY())
    with pytest.raises(SystemExit) as e:
        main(["check", CLAIM])
    assert "no input" in str(e.value.code)


# --- output format -------------------------------------------------------------------

def test_plain_output_sure(check_p, capsys):
    check_p("t", 0.9)
    main(["check", CLAIM, "t"])
    assert capsys.readouterr().out == "true\t0.90\n"


def test_plain_output_unsure_has_third_column(check_p, capsys):
    check_p("t", 0.3)
    main(["check", CLAIM, "t"])
    assert capsys.readouterr().out == "false\t0.70\tunsure\n"


def test_plain_output_decide(fake_model, capsys):
    fake_model.entail[("t", "This text is about b.")] = fake_model.decide_logit(9)
    fake_model.entail[("t", "This text is about a.")] = fake_model.decide_logit(1)
    main(["decide", "-o", "a", "-o", "b", "t"])
    assert capsys.readouterr().out == "b\t0.90\n"


def test_threshold_flag_changes_sure(check_p, capsys):
    check_p("t", 0.7)
    main(["check", CLAIM, "t", "--threshold", "0.6"])
    assert capsys.readouterr().out == "true\t0.70\n"


def test_json_output_one_object_per_input(check_p, stdin, capsys):
    check_p("spam", 0.9)
    check_p("ham", 0.1)
    stdin("spam\nham\n")
    main(["check", CLAIM, "--lines", "--json"])
    rows = [json.loads(line) for line in out_lines(capsys)]
    assert rows == [
        {"answer": "true", "confidence": 0.9, "scores": {"true": 0.9, "false": 0.1},
         "sure": True, "model": DEFAULT_MODEL},
        {"answer": "false", "confidence": 0.9, "scores": {"true": 0.1, "false": 0.9},
         "sure": True, "model": DEFAULT_MODEL},
    ]


def test_json_output_model_name(check_p, monkeypatch, capsys):
    monkeypatch.setenv("YN_MODEL", "org/test-model")
    check_p("t", 0.9)
    main(["check", CLAIM, "t", "--json"])
    assert json.loads(capsys.readouterr().out)["model"] == "org/test-model"


# --- exit codes ----------------------------------------------------------------------

def test_without_exit_code_flag_always_zero(check_p, capsys):
    check_p("t", 0.1)  # false and sure
    assert main(["check", CLAIM, "t"]) == 0


def test_without_exit_code_flag_unsure_still_zero(check_p, capsys):
    check_p("t", 0.5)
    assert main(["check", CLAIM, "t"]) == 0


def test_check_exit_code_true_is_0(check_p, capsys):
    check_p("t", 0.95)
    assert main(["check", CLAIM, "t", "--exit-code"]) == 0


def test_check_exit_code_false_is_1(check_p, capsys):
    check_p("t", 0.05)
    assert main(["check", CLAIM, "t", "--exit-code"]) == EXIT_FALSE == 1


def test_check_exit_code_unsure_is_2(check_p, capsys):
    check_p("t", 0.6)
    assert main(["check", CLAIM, "t", "--exit-code"]) == EXIT_UNSURE == 2


def test_check_exit_code_any_false_among_lines_is_1(check_p, capsys):
    check_p("a", 0.95)
    check_p("b", 0.05)
    assert main(["check", CLAIM, "a\nb", "--lines", "--exit-code"]) == 1


def test_check_exit_code_unsure_beats_false(check_p, capsys):
    check_p("a", 0.05)  # sure false
    check_p("b", 0.6)   # unsure
    assert main(["check", CLAIM, "a\nb", "--lines", "--exit-code"]) == 2


def test_decide_exit_code_sure_is_0(fake_model, capsys):
    fake_model.entail[("t", "This text is about a.")] = fake_model.decide_logit(99)
    assert main(["decide", "-o", "a", "-o", "b", "t", "--exit-code"]) == 0


def test_decide_exit_code_unsure_is_2(fake_model, capsys):
    assert main(["decide", "-o", "a", "-o", "b", "t", "--exit-code"]) == 2


def test_decide_answer_named_false_is_not_exit_1(fake_model, capsys):
    # The "false" -> 1 rule is for `check` only.
    fake_model.entail[("t", "This text is about false.")] = fake_model.decide_logit(99)
    assert main(["decide", "-o", "true", "-o", "false", "t", "--exit-code"]) == 0


# --- usage errors --------------------------------------------------------------------

def test_empty_claim_exits_64_with_message(fake_model, capsys):
    assert main(["check", "   ", "text"]) == EXIT_USAGE == 64
    err = capsys.readouterr()
    assert err.err == "yn: claim must be a non-empty string\n"
    assert err.out == ""


def test_empty_text_exits_64(fake_model, capsys):
    assert main(["check", CLAIM, ""]) == 64
    assert "input must be a non-empty string" in capsys.readouterr().err


def test_empty_stdin_exits_64(fake_model, stdin, capsys):
    stdin("")
    assert main(["check", CLAIM]) == 64
    assert "input must be a non-empty string" in capsys.readouterr().err


def test_single_option_exits_64(fake_model, capsys):
    assert main(["decide", "-o", "only", "t"]) == 64
    assert capsys.readouterr().err == "yn: options must have 2-50 items, got 1\n"


def test_duplicate_options_exits_64(fake_model, capsys):
    assert main(["decide", "-o", "a", "-o", "a ", "t"]) == 64
    assert capsys.readouterr().err == "yn: options must be unique\n"


def test_usage_error_wins_over_exit_code_flag(fake_model, capsys):
    assert main(["decide", "-o", "a", "t", "--exit-code"]) == 64


# --- confidence display and bad config --------------------------------------------------

@pytest.mark.parametrize("conf, shown", [
    (0.8496, "0.84"), (0.85, "0.85"), (0.29, "0.29"), (0.999, "0.99"), (1.0, "1.00"), (0.0, "0.00"),
])
def test_two_places_rounds_down(conf, shown):
    from yn.cli import _two_places
    assert _two_places(conf) == shown


@pytest.mark.parametrize("bad", ["abc", "1.5", "-0.1"])
def test_bad_threshold_env_is_clear_error_not_traceback(fake_model, monkeypatch, capsys, bad):
    monkeypatch.setenv("YN_THRESHOLD", bad)
    assert main(["check", "This is spam.", "hello"]) == EXIT_USAGE
    err = capsys.readouterr().err
    assert "YN_THRESHOLD must be a number from 0 to 1" in err
    assert "Traceback" not in err
