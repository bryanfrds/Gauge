"""Unit tests for the `yn` CLI with a fake model (no weights loaded)."""

from __future__ import annotations

import io
import json

import pytest

from yn.cli import EXIT_ERROR, EXIT_FALSE, EXIT_UNSURE, EXIT_USAGE, _parser, main
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
    assert e.value.code == EXIT_USAGE  # not 2, which means "unsure"


def test_parse_requires_subcommand():
    with pytest.raises(SystemExit) as e:
        _parser().parse_args([])
    assert e.value.code == EXIT_USAGE


def test_parse_rejects_non_numeric_threshold():
    with pytest.raises(SystemExit) as e:
        _parser().parse_args(["check", CLAIM, "x", "--threshold", "high"])
    assert e.value.code == EXIT_USAGE


@pytest.mark.parametrize("bad", ["5", "-0.1", "1.01"])
def test_parse_rejects_out_of_range_threshold(bad):
    with pytest.raises(SystemExit) as e:
        _parser().parse_args(["check", CLAIM, "x", "--threshold", bad])
    assert e.value.code == EXIT_USAGE


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


def test_lines_with_only_blank_lines_is_usage_error(fake_model, stdin, capsys):
    # Exiting 0 would read as "true" to a hook.
    stdin("\n  \n")
    assert main(["check", CLAIM, "--lines", "--exit-code"]) == EXIT_USAGE
    out, err = capsys.readouterr()
    assert out == "" and "no non-blank lines" in err
    assert fake_model.calls == []


def test_no_input_on_tty_exits_with_message(fake_model, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", TTY())
    assert main(["check", CLAIM]) == EXIT_USAGE
    assert "no input" in capsys.readouterr().err


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
    assert main(["check", "This is spam.", "hello"]) == EXIT_ERROR
    err = capsys.readouterr().err
    assert "YN_THRESHOLD must be a number from 0 to 1" in err
    assert "Traceback" not in err


def test_unexpected_error_is_exit_3_not_false(fake_model, monkeypatch, capsys):
    # A crash must never look like "false" (exit 1) to a hook.
    def boom(self, texts, claim):
        raise OSError("model files missing")
    monkeypatch.setattr("yn.model.Decider.check_many", boom)
    assert main(["check", CLAIM, "hello", "--exit-code"]) == EXIT_ERROR
    assert "OSError: model files missing" in capsys.readouterr().err


def test_broken_pipe_exits_quietly_not_false(check_p, monkeypatch):
    # `yn ... | head -1`: the reader closes early. Must not exit 1 ("false").
    import os
    import sys

    read_fd, write_fd = os.pipe()

    class ClosedPipe:
        def write(self, s):
            raise BrokenPipeError

        def flush(self):
            raise BrokenPipeError

        def fileno(self):
            return write_fd

    monkeypatch.setattr(sys, "stdout", ClosedPipe())
    redirected = []
    monkeypatch.setattr(os, "dup2", lambda fd, fd2: redirected.append(fd2))
    try:
        assert main(["check", CLAIM, "x", "--exit-code"]) == EXIT_ERROR
        assert redirected == [write_fd]
    finally:
        os.close(read_fd)
        os.close(write_fd)


# --- export-onnx ---------------------------------------------------------------------

def test_parse_export_onnx_defaults():
    a = _parser().parse_args(["export-onnx"])
    assert (a.cmd, a.model, a.out) == ("export-onnx", None, None)


def test_parse_export_onnx_with_model_and_out():
    a = _parser().parse_args(["export-onnx", "--model", "some/model", "--out", "/tmp/onnx"])
    assert (a.cmd, a.model, a.out) == ("export-onnx", "some/model", "/tmp/onnx")


@pytest.mark.parametrize("flag, field", [("--model", "model"), ("--out", "out")])
def test_parse_export_onnx_accepts_either_flag_alone(flag, field):
    a = _parser().parse_args(["export-onnx", flag, "value"])
    assert getattr(a, field) == "value"
    other = "out" if field == "model" else "model"
    assert getattr(a, other) is None


def test_parse_export_onnx_takes_no_positional_text():
    with pytest.raises(SystemExit) as e:
        _parser().parse_args(["export-onnx", "some/model"])
    assert e.value.code == EXIT_USAGE


@pytest.fixture
def fake_export(monkeypatch, tmp_path):
    """Record calls to onnx_backend.export instead of exporting anything."""
    from yn import onnx_backend

    calls = []

    def export(model, out_dir=None):
        calls.append((model, out_dir))
        return tmp_path / "exported"

    monkeypatch.setattr(onnx_backend, "export", export)
    return calls


def test_export_onnx_uses_default_model(fake_export, capsys):
    assert main(["export-onnx"]) == 0
    assert fake_export == [(DEFAULT_MODEL, None)]
    assert "exported" in capsys.readouterr().err


def test_export_onnx_model_flag_wins_over_env(monkeypatch, fake_export):
    monkeypatch.setenv("YN_MODEL", "from/env")
    assert main(["export-onnx", "--model", "from/flag"]) == 0
    assert fake_export == [("from/flag", None)]


def test_export_onnx_falls_back_to_env_model(monkeypatch, fake_export):
    monkeypatch.setenv("YN_MODEL", "from/env")
    assert main(["export-onnx"]) == 0
    assert fake_export == [("from/env", None)]


def test_export_onnx_passes_out_directory(fake_export, tmp_path):
    assert main(["export-onnx", "--out", str(tmp_path / "dest")]) == 0
    assert fake_export == [(DEFAULT_MODEL, str(tmp_path / "dest"))]


def test_export_onnx_prints_nothing_on_stdout(fake_export, capsys):
    main(["export-onnx"])
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "exported" in captured.err


def test_export_onnx_promises_auto_pickup_only_for_the_default_location(
    monkeypatch, tmp_path, capsys
):
    """The export really is where YN will look, so the auto-pickup line is true."""
    from yn import onnx_backend

    monkeypatch.setenv("YN_ONNX_DIR", str(tmp_path))
    monkeypatch.setattr(onnx_backend, "export",
                        lambda model, out_dir=None: onnx_backend.export_dir(model))
    main(["export-onnx"])
    assert "YN_BACKEND=torch" in capsys.readouterr().err


def test_export_onnx_tells_you_how_to_use_an_out_directory(
    monkeypatch, tmp_path, capsys
):
    """--out puts it somewhere YN won't look; the advice must actually lead there."""
    import re

    from yn import onnx_backend
    from yn.model import DEFAULT_MODEL

    out = tmp_path / "elsewhere" / onnx_backend._slug(DEFAULT_MODEL)
    monkeypatch.setattr(onnx_backend, "export", lambda model, out_dir=None: out)
    main(["export-onnx", "--out", str(out)])
    err = capsys.readouterr().err
    assert "used automatically" not in err
    value = re.search(r"YN_ONNX_DIR=(\S+)", err).group(1)
    monkeypatch.setenv("YN_ONNX_DIR", value)
    assert onnx_backend.export_dir(DEFAULT_MODEL) == out


def test_export_onnx_says_a_wrongly_named_out_directory_must_be_renamed(
    monkeypatch, tmp_path, capsys
):
    """YN looks in YN_ONNX_DIR/<slug>, so no setting reaches a folder named "m"."""
    from yn import onnx_backend
    from yn.model import DEFAULT_MODEL

    out = tmp_path / "elsewhere" / "m"
    monkeypatch.setattr(onnx_backend, "export", lambda model, out_dir=None: out)
    main(["export-onnx", "--out", str(out)])
    err = capsys.readouterr().err
    assert "YN_ONNX_DIR=" not in err
    assert str(out.parent / onnx_backend._slug(DEFAULT_MODEL)) in err


def test_export_onnx_tells_you_to_set_yn_model_for_another_model(
    monkeypatch, tmp_path, capsys
):
    from yn import onnx_backend

    monkeypatch.setenv("YN_ONNX_DIR", str(tmp_path))
    monkeypatch.setattr(onnx_backend, "export",
                        lambda model, out_dir=None: onnx_backend.export_dir(model))
    main(["export-onnx", "--model", "other/thing"])
    err = capsys.readouterr().err
    assert "YN_MODEL=other/thing" in err
    assert "used automatically" not in err


def test_export_onnx_does_not_read_stdin(fake_export, monkeypatch):
    monkeypatch.setattr("sys.stdin", TTY(""))  # reading it would raise "no input"
    assert main(["export-onnx"]) == 0


def test_export_onnx_missing_extra_is_a_clear_setup_error(monkeypatch, capsys):
    from yn import onnx_backend

    def export(model, out_dir=None):
        raise ImportError("No module named 'onnxscript'")

    monkeypatch.setattr(onnx_backend, "export", export)
    assert main(["export-onnx"]) == EXIT_ERROR
    err = capsys.readouterr().err
    assert "pip install 'yn[export]'" in err
    assert "onnxscript" in err
