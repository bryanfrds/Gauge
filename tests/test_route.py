"""Unit tests for yn/route.py with the fake model."""

from __future__ import annotations

import json

import pytest

from yn.cli import EXIT_ERROR, EXIT_UNSURE, EXIT_USAGE, main
from yn.model import InputError
from yn.route import DEFAULT_ROUTES, check_routes, default_routes, read_routes_file, route_many

TASK = "Fix the typo in the README title"
ROUTES = [{"model": "small", "when": "A quick edit"},
          {"model": "big", "when": "A hard project."}]


def set_weights(fake_model, task, weights: dict[str, float], routes=ROUTES):
    """Give each route's statement a softmax weight for `task`."""
    for (model, when) in check_routes(routes):
        fake_model.entail[(task, when)] = fake_model.decide_logit(weights[model])


# --- check_routes -------------------------------------------------------------------

def test_default_routes_are_valid_and_cheapest_first():
    pairs = check_routes(DEFAULT_ROUTES)
    assert [m for m, _ in pairs] == [
        "claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5-5", "claude-fable-5-1"]


def test_when_gets_a_period_so_it_is_a_statement():
    assert check_routes(ROUTES)[0] == ("small", "A quick edit.")
    assert check_routes(ROUTES)[1] == ("big", "A hard project.")


@pytest.mark.parametrize("routes, msg", [
    ("nope", "list of 2-50"),
    ([ROUTES[0]], "list of 2-50"),
    ([{"model": "a"}, ROUTES[1]], 'exactly "model" and "when"'),
    ([{"model": "a", "when": "x", "extra": 1}, ROUTES[1]], 'exactly "model" and "when"'),
    ([{"model": "", "when": "x"}, ROUTES[1]], "non-empty"),
    ([{"model": "a", "when": 3}, ROUTES[1]], "non-empty"),
    ([{"model": "big", "when": "x"}, ROUTES[1]], "different model"),
    ([{"model": "a", "when": "A hard project"}, ROUTES[1]], 'different "when"'),
])
def test_bad_routes_are_input_errors(routes, msg):
    with pytest.raises(InputError, match=msg):
        check_routes(routes)


# --- route_many ---------------------------------------------------------------------

def test_answer_and_scores_use_model_names(decider, fake_model):
    set_weights(fake_model, TASK, {"small": 9, "big": 1})
    r = route_many(decider, [TASK], ROUTES)[0]
    assert r.answer == "small"
    assert r.scores == {"small": 0.9, "big": 0.1}
    assert r.confidence == 0.9 and r.sure is True


def test_unsure_when_close(decider, fake_model):
    set_weights(fake_model, TASK, {"small": 6, "big": 4})
    r = route_many(decider, [TASK], ROUTES)[0]
    assert r.answer == "small" and r.sure is False


def test_defaults_used_when_no_routes_given(decider, fake_model):
    r = route_many(decider, [TASK])[0]
    assert set(r.scores) == {d["model"] for d in DEFAULT_ROUTES}


def test_results_in_task_order(decider, fake_model):
    set_weights(fake_model, "a", {"small": 9, "big": 1})
    set_weights(fake_model, "b", {"small": 1, "big": 9})
    assert [r.answer for r in route_many(decider, ["a", "b"], ROUTES)] == ["small", "big"]


# --- routes files and YN_ROUTES -----------------------------------------------------

def test_read_routes_file(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps(ROUTES))
    assert read_routes_file(str(p)) == ROUTES


@pytest.mark.parametrize("content", [None, "not json", json.dumps([ROUTES[0]])])
def test_bad_routes_file_is_input_error(tmp_path, content):
    p = tmp_path / "r.json"
    if content is not None:
        p.write_text(content)
    with pytest.raises(InputError):
        read_routes_file(str(p))


def test_yn_routes_env(tmp_path, monkeypatch):
    p = tmp_path / "r.json"
    p.write_text(json.dumps(ROUTES))
    monkeypatch.setenv("YN_ROUTES", str(p))
    assert default_routes() == ROUTES


def test_bad_yn_routes_env_is_setup_error_not_caller_error(tmp_path, monkeypatch):
    monkeypatch.setenv("YN_ROUTES", str(tmp_path / "missing.json"))
    with pytest.raises(RuntimeError, match="YN_ROUTES") as e:
        default_routes()
    assert not isinstance(e.value, InputError)


# --- CLI ----------------------------------------------------------------------------

def test_cli_route_plain_output(fake_model, tmp_path, capsys):
    p = tmp_path / "r.json"
    p.write_text(json.dumps(ROUTES))
    set_weights(fake_model, TASK, {"small": 9, "big": 1})
    assert main(["route", "--routes", str(p), TASK]) == 0
    assert capsys.readouterr().out == "small\t0.90\n"


def test_cli_route_unsure_exit_code(fake_model, tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps(ROUTES))
    set_weights(fake_model, TASK, {"small": 6, "big": 4})
    assert main(["route", "--routes", str(p), "--exit-code", TASK]) == EXIT_UNSURE


def test_cli_bad_routes_file_is_usage_error(fake_model, tmp_path, capsys):
    assert main(["route", "--routes", str(tmp_path / "nope.json"), TASK]) == EXIT_USAGE
    assert "can't read routes file" in capsys.readouterr().err


def test_cli_bad_yn_routes_env_is_setup_error(fake_model, tmp_path, monkeypatch):
    monkeypatch.setenv("YN_ROUTES", str(tmp_path / "nope.json"))
    assert main(["route", TASK]) == EXIT_ERROR


def test_whitespace_only_difference_in_model_is_duplicate():
    with pytest.raises(InputError, match="different model"):
        check_routes([{"model": " a ", "when": "x"}, {"model": "a", "when": "y"}])


def test_overlong_when_is_input_error():
    with pytest.raises(InputError, match='route 0: "when" is too long'):
        check_routes([{"model": "a", "when": "x" * 1001}, ROUTES[1]])


def test_overlong_when_in_yn_routes_is_setup_error(decider, tmp_path, monkeypatch):
    p = tmp_path / "r.json"
    p.write_text(json.dumps([{"model": "a", "when": "x" * 1001}, ROUTES[1]]))
    monkeypatch.setenv("YN_ROUTES", str(p))
    with pytest.raises(RuntimeError, match="YN_ROUTES") as e:
        route_many(decider, [TASK])
    assert not isinstance(e.value, InputError)


def test_token_limit_on_yn_routes_is_setup_error(decider, monkeypatch):
    """Statements can pass the character check but fail the token check (e.g. CJK)."""
    def too_long(self, statements):
        raise InputError("a claim or option is 900 tokens; the limit is 400. Shorten it.")
    monkeypatch.setattr("yn.model.Decider.check_statements", too_long)
    with pytest.raises(RuntimeError, match="YN_ROUTES: a claim or option is 900 tokens"):
        route_many(decider, [TASK])
    with pytest.raises(InputError):  # the same problem in caller-supplied routes
        route_many(decider, [TASK], ROUTES)


def test_cli_routes_flag_overrides_yn_routes(fake_model, tmp_path, monkeypatch, capsys):
    p = tmp_path / "r.json"
    p.write_text(json.dumps(ROUTES))
    monkeypatch.setenv("YN_ROUTES", str(tmp_path / "missing.json"))  # would error if read
    set_weights(fake_model, TASK, {"small": 9, "big": 1})
    assert main(["route", "--routes", str(p), TASK]) == 0
    assert capsys.readouterr().out == "small\t0.90\n"


def test_cli_overlong_when_in_yn_routes_exits_3(fake_model, tmp_path, monkeypatch):
    p = tmp_path / "r.json"
    p.write_text(json.dumps([{"model": "a", "when": "x" * 1001}, ROUTES[1]]))
    monkeypatch.setenv("YN_ROUTES", str(p))
    assert main(["route", TASK]) == EXIT_ERROR


@pytest.mark.parametrize("when, ok", [
    ("x" * 999, True),            # 1000 with the added period
    ("x" * 999 + ".", True),      # exactly 1000, already a statement
    ("x" * 1000, False),          # 1001 once the period is added
])
def test_length_limit_counts_the_added_period(when, ok):
    routes = [{"model": "a", "when": when}, ROUTES[1]]
    if ok:
        check_routes(routes)
    else:
        with pytest.raises(InputError, match="1001 characters with its final period"):
            check_routes(routes)


def test_1000_char_unpunctuated_when_in_yn_routes_is_setup_error(decider, tmp_path,
                                                                  monkeypatch):
    p = tmp_path / "r.json"
    p.write_text(json.dumps([{"model": "a", "when": "x" * 1000}, ROUTES[1]]))
    monkeypatch.setenv("YN_ROUTES", str(p))
    with pytest.raises(RuntimeError, match="YN_ROUTES") as e:
        route_many(decider, [TASK])
    assert not isinstance(e.value, InputError)
    assert main(["route", TASK]) == EXIT_ERROR
