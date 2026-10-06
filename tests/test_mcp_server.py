"""Unit tests for the gauge-mcp server with a fake model (no weights loaded).

Calls go through `MCPServer.call_tool`, the same path the stdio transport uses, so
argument validation and error wrapping are exercised as a client would see them.
"""

from __future__ import annotations

import json

import pytest
from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError

import gauge.mcp_server as server
from gauge.model import Decider

pytestmark = pytest.mark.anyio

CLAIM = "This email is spam."
DECISION_TOOLS = {"gauge_check", "gauge_decide", "gauge_check_batch", "gauge_decide_batch"}
ROUTE_TOOLS = {"gauge_route", "gauge_route_batch"}
TOOLS = DECISION_TOOLS | ROUTE_TOOLS


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def decider(fake_model, monkeypatch):
    d = Decider(model_name="fake-model", threshold=0.85)
    monkeypatch.setattr(server, "get_decider", lambda: d)
    return d


async def call(name: str, args: dict):
    result = await server.mcp.call_tool(name, args)
    assert result.is_error is False
    # Tools return a plain dict, so the SDK sends it as one JSON text block.
    assert len(result.content) == 1
    return json.loads(result.content[0].text)


# --- registration --------------------------------------------------------------------

async def test_exactly_six_tools_registered():
    tools = await server.mcp.list_tools()
    assert {t.name for t in tools} == TOOLS
    assert len(tools) == 6


@pytest.mark.parametrize("name, params", [
    ("gauge_check", {"input", "claim"}),
    ("gauge_decide", {"input", "options"}),
    ("gauge_check_batch", {"inputs", "claim"}),
    ("gauge_decide_batch", {"inputs", "options"}),
    ("gauge_route", {"task", "routes"}),
    ("gauge_route_batch", {"tasks", "routes"}),
])
async def test_tool_input_schema_matches_function_signature(name, params):
    # Guards the error-wrapping decorator: without functools.wraps the schema
    # would be (*args, **kwargs).
    tool = {t.name: t for t in await server.mcp.list_tools()}[name]
    assert set(tool.input_schema["properties"]) == params
    # `routes` is optional; everything else is required.
    assert set(tool.input_schema["required"]) == params - {"routes"}


async def test_every_tool_description_includes_when_to_use():
    for t in await server.mcp.list_tools():
        if t.name in DECISION_TOOLS:
            assert server.WHEN_TO_USE in t.description
        else:
            assert server.ROUTE_HELP in t.description


# --- successful calls ----------------------------------------------------------------

async def test_gauge_check_returns_decision_dict(decider, fake_model):
    fake_model.entail[("win now", CLAIM)] = fake_model.check_logit(0.9)
    out = await call("gauge_check", {"input": "win now", "claim": CLAIM})
    assert out == {"answer": "true", "confidence": 0.9, "scores": {"true": 0.9, "false": 0.1},
                   "sure": True, "model": "fake-model"}


async def test_gauge_decide_returns_chosen_option(decider, fake_model):
    fake_model.entail[("t", "This text is about b.")] = fake_model.decide_logit(3)
    out = await call("gauge_decide", {"input": "t", "options": ["a", "b"]})
    assert out["answer"] == "b"
    assert out["scores"] == {"a": 0.25, "b": 0.75}


async def test_gauge_check_batch_results_in_order(decider, fake_model):
    fake_model.entail[("x", CLAIM)] = fake_model.check_logit(0.1)
    fake_model.entail[("y", CLAIM)] = fake_model.check_logit(0.9)
    out = await call("gauge_check_batch", {"inputs": ["x", "y"], "claim": CLAIM})
    assert [r["answer"] for r in out["results"]] == ["false", "true"]


async def test_gauge_decide_batch_results_in_order(decider, fake_model):
    fake_model.entail[("x", "This text is about a.")] = fake_model.decide_logit(9)
    fake_model.entail[("y", "This text is about b.")] = fake_model.decide_logit(9)
    out = await call("gauge_decide_batch", {"inputs": ["x", "y"], "options": ["a", "b"]})
    assert [r["answer"] for r in out["results"]] == ["a", "b"]


async def test_batch_with_empty_inputs_returns_empty_results(decider):
    assert await call("gauge_check_batch", {"inputs": [], "claim": CLAIM}) == {"results": []}


# --- errors --------------------------------------------------------------------------

@pytest.mark.parametrize("name, args, message", [
    ("gauge_check", {"input": "", "claim": CLAIM}, "input must be a non-empty string"),
    ("gauge_check", {"input": "t", "claim": " "}, "claim must be a non-empty string"),
    ("gauge_decide", {"input": "t", "options": ["a"]}, "options must have 2-50 items, got 1"),
    ("gauge_decide", {"input": "t", "options": ["a", " a"]}, "options must be unique"),
    ("gauge_check_batch", {"inputs": ["ok", ""], "claim": CLAIM}, "input must be a non-empty string"),
    ("gauge_decide_batch", {"inputs": ["t"], "options": ["a", "a"]}, "options must be unique"),
])
async def test_value_error_surfaces_as_tool_error_with_reason(decider, name, args, message):
    with pytest.raises(ToolError) as e:
        await server.mcp.call_tool(name, args)
    # A deliberate ToolError, not a crash: crashes hide the reason from the agent.
    assert not isinstance(e.value, UnexpectedToolError)
    assert str(e.value) == f"Error executing tool {name}: {message}"


def test_decorated_function_raises_tool_error_when_called_directly(decider):
    with pytest.raises(ToolError, match="^input must be a non-empty string$") as e:
        server.gauge_check("", CLAIM)
    assert isinstance(e.value.__cause__, ValueError)


def test_decorated_function_returns_dict_when_called_directly(decider):
    assert server.gauge_check("t", CLAIM)["answer"] in {"true", "false"}


async def test_non_value_error_is_not_converted(decider, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(Decider, "check_many", boom)
    with pytest.raises(UnexpectedToolError) as e:
        await server.mcp.call_tool("gauge_check", {"input": "t", "claim": CLAIM})
    assert "model exploded" not in str(e.value)


async def test_wrong_argument_type_is_rejected_before_model(decider, fake_model):
    with pytest.raises(ToolError):
        await server.mcp.call_tool("gauge_decide", {"input": "t", "options": "a,b"})
    assert fake_model.calls == []


# --- routing tools --------------------------------------------------------------------

async def test_gauge_route_returns_model_name(decider, fake_model):
    routes = [{"model": "small", "when": "A quick edit."},
              {"model": "big", "when": "A hard project."}]
    fake_model.entail[("fix typo", "A quick edit.")] = fake_model.decide_logit(9)
    fake_model.entail[("fix typo", "A hard project.")] = fake_model.decide_logit(1)
    out = await call("gauge_route", {"task": "fix typo", "routes": routes})
    assert out["answer"] == "small" and out["scores"] == {"small": 0.9, "big": 0.1}


async def test_gauge_route_batch_defaults_to_claude_models(decider, fake_model):
    out = await call("gauge_route_batch", {"tasks": ["a", "b"]})
    assert len(out["results"]) == 2
    assert set(out["results"][0]["scores"]) == {
        "claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5-5", "claude-fable-5-1"}


async def test_gauge_route_bad_routes_is_tool_error(decider, fake_model):
    # The schema rejects a missing field before our code runs...
    with pytest.raises(ToolError, match="when") as e:
        await server.mcp.call_tool("gauge_route", {"task": "x", "routes": [{"model": "a"}, {"model": "b"}]})
    assert not isinstance(e.value, UnexpectedToolError)
    # ...and our own checks catch what the schema can't.
    dup = [{"model": "a", "when": "Same."}, {"model": "b", "when": "Same."}]
    with pytest.raises(ToolError, match='different "when"') as e:
        await server.mcp.call_tool("gauge_route", {"task": "x", "routes": dup})
    assert not isinstance(e.value, UnexpectedToolError)


async def test_route_schema_describes_model_and_when():
    tool = {t.name: t for t in await server.mcp.list_tools()}["gauge_route"]
    text = str(tool.input_schema)
    assert "model" in text and "when" in text


@pytest.mark.parametrize("routes_file", [
    None,  # missing file
    [{"model": "a", "when": "x" * 1200}, {"model": "b", "when": "y"}],  # over the limit
])
async def test_bad_gauge_routes_is_not_blamed_on_caller(decider, fake_model, tmp_path,
                                                     monkeypatch, routes_file):
    p = tmp_path / "routes.json"
    if routes_file is not None:
        p.write_text(json.dumps(routes_file))
    monkeypatch.setenv("GAUGE_ROUTES", str(p))
    with pytest.raises(UnexpectedToolError):
        await server.mcp.call_tool("gauge_route", {"task": "fix typo"})


async def test_route_with_unknown_key_is_rejected_like_the_cli(decider, fake_model):
    routes = [{"model": "a", "when": "A.", "extra": 1}, {"model": "b", "when": "B."}]
    with pytest.raises(ToolError, match="extra") as e:
        await server.mcp.call_tool("gauge_route", {"task": "x", "routes": routes})
    assert not isinstance(e.value, UnexpectedToolError)
