"""`yn-mcp`: MCP server so Claude Code, Codex and other agents can use YN as a tool.

Runs over stdio (local only). Stdout carries the protocol, so never print to it.
"""

from __future__ import annotations

import functools
import sys
import threading

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from yn.model import InputError, get_decider
from yn.route import route_many

WHEN_TO_USE = (
    "Use for fast, cheap decisions over text, especially many items at once: "
    "filtering, flagging, sorting, routing. Do NOT use for questions that need "
    "reasoning, outside facts, maths or writing. If a result has sure=false, "
    "judge that item yourself."
)

mcp = MCPServer(
    "yn",
    instructions="YN is a small local decision model. " + WHEN_TO_USE,
    log_level="WARNING",
)


def _bad_input_to_tool_error(fn):
    """Show the agent why its request was rejected. Other errors stay generic, so a
    setup problem is never blamed on the caller."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except InputError as e:
            raise ToolError(str(e)) from e

    return wrapper


@mcp.tool(description="Is a statement true of the text? Returns answer 'true'/'false', "
          "confidence 0-1 and sure. Write `claim` as a plain statement, not a question, "
          "e.g. 'This email is spam.' " + WHEN_TO_USE)
@_bad_input_to_tool_error
def yn_check(input: str, claim: str) -> dict:
    return get_decider().check(input, claim).to_dict()


@mcp.tool(description="Pick the option that best fits the text. Returns answer, "
          "confidence, scores per option and sure. Options may be short labels "
          "('billing') or full statements ending in a period ('The customer wants a "
          "refund.'); statements work better. 2-50 options. " + WHEN_TO_USE)
@_bad_input_to_tool_error
def yn_decide(input: str, options: list[str]) -> dict:
    return get_decider().decide(input, options).to_dict()


@mcp.tool(description="yn_check over many texts in one call. Much faster than calling "
          "yn_check repeatedly. Returns one result per input, in order. " + WHEN_TO_USE)
@_bad_input_to_tool_error
def yn_check_batch(inputs: list[str], claim: str) -> dict:
    return {"results": [r.to_dict() for r in get_decider().check_many(inputs, claim)]}


@mcp.tool(description="yn_decide over many texts in one call. Much faster than calling "
          "yn_decide repeatedly. Returns one result per input, in order. " + WHEN_TO_USE)
@_bad_input_to_tool_error
def yn_decide_batch(inputs: list[str], options: list[str]) -> dict:
    return {"results": [r.to_dict() for r in get_decider().decide_many(inputs, options)]}


ROUTE_HELP = (
    "Returns answer (the model name), confidence, scores per model and sure. "
    "Default routes are the current Claude models, cheapest first: claude-haiku-4-5, "
    "claude-sonnet-5, claude-opus-5-5, claude-fable-5-1. Pass `routes` as a list of "
    '{"model": ..., "when": "a sentence describing tasks that model suits."} to use '
    "your own (e.g. Codex models). The pick is a suggestion: if sure=false, choose "
    "yourself; the stand-in model often confuses neighbouring tiers."
)


@mcp.tool(description="Suggest which AI model should handle a task. " + ROUTE_HELP)
@_bad_input_to_tool_error
def yn_route(task: str, routes: list[dict[str, str]] | None = None) -> dict:
    return route_many(get_decider(), [task], routes)[0].to_dict()


@mcp.tool(description="yn_route over many tasks in one call. Returns one result per "
          "task, in order. " + ROUTE_HELP)
@_bad_input_to_tool_error
def yn_route_batch(tasks: list[str], routes: list[dict[str, str]] | None = None) -> dict:
    return {"results": [r.to_dict() for r in route_many(get_decider(), tasks, routes)]}


def _warm_up() -> None:
    try:
        get_decider().load()
    except Exception as e:  # surfaced again on the first tool call
        print(f"yn-mcp: model load failed: {e}", file=sys.stderr)


def main() -> None:
    # Load the model in the background so the MCP handshake isn't delayed.
    threading.Thread(target=_warm_up, daemon=True).start()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
